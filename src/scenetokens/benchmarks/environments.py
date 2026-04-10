r"""Benchmark creation for the Environments benchmark.

Clusters scenarios by road topology using NetLSD graph descriptors and KMeans, then assigns train/validation/testing
splits based on cluster hardness (silhouette score). Results are written as a CSV file alongside model artifacts and
cluster visualizations.

Example usage:

    uv run -m scenetokens.create_benchmark benchmark=environments \\
        input_data_path=/datasets/waymo/processed/mini_causal \\
        output_data_path=/datasets/waymo/processed/environment_benchmark

See configs/benchmark/environments.yaml for all available options.
"""

import functools
import multiprocessing
import pickle  # nosec B403
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import netlsd
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from scenetokens.benchmarks.common import copy_scenario, create_split_dirs, get_scenario_mapping
from scenetokens.utils.map_utils import map_infos_to_graph, simplify_graph, visualize_scenario_graph


def _compute_graph_descriptor(
    filepath: Path,
    *,
    ego_centered: bool = False,
    k_polylines: int = 100,
    map_range: float = 100.0,
) -> tuple[str, str, NDArray[np.float64]] | None:
    """Loads a scenario pickle file, builds its map graph, and computes a NetLSD descriptor.

    When ego_centered is True the graph is filtered to match base_dataset.py's pipeline: polyline points are
    transformed to the ego-centric frame (translate + rotate by heading), filtered by the L∞ range box, and the
    top-K elements by average point distance are kept.

    Args:
        filepath: Path to the scenario pickle file.
        ego_centered: If True, restrict the graph to map elements within range of the ego agent.
        k_polylines: Maximum number of map elements to retain when ego_centered is True. Defaults to 100.
        map_range: L∞ half-width of the ego-centric range box in metres. Defaults to 100.0.

    Returns:
        A tuple of (scenario_id, split, descriptor) or None on failure.
    """
    if not filepath.exists():
        return None

    try:
        with filepath.open("rb") as f:
            scenario = pickle.load(f)  # nosec B301
    except (OSError, pickle.UnpicklingError):
        return None

    scenario_id = scenario.get("scenario_id")
    if scenario_id is None:
        return None

    ego_xy: NDArray[np.float64] | None = None
    ego_heading: float = 0.0
    if ego_centered:
        sdc_track_index = scenario["sdc_track_index"]
        curr_time_index = scenario["current_time_index"]
        trajs = scenario["track_infos"]["trajs"][sdc_track_index, curr_time_index]
        ego_xy = trajs[:2]
        ego_heading = float(trajs[6])

    map_infos = scenario.get("map_infos", {})
    graph = simplify_graph(
        map_infos_to_graph(
            map_infos,
            ref_xy=ego_xy,
            k_polylines=k_polylines if ego_centered else None,
            ref_heading=ego_heading,
            map_range=map_range,
        )
    )

    if graph.number_of_nodes() == 0:
        return None

    descriptor: NDArray[np.float64] = netlsd.heat(graph)
    split = filepath.parent.name

    return scenario_id, split, descriptor


def _load_descriptor_cache(cache_path: Path) -> dict[str, tuple[str, str, NDArray[np.float64]]]:
    """Loads the descriptor cache from disk.

    The cache maps absolute filepath strings to (scenario_id, split, descriptor) tuples. Returns an empty dict if the
    cache file does not exist or cannot be read.

    Args:
        cache_path: Path to the cache pickle file.

    Returns:
        Dict mapping filepath string to (scenario_id, split, descriptor).
    """
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("rb") as f:
            return pickle.load(f)  # nosec B301
    except (OSError, pickle.UnpicklingError):
        return {}


def _save_descriptor_cache(cache: dict[str, tuple[str, str, NDArray[np.float64]]], cache_path: Path) -> None:
    """Saves the descriptor cache to disk.

    Args:
        cache: Dict mapping filepath string to (scenario_id, split, descriptor).
        cache_path: Destination path for the cache pickle file.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cache, f)


def _compute_descriptors_with_cache(  # noqa: PLR0913
    filepaths: list[Path],
    cache_path: Path,
    ego_centered: bool,  # noqa: FBT001
    k_polylines: int,
    num_workers: int,
    root_path: Path | None = None,
    overwrite: bool = False,  # noqa: FBT001, FBT002
    map_range: float = 100.0,
) -> tuple[list[str], list[str], NDArray[np.float64]]:
    """Computes NetLSD descriptors for a list of scenario files, using a disk cache for already-computed results.

    Descriptors for filepaths not present in the cache are computed (in parallel if num_workers > 0) and added to the
    cache. The updated cache is saved to disk before returning.

    The ``input_set`` label for each scenario is derived as the path of its parent directory relative to
    ``root_path`` (e.g. ``training/shard_0``). When ``root_path`` is None the immediate parent directory name is
    used as a fallback.

    Args:
        filepaths: List of scenario file paths.
        cache_path: Path to the descriptor cache pickle file.
        ego_centered: Passed through to `_compute_graph_descriptor`.
        k_polylines: Passed through to `_compute_graph_descriptor`.
        num_workers: Number of parallel workers (0 = single process).
        root_path: Root of the input dataset tree. When provided, each scenario's split label is set to the relative
            path from this root to the scenario's parent directory (e.g. ``training/shard_0``).
        overwrite: If True, evict all entries for the given filepaths from the cache before computing, forcing
            recomputation even if descriptors were previously cached. Defaults to False.
        map_range: Passed through to `_compute_graph_descriptor`. Defaults to 100.0.

    Returns:
        Tuple of (scenario_ids, splits, descriptor_matrix) for all valid filepaths.

    Raises:
        ValueError: If no valid descriptors could be computed or retrieved.
    """
    cache = _load_descriptor_cache(cache_path)

    if overwrite:
        for fp in filepaths:
            cache.pop(str(fp), None)

    uncached = [fp for fp in filepaths if str(fp) not in cache]
    if uncached:
        print(f"Computing descriptors for {len(uncached)} scenarios ({len(filepaths) - len(uncached)} cached)...")
        worker_fn = functools.partial(
            _compute_graph_descriptor, ego_centered=ego_centered, k_polylines=k_polylines, map_range=map_range
        )
        if num_workers == 0:
            new_results = [worker_fn(fp) for fp in tqdm(uncached, desc="Encoding map graphs")]
        else:
            with multiprocessing.Pool(num_workers) as pool:
                new_results = list(
                    tqdm(
                        pool.imap_unordered(worker_fn, uncached),
                        total=len(uncached),
                        desc="Encoding map graphs",
                    )
                )
        for fp, result in zip(uncached, new_results, strict=False):
            if result is not None:
                cache[str(fp)] = result
        _save_descriptor_cache(cache, cache_path)
        print(f"Descriptor cache updated at {cache_path}")
    else:
        print(f"All {len(filepaths)} descriptors loaded from cache.")

    def _split_label(fp: Path) -> str:
        if root_path is not None:
            try:
                return str(fp.parent.relative_to(root_path))
            except ValueError:
                pass
        return fp.parent.name

    valid_entries = [(fp, cache[str(fp)]) for fp in filepaths if str(fp) in cache]
    if not valid_entries:
        error_message = "No valid scenario descriptors were computed. Check the input data path."
        raise ValueError(error_message)

    scenario_ids = [entry[0] for _, entry in valid_entries]
    splits = [_split_label(fp) for fp, _ in valid_entries]
    descriptors = [entry[2] for _, entry in valid_entries]
    return scenario_ids, splits, np.stack(descriptors)


def select_train_test_splits(
    clusters_df: pd.DataFrame,
    silhouette_scores: NDArray[np.float64],
    target_train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Assigns each scenario to an output split (training/validation/testing) based on cluster hardness.

    Clusters are ranked by their mean silhouette score (ascending). The hardest clusters (lowest mean silhouette,
    i.e. most ambiguous) are designated as the test set. Clusters are greedily added to the test set until the test
    fraction reaches approximately (1 - target_train_ratio) of all scenarios.

    The remaining (non-test) scenarios are randomly split into validation and training according to validation_ratio.

    Args:
        clusters_df: DataFrame with columns: scenario_id, split, cluster.
        silhouette_scores: Per-sample silhouette scores aligned with clusters_df rows.
        target_train_ratio: Desired fraction of all scenarios in the training+validation output sets. Defaults to 0.80.
        validation_ratio: Fraction of non-test scenarios assigned to validation. Defaults to 0.10.
        rng: NumPy random generator for reproducible shuffling. If None, uses np.random.shuffle.

    Returns:
        Copy of clusters_df with added columns: silhouette_score, input_set, output_set.
    """
    df = clusters_df.copy()
    df["silhouette_score"] = silhouette_scores
    df = df.rename(columns={"split": "input_set"})

    total = len(df)
    target_test_count = total * (1.0 - target_train_ratio)

    cluster_stats = (
        df.groupby("cluster")
        .agg(mean_silhouette=("silhouette_score", "mean"), size=("silhouette_score", "count"))
        .reset_index()
        .sort_values("mean_silhouette", ascending=True)
    )

    test_clusters: set[int] = set()
    accumulated = 0
    for _, row in cluster_stats.iterrows():
        if accumulated >= target_test_count:
            break
        test_clusters.add(int(row["cluster"]))
        accumulated += int(row["size"])

    non_test_indices = df.index[~df["cluster"].isin(test_clusters)].tolist()
    if rng is not None:
        rng.shuffle(non_test_indices)
    else:
        np.random.default_rng().shuffle(non_test_indices)
    num_val = int(len(non_test_indices) * validation_ratio)
    val_indices: set[int] = set(non_test_indices[:num_val])

    def _assign_output_set(row: pd.Series) -> str:  # pyright: ignore[reportMissingTypeArgument]
        if row["cluster"] in test_clusters:
            return "testing"
        if row.name in val_indices:
            return "validation"
        return "training"

    df["output_set"] = df.apply(_assign_output_set, axis=1)
    return df


def visualize_clusters(  # noqa: PLR0913, PLR0915
    clusters_df: pd.DataFrame,
    descriptor_matrix: NDArray[np.float64],
    input_data_path: Path,
    output_data_path: Path,
    n_examples: int = 30,
    seed: int = 42,
    test_clusters: set[int] | None = None,
    ego_centered: bool = False,  # noqa: FBT001, FBT002
    k_polylines: int = 100,
    map_range: float = 100.0,
    reduction: str = "pca",
) -> None:
    """Saves per-cluster graph visualizations and a 2-D scatter plot of all descriptors.

    For each cluster, up to `n_examples` scenario graphs are rendered as PNGs under
    `output_data_path/cluster_results/cluster_<id>/`. A single scatter plot coloured by cluster label is saved as
    `output_data_path/cluster_results/cluster_scatter.png`. When clusters_df contains a ``silhouette_score`` column,
    per-cluster mean silhouette scores are annotated in the legend. Test clusters are marked with a distinct marker
    style.

    Args:
        clusters_df: DataFrame with columns scenario_id, cluster, and optionally silhouette_score and output_set.
        descriptor_matrix: Array of shape (N, D) — one descriptor row per scenario.
        input_data_path: Root path of the processed scenario data.
        output_data_path: Directory where cluster results will be written.
        n_examples: Number of example graphs per cluster. Defaults to 30.
        seed: Random seed for example sampling. Defaults to 42.
        test_clusters: Optional set of cluster IDs designated as the test set; shown with a distinct marker.
        ego_centered: If True, each scenario graph is filtered using the ego-centric L∞ range filter, matching the
            encoded graph. Defaults to False.
        k_polylines: Maximum number of map elements to retain when ego_centered is True. Defaults to 100.
        map_range: L∞ half-width of the ego-centric range box in metres. Defaults to 100.0.
        reduction: Dimensionality reduction algorithm for the scatter plot. ``"pca"`` (default) or ``"tsne"``.
    """
    rng = np.random.default_rng(seed)
    cluster_results_path = output_data_path / "cluster_results"

    filepath_map = {fp.stem: fp for fp in input_data_path.rglob("*.pkl") if "infos" not in fp.stem}

    split_col = "input_set" if "input_set" in clusters_df.columns else "split"
    for cluster_id, group in clusters_df.groupby("cluster"):
        cluster_dir = cluster_results_path / f"cluster_{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        scenario_ids = group["scenario_id"].tolist()
        n_chosen = min(n_examples, len(scenario_ids))
        chosen = rng.choice(scenario_ids, size=n_chosen, replace=False).tolist()

        for scenario_id in tqdm(chosen, desc=f"Visualizing cluster {cluster_id}", leave=False):
            filepath = filepath_map.get(str(scenario_id))
            if filepath is not None:
                visualize_scenario_graph(
                    filepath, cluster_dir, ego_centered=ego_centered, k_polylines=k_polylines, map_range=map_range
                )

    del split_col
    print(f"Saved per-cluster graph examples to {cluster_results_path}")

    cluster_mean_sil: dict[Any, float] = {}
    if "silhouette_score" in clusters_df.columns:
        cluster_mean_sil = clusters_df.groupby("cluster")["silhouette_score"].mean().to_dict()

    labels = clusters_df["cluster"].to_numpy()
    n_clusters = int(labels.max()) + 1
    cmap = get_cmap("tab20", n_clusters)

    if reduction == "tsne":
        coords = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(
            descriptor_matrix
        )
        xlabel, ylabel, title = "t-SNE 1", "t-SNE 2", "Environment clusters (t-SNE of NetLSD descriptors)"
    else:
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(descriptor_matrix)
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)"
        title = "Environment clusters (PCA of NetLSD descriptors)"

    fig, ax = plt.subplots(figsize=(10, 8))

    if test_clusters:
        train_mask = np.array([lbl not in test_clusters for lbl in labels])
        test_mask = ~train_mask
        if train_mask.any():
            ax.scatter(
                coords[train_mask, 0],
                coords[train_mask, 1],
                c=labels[train_mask],
                cmap=cmap,
                vmin=0,
                vmax=n_clusters - 1,
                s=12,
                alpha=0.6,
                linewidths=0,
                marker=MarkerStyle("o"),
                label="train/val",
            )
        if test_mask.any():
            ax.scatter(
                coords[test_mask, 0],
                coords[test_mask, 1],
                c=labels[test_mask],
                cmap=cmap,
                vmin=0,
                vmax=n_clusters - 1,
                s=20,
                alpha=0.8,
                linewidths=0.5,
                edgecolors="black",
                marker=MarkerStyle("x"),
                label="test",
            )
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, s=10, alpha=0.6, linewidths=0)
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(n_clusters))
        cbar.set_label("Cluster", fontsize=10)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)

    if cluster_mean_sil:
        legend_handles = []
        for c, mean_sil in sorted(cluster_mean_sil.items()):
            color = cmap(int(c) / max(n_clusters - 1, 1))
            is_test = test_clusters is not None and c in test_clusters
            marker = "x" if is_test else "o"
            tag = " [test]" if is_test else ""
            handle = Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=7,
                label=f"C{c}{tag}  sil={mean_sil:.3f}",
            )
            legend_handles.append(handle)
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, framealpha=0.8)

    fig.tight_layout()
    scatter_path = cluster_results_path / "cluster_scatter.png"
    fig.savefig(str(scatter_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot to {scatter_path}")


def create_environments_benchmark(config: DictConfig) -> None:  # noqa: PLR0915
    """Clusters scenarios by road topology using NetLSD graph descriptors and KMeans.

    The pipeline runs in two phases:

    1. **Fit phase** — A random sample of P% of all scenarios (or `num_scenarios` if specified) is used to fit a
       StandardScaler and KMeans model. Silhouette scores on this sample drive cluster-hardness ranking.
    2. **Assign phase** — The remaining scenarios are embedded using the cached descriptors and assigned to the
       nearest cluster using the fitted model.

    After clustering, `select_train_test_splits` designates the hardest clusters (lowest mean silhouette) as the test
    set so that the test split is more challenging. The results are saved as `environment_benchmark.csv`.

    Args:
        config: Hydra config.
            Expected keys: input_data_path, output_data_path, cache_path, n_clusters, n_examples, sample_percentage,
            num_scenarios, num_workers, ego_centered, k_polylines, seed, overwrite, map_range, reduction,
            validation_ratio.

    Raises:
        ValueError: If no valid scenario descriptors could be computed.
    """
    input_data_path = Path(config.input_data_path)
    cache_path = Path(config.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    output_path = Path(config.output_data_path)
    output_path.mkdir(parents=True, exist_ok=True)
    descriptor_filepath = Path(cache_path) / "descriptors_cache.pkl"

    all_filepaths: list[Path] = [fp for fp in input_data_path.rglob("*.pkl") if "infos" not in fp.stem]
    print(f"Found {len(all_filepaths)} scenario files.")

    num_samples = (
        config.num_scenarios
        if config.num_scenarios is not None
        else max(1, round(len(all_filepaths) * config.sample_percentage))
    )
    num_samples = min(num_samples, len(all_filepaths))
    print(f"Using {num_samples} scenarios ({num_samples / len(all_filepaths):.1%}) to fit the model.")

    sample_indices = rng.choice(len(all_filepaths), size=num_samples, replace=False)
    sample_mask = np.zeros(len(all_filepaths), dtype=bool)
    sample_mask[sample_indices] = True
    sample_filepaths = [fp for fp, m in zip(all_filepaths, sample_mask, strict=False) if m]
    remaining_filepaths = [fp for fp, m in zip(all_filepaths, sample_mask, strict=False) if not m]

    print(f"\n[Phase 1] Computing descriptors for {len(sample_filepaths)} sample scenarios...")
    sample_ids, sample_splits, sample_descriptors = _compute_descriptors_with_cache(
        sample_filepaths,
        descriptor_filepath,
        config.ego_centered,
        config.k_polylines,
        config.num_workers,
        root_path=input_data_path,
        overwrite=config.overwrite,
        map_range=config.map_range,
    )

    print(f"Fitting StandardScaler and KMeans({config.n_clusters}) on sample...")
    scaler = StandardScaler()
    sample_scaled = scaler.fit_transform(sample_descriptors)
    kmeans = KMeans(n_clusters=config.n_clusters, random_state=config.seed, n_init="auto")
    sample_labels = kmeans.fit_predict(sample_scaled)

    all_ids = list(sample_ids)
    all_splits = list(sample_splits)
    all_scaled = list(sample_scaled)
    all_labels = list(sample_labels)

    if remaining_filepaths:
        print(f"\n[Phase 2] Computing descriptors for {len(remaining_filepaths)} remaining scenarios...")
        rem_ids, rem_splits, rem_descriptors = _compute_descriptors_with_cache(
            remaining_filepaths,
            descriptor_filepath,
            config.ego_centered,
            config.k_polylines,
            config.num_workers,
            root_path=input_data_path,
            overwrite=config.overwrite,
            map_range=config.map_range,
        )
        rem_scaled = scaler.transform(rem_descriptors)
        rem_labels = kmeans.predict(rem_scaled)

        all_ids += list(rem_ids)
        all_splits += list(rem_splits)
        all_scaled += list(rem_scaled)
        all_labels += list(rem_labels)

    all_scaled_matrix = np.stack(all_scaled)
    all_labels_array = np.array(all_labels)

    print(f"\nComputing silhouette scores for {len(all_ids)} scenarios...")
    sil_scores: NDArray[np.float64] = np.asarray(silhouette_samples(all_scaled_matrix, all_labels_array))

    clusters_df = pd.DataFrame({"scenario_id": all_ids, "split": all_splits, "cluster": all_labels_array})
    benchmark_df = select_train_test_splits(clusters_df, sil_scores, validation_ratio=config.validation_ratio, rng=rng)

    output_df = benchmark_df[["scenario_id", "cluster", "silhouette_score", "input_set", "output_set"]].rename(
        columns={"cluster": "cluster_label"}
    )
    benchmark_csv_path = cache_path / "environment_benchmark.csv"
    output_df.to_csv(benchmark_csv_path, index=False)
    print(f"\nSaved benchmark split assignments to {benchmark_csv_path}")

    kmeans_path = cache_path / "kmeans_model.pkl"
    with kmeans_path.open("wb") as f:
        pickle.dump(kmeans, f)

    scaler_path = cache_path / "scaler.pkl"
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved KMeans model to {kmeans_path}")
    print(f"Saved StandardScaler to {scaler_path}")

    cluster_stats = (
        benchmark_df.groupby("cluster")
        .agg(size=("scenario_id", "count"), mean_sil=("silhouette_score", "mean"))
        .sort_index()
    )
    test_cluster_ids = set(benchmark_df[benchmark_df["output_set"] == "testing"]["cluster"].unique())
    print("\nCluster summary:")
    for cluster_id, row in cluster_stats.iterrows():
        tag = " [TEST]" if cluster_id in test_cluster_ids else ""
        print(f"  Cluster {cluster_id}: {int(row['size'])} scenarios, mean_sil={row['mean_sil']:.3f}{tag}")

    split_counts = benchmark_df["output_set"].value_counts()
    print("\nOutput split distribution:")
    for split_name, count in split_counts.items():
        print(f"  {split_name}: {count} ({count / len(benchmark_df):.1%})")

    descriptor_cache = _load_descriptor_cache(descriptor_filepath)
    id_to_filepath: dict[str, Path] = {v[0]: Path(k) for k, v in descriptor_cache.items()}
    input_scenario_mapping = {sid: id_to_filepath[sid] for sid in benchmark_df["scenario_id"] if sid in id_to_filepath}

    create_split_dirs(output_path)

    output_scenario_mapping: dict[str, Path] = {}
    for split in ["training", "validation", "testing"]:
        split_ids = benchmark_df[benchmark_df["output_set"] == split]["scenario_id"].tolist()
        output_scenario_mapping.update(get_scenario_mapping(split_ids, output_path, split))

    with multiprocessing.Pool(config.num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    functools.partial(
                        copy_scenario,
                        input_scenario_mapping=input_scenario_mapping,
                        output_scenario_mapping=output_scenario_mapping,
                    ),
                    list(output_scenario_mapping.keys()),
                ),
                total=len(output_scenario_mapping),
                desc="Copying scenarios",
            )
        )

    cluster_results_path = cache_path / "cluster_results"
    if cluster_results_path.exists():
        shutil.rmtree(cluster_results_path)

    sample_cluster_df = benchmark_df.iloc[: len(sample_ids)].copy()
    visualize_clusters(
        sample_cluster_df,
        np.stack(sample_scaled),  # pyright: ignore[reportArgumentType, reportCallIssue]
        input_data_path,
        cache_path,
        config.n_examples,
        config.seed,
        test_clusters=test_cluster_ids,
        ego_centered=config.ego_centered,
        k_polylines=config.k_polylines,
        map_range=config.map_range,
        reduction=config.reduction,
    )
