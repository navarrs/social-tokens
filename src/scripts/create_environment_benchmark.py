import functools
import multiprocessing
import operator
import pickle  # nosec B403
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import netlsd
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# Color scheme for each map element type in graph visualizations.
_NODE_COLORS: dict[str, str] = {
    "lane": "#4A90D9",
    "road_line": "#F5A623",
    "road_edge": "#7B68EE",
    "crosswalk": "#50C878",
    "speed_bump": "#FF6B6B",
    "stop_sign": "#FF4500",
}
_DEFAULT_NODE_COLOR = "#AAAAAA"
_MAP_ELEMENT_TYPES = ("lane", "road_line", "road_edge", "crosswalk", "speed_bump", "stop_sign")


def _filter_map_elements_by_proximity(
    map_infos: dict[str, Any],
    all_polylines: NDArray[np.float64],
    ego_xy: NDArray[np.float64],
    k: int,
) -> dict[str, Any]:
    """Returns a shallow copy of map_infos keeping only the K elements whose polyline centroids are closest to ego_xy.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.
        all_polylines: Array of shape (N, >=2) with all polyline points.
        ego_xy: Ego position (x, y) at the current time step.
        k: Number of closest map elements to retain.

    Returns:
        Updated map_infos dict with each element list filtered to the K nearest elements.
    """
    candidates: list[tuple[float, str, dict]] = []
    for etype in _MAP_ELEMENT_TYPES:
        for element in map_infos.get(etype, []):
            start, end = element.get("polyline_index", (0, 0))
            if all_polylines.shape[0] > 0 and end > start:
                centroid_xy = all_polylines[start:end, :2].mean(axis=0)
                dist = float(np.linalg.norm(centroid_xy - ego_xy))
            else:
                dist = float("inf")
            candidates.append((dist, etype, element))

    candidates.sort(key=operator.itemgetter(0))
    filtered: dict[str, list] = {etype: [] for etype in _MAP_ELEMENT_TYPES}
    for _, etype, element in candidates[:k]:
        filtered[etype].append(element)

    return {**map_infos, **filtered}


def _map_infos_to_graph(
    map_infos: dict[str, Any],
    ego_xy: NDArray[np.float64] | None = None,
    k_polylines: int | None = None,
) -> nx.DiGraph:
    """Converts scenario map information into a directed NetworkX graph.

    Nodes represent lanes, road lines, road edges, crosswalks, speed bumps, and stop signs. Directed edges connect
    lanes via their entry/exit lane relationships. Each node stores the centroid (x, y, z) of its polyline as
    position attributes when available.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.
        ego_xy: If provided together with k_polylines, restricts the graph to the K map elements whose polyline
            centroids are closest to this (x, y) position.
        k_polylines: Number of closest map elements to retain when ego_xy is set.

    Returns:
        nx.DiGraph: Directed graph representing the road topology.
    """
    graph = nx.DiGraph()
    all_polylines: NDArray[np.float64] = np.asarray(map_infos.get("all_polylines", np.empty((0, 7))))

    if ego_xy is not None and k_polylines is not None:
        map_infos = _filter_map_elements_by_proximity(map_infos, all_polylines, ego_xy, k_polylines)

    def _node_pos(element: dict[str, Any]) -> dict[str, float]:
        start, end = element.get("polyline_index", (0, 0))
        if all_polylines.shape[0] > 0 and end > start:
            centroid = all_polylines[start:end, :3].mean(axis=0)
            return {"x": float(centroid[0]), "y": float(centroid[1]), "z": float(centroid[2])}
        return {}

    # Add lane nodes with attributes and connectivity edges.
    for lane in map_infos.get("lane", []):
        lane_id = lane["id"]
        graph.add_node(lane_id, type="lane", speed_limit_mph=lane.get("speed_limit_mph", 0.0), **_node_pos(lane))
        for entry_id in lane.get("entry_lanes", []):
            graph.add_edge(entry_id, lane_id)
        for exit_id in lane.get("exit_lanes", []):
            graph.add_edge(lane_id, exit_id)

    # Add non-lane map element nodes (enrich structural encoding without connectivity).
    for road_line in map_infos.get("road_line", []):
        graph.add_node(road_line["id"], type="road_line", **_node_pos(road_line))

    for road_edge in map_infos.get("road_edge", []):
        graph.add_node(road_edge["id"], type="road_edge", **_node_pos(road_edge))

    for crosswalk in map_infos.get("crosswalk", []):
        graph.add_node(crosswalk["id"], type="crosswalk", **_node_pos(crosswalk))

    for speed_bump in map_infos.get("speed_bump", []):
        graph.add_node(speed_bump["id"], type="speed_bump", **_node_pos(speed_bump))

    for stop_sign in map_infos.get("stop_sign", []):
        graph.add_node(stop_sign["id"], type="stop_sign", **_node_pos(stop_sign))

    return graph


def _simplify_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Returns a simplified copy of the road topology graph for more stable encoding.

    Two simplifications are applied in order:

    1. **Remove isolated nodes** — nodes with no edges (degree zero) do not contribute to the graph's Laplacian
        structure but inflate the heat-trace baseline, causing NetLSD descriptors to vary with the number of
        non-connected map elements rather than road topology.
    2. **Keep the largest weakly-connected component** — small disconnected lane fragments at the scene boundary add
        noise without representing the main road network.

    Args:
        graph: Full road topology graph produced by `_map_infos_to_graph`.

    Returns:
        Simplified directed graph containing only the largest connected lane subgraph.
    """
    graph = graph.copy()
    graph.remove_nodes_from(list(nx.isolates(graph)))

    if graph.number_of_nodes() == 0:
        return graph

    largest_wcc = max(nx.weakly_connected_components(graph), key=len)
    return nx.DiGraph(graph.subgraph(largest_wcc))


def _build_positioned_graph(
    map_infos: dict[str, Any],
    ego_xy: NDArray[np.float64] | None = None,
    k_polylines: int | None = None,
) -> tuple[nx.DiGraph, dict[Any, NDArray[np.float64]]]:
    """Builds a map graph and extracts a matplotlib-compatible position dict from node attributes.

    Reads (x, y) from node attributes set by `_map_infos_to_graph`. When ego_xy and k_polylines are provided the
    graph is filtered to the K nearest map elements and simplified, matching the graph that was passed to NetLSD
    during encoding. Nodes without position attributes are placed via a spring layout fallback.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.
        ego_xy: Ego position (x, y). When provided together with k_polylines, restricts the graph to the K nearest
            map elements and applies `_simplify_graph`, reproducing the encoding graph exactly.
        k_polylines: Number of closest map elements to retain when ego_xy is set.

    Returns:
        The directed graph and a dict mapping node id to (x, y) position.
    """
    graph = _simplify_graph(_map_infos_to_graph(map_infos, ego_xy=ego_xy, k_polylines=k_polylines))

    pos: dict[Any, NDArray[np.float64]] = {
        n: np.array([data["x"], data["y"]]) for n, data in graph.nodes(data=True) if "x" in data and "y" in data
    }

    # Fall back to spring layout for any nodes that lack position attributes.
    missing = [n for n in graph.nodes if n not in pos]
    if missing:
        if pos:
            fallback = nx.spring_layout(graph, pos=dict(pos), fixed=list(pos.keys()), seed=0)
        else:
            fallback = nx.spring_layout(graph, seed=0)
        pos.update({n: fallback[n] for n in missing})

    return graph, pos


def _compute_graph_descriptor(
    filepath: Path,
    *,
    ego_centered: bool = False,
    k_polylines: int = 100,
) -> tuple[str, str, NDArray[np.float64]] | None:
    """Loads a scenario pickle file, builds its map graph, and computes a NetLSD descriptor.

    Args:
        filepath: Path to the scenario pickle file.
        ego_centered: If True, restrict the graph to the K map elements closest to the ego agent's position at the
            current time step.
        k_polylines: Number of closest map elements to retain when ego_centered is True. Defaults to 100.

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
    if ego_centered:
        sdc_track_index = scenario["sdc_track_index"]
        curr_time_index = scenario["current_time_index"]
        trajs = scenario["track_infos"]["trajs"][sdc_track_index, curr_time_index]
        ego_xy = trajs[:2]  # x, y

    map_infos = scenario.get("map_infos", {})
    graph = _simplify_graph(
        _map_infos_to_graph(map_infos, ego_xy=ego_xy, k_polylines=k_polylines if ego_centered else None)
    )

    if graph.number_of_nodes() == 0:
        return None

    descriptor: NDArray[np.float64] = netlsd.heat(graph)

    # The split is the parent directory name (e.g. training/validation/testing).
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
) -> tuple[list[str], list[str], NDArray[np.float64]]:
    """Computes NetLSD descriptors for a list of scenario files, using a disk cache for already-computed results.

    Descriptors for filepaths not present in the cache are computed (in parallel if num_workers > 0) and added to the
    cache. The updated cache is saved to disk before returning.

    The ``input_set`` label for each scenario is derived as the path of its parent directory relative to
    ``root_path`` (e.g. ``training/shard_0``). When ``root_path`` is None the immediate parent directory name is
    used as a fallback.

    Args:
        filepaths: List of paths to scenario pickle files.
        cache_path: Path to the descriptor cache pickle file.
        ego_centered: Passed through to `_compute_graph_descriptor`.
        k_polylines: Passed through to `_compute_graph_descriptor`.
        num_workers: Number of parallel workers (0 = single process).
        root_path: Root of the input dataset tree. When provided, each scenario's split label is set to the relative
            path from this root to the scenario's parent directory (e.g. ``training/shard_0``).
        overwrite: If True, evict all entries for the given filepaths from the cache before computing, forcing
            recomputation even if descriptors were previously cached. Defaults to False.

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
        worker_fn = functools.partial(_compute_graph_descriptor, ego_centered=ego_centered, k_polylines=k_polylines)
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
) -> pd.DataFrame:
    """Assigns each scenario to an output split (training/validation/testing) based on cluster hardness.

    Clusters are ranked by their mean silhouette score (ascending). The hardest clusters (lowest mean silhouette,
    i.e. most ambiguous) are designated as the test set. Clusters are greedily added to the test set until the test
    fraction reaches approximately (1 - target_train_ratio) of all scenarios.

    Within train/val clusters, each scenario's output split mirrors its original input split:
    - "training" → "training"
    - "validation" → "validation"
    - "testing" (originally in a test folder but now in a train cluster) → "training"

    Args:
        clusters_df: DataFrame with columns: scenario_id, split, cluster.
        silhouette_scores: Per-sample silhouette scores aligned with clusters_df rows.
        target_train_ratio: Desired fraction of all scenarios in the training+validation output sets. Defaults to 0.80.

    Returns:
        Copy of clusters_df with added columns: silhouette_score, input_set, output_set.
    """
    df = clusters_df.copy()
    df["silhouette_score"] = silhouette_scores
    df = df.rename(columns={"split": "input_set"})

    total = len(df)
    target_test_count = total * (1.0 - target_train_ratio)

    # Compute per-cluster stats.
    cluster_stats = (
        df.groupby("cluster")
        .agg(mean_silhouette=("silhouette_score", "mean"), size=("silhouette_score", "count"))
        .reset_index()
        .sort_values("mean_silhouette", ascending=True)  # hardest clusters first
    )

    # Greedily assign hardest clusters to test until target_test_count is met.
    test_clusters: set[int] = set()
    accumulated = 0
    for _, row in cluster_stats.iterrows():
        if accumulated >= target_test_count:
            break
        test_clusters.add(int(row["cluster"]))
        accumulated += int(row["size"])

    def _assign_output_set(row: pd.Series) -> str:
        if row["cluster"] in test_clusters:
            return "testing"
        if row["input_set"] == "validation":
            return "validation"
        return "training"

    df["output_set"] = df.apply(_assign_output_set, axis=1)
    return df


def _visualize_scenario_graph(
    filepath: Path,
    output_dir: Path,
    *,
    ego_centered: bool = False,
    k_polylines: int = 100,
) -> None:
    """Renders a scenario's road topology graph and saves it as a PNG.

    Nodes are drawn at their real-world (x, y) map coordinates and coloured by element type. Lane connectivity edges
    are shown as directed arrows. When ego_centered is True the graph is filtered and simplified to match exactly
    what was encoded by NetLSD.

    Args:
        filepath: Path to the scenario pickle file.
        output_dir: Directory in which to save the PNG.
        ego_centered: If True, restrict the graph to the K map elements closest to the ego agent's position,
            reproducing the encoding graph. Defaults to False.
        k_polylines: Number of closest map elements to retain when ego_centered is True. Defaults to 100.
    """
    try:
        with filepath.open("rb") as f:
            scenario = pickle.load(f)  # nosec B301
    except (OSError, pickle.UnpicklingError):
        return

    scenario_id = scenario.get("scenario_id", filepath.stem)
    map_infos = scenario.get("map_infos", {})

    ego_xy: NDArray[np.float64] | None = None
    if ego_centered:
        sdc_track_index = scenario["sdc_track_index"]
        curr_time_index = scenario["current_time_index"]
        trajs = scenario["track_infos"]["trajs"][sdc_track_index, curr_time_index]
        ego_xy = trajs[:2]

    graph, pos = _build_positioned_graph(map_infos, ego_xy=ego_xy, k_polylines=k_polylines if ego_centered else None)

    if graph.number_of_nodes() == 0:
        return

    node_colors = [_NODE_COLORS.get(graph.nodes[n].get("type", ""), _DEFAULT_NODE_COLOR) for n in graph.nodes]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors, node_size=20)
    nx.draw_networkx_edges(graph, pos=pos, ax=ax, edge_color="#CCCCCC", arrowsize=6, width=0.5)
    ax.set_axis_off()

    # Legend
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label)
        for label, color in _NODE_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)
    ax.set_title(f"{scenario_id}", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{scenario_id}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


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
) -> None:
    """Saves per-cluster graph visualizations and a 2-D PCA scatter plot of all descriptors.

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
        ego_centered: If True, each scenario graph is filtered to the K nearest map elements, matching the encoded
            graph. Defaults to False.
        k_polylines: Number of closest map elements to retain when ego_centered is True. Defaults to 100.
    """
    rng = np.random.default_rng(seed)
    cluster_results_path = output_data_path / "cluster_results"

    # Build scenario_id -> filepath lookup.
    filepath_map = {fp.stem: fp for fp in input_data_path.rglob("*.pkl") if "infos" not in fp.stem}

    #  Per-cluster graph examples
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
                _visualize_scenario_graph(filepath, cluster_dir, ego_centered=ego_centered, k_polylines=k_polylines)

    del split_col  # used above for column name resolution
    print(f"Saved per-cluster graph examples to {cluster_results_path}")

    # Compute per-cluster mean silhouette for legend annotation (when the column is present in the df).
    cluster_mean_sil: dict[Any, float] = {}
    if "silhouette_score" in clusters_df.columns:
        cluster_mean_sil = clusters_df.groupby("cluster")["silhouette_score"].mean().to_dict()

    # Scatter plot
    n_components = 2
    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(descriptor_matrix)

    labels = clusters_df["cluster"].to_numpy()
    n_clusters = int(labels.max()) + 1
    cmap = plt.get_cmap("tab20", n_clusters)

    fig, ax = plt.subplots(figsize=(10, 8))

    if test_clusters:
        # Draw train/val clusters with circles, test clusters with crosses.
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
                marker="o",
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
                marker="x",
                label="test",
            )
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, s=10, alpha=0.6, linewidths=0)
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(n_clusters))
        cbar.set_label("Cluster", fontsize=10)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=10)
    ax.set_title("Environment clusters (PCA of NetLSD descriptors)", fontsize=12)

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
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot to {scatter_path}")


def run(  # noqa: PLR0913, PLR0915
    input_data_path: Path,
    output_path: Path,
    n_clusters: int = 10,
    n_examples: int = 30,
    sample_percentage: float = 0.10,
    num_scenarios: int | None = None,
    num_workers: int = 8,
    ego_centered: bool = False,  # noqa: FBT001, FBT002
    k_polylines: int = 384,
    seed: int = 42,
    overwrite: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Clusters scenarios by road topology using NetLSD graph descriptors and KMeans.

    The pipeline runs in two phases:

    1. **Fit phase** — A random sample of P% of all scenarios (or `num_scenarios` if specified) is used to fit a
       StandardScaler and KMeans model. Silhouette scores on this sample drive cluster-hardness ranking.
    2. **Assign phase** — The remaining scenarios are embedded using the cached descriptors and assigned to the
       nearest cluster using the fitted model.

    After clustering, `select_train_test_splits` designates the hardest clusters (lowest mean silhouette) as the test
    set so that the test split is more challenging. The results are saved as `environment_benchmark.csv`.

    Args:
        input_data_path: Path to the processed scenario data with train/val/test splits.
        output_path: Path to the output directory.
        n_clusters: Number of KMeans clusters. Defaults to 10.
        n_examples: Number of graph visualizations to save per cluster. Defaults to 30.
        sample_percentage: Fraction of all scenarios used to fit the scaler and KMeans model. Defaults to 0.10.
        num_scenarios: Explicit number of scenarios to use for fitting. Overrides sample_percentage when set.
        num_workers: Number of parallel workers. Defaults to 8.
        ego_centered: If True, restrict each scenario's graph to the K map elements closest to the ego agent.
        k_polylines: Number of closest map elements to retain when ego_centered is True. Defaults to 384.
        seed: Random seed for KMeans, sampling, and visualization. Defaults to 42.
        overwrite: If True, recompute all descriptors and overwrite the cache. Defaults to False.

    Raises:
        ValueError: If no valid scenario descriptors could be computed.
    """
    rng = np.random.default_rng(seed)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = output_path / "descriptors_cache.pkl"

    # Collect all scenario files
    all_filepaths: list[Path] = [fp for fp in input_data_path.rglob("*.pkl") if "infos" not in fp.stem]
    print(f"Found {len(all_filepaths)} scenario files.")

    # Determine sample size for fitting
    num_samples = num_scenarios if num_scenarios is not None else max(1, round(len(all_filepaths) * sample_percentage))
    num_samples = min(num_samples, len(all_filepaths))
    print(f"Using {num_samples} scenarios ({num_samples / len(all_filepaths):.1%}) to fit the model.")

    sample_indices = rng.choice(len(all_filepaths), size=num_samples, replace=False)
    sample_mask = np.zeros(len(all_filepaths), dtype=bool)
    sample_mask[sample_indices] = True
    sample_filepaths = [fp for fp, m in zip(all_filepaths, sample_mask, strict=False) if m]
    remaining_filepaths = [fp for fp, m in zip(all_filepaths, sample_mask, strict=False) if not m]

    # Phase 1: Compute descriptors for sample and fit model
    print(f"\n[Phase 1] Computing descriptors for {len(sample_filepaths)} sample scenarios...")
    sample_ids, sample_splits, sample_descriptors = _compute_descriptors_with_cache(
        sample_filepaths,
        cache_path,
        ego_centered,
        k_polylines,
        num_workers,
        root_path=input_data_path,
        overwrite=overwrite,
    )

    print(f"Fitting StandardScaler and KMeans({n_clusters}) on sample...")
    scaler = StandardScaler()
    sample_scaled = scaler.fit_transform(sample_descriptors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    sample_labels = kmeans.fit_predict(sample_scaled)

    # Phase 2: Compute descriptors for remaining scenarios and assign clusters
    all_ids = list(sample_ids)
    all_splits = list(sample_splits)
    all_scaled = list(sample_scaled)
    all_labels = list(sample_labels)

    if remaining_filepaths:
        print(f"\n[Phase 2] Computing descriptors for {len(remaining_filepaths)} remaining scenarios...")
        rem_ids, rem_splits, rem_descriptors = _compute_descriptors_with_cache(
            remaining_filepaths,
            cache_path,
            ego_centered,
            k_polylines,
            num_workers,
            root_path=input_data_path,
            overwrite=overwrite,
        )
        rem_scaled = scaler.transform(rem_descriptors)
        rem_labels = kmeans.predict(rem_scaled)

        all_ids += list(rem_ids)
        all_splits += list(rem_splits)
        all_scaled += list(rem_scaled)
        all_labels += list(rem_labels)

    all_scaled_matrix = np.stack(all_scaled)
    all_labels_array = np.array(all_labels)

    # --- Compute per-sample silhouette scores ---
    print(f"\nComputing silhouette scores for {len(all_ids)} scenarios...")
    sil_scores: NDArray[np.float64] = np.asarray(silhouette_samples(all_scaled_matrix, all_labels_array))

    # --- Build clusters DataFrame and assign train/test splits ---
    clusters_df = pd.DataFrame({"scenario_id": all_ids, "split": all_splits, "cluster": all_labels_array})
    benchmark_df = select_train_test_splits(clusters_df, sil_scores)

    # Rename for output schema: scenario_id, cluster_label, silhouette_score, input_set, output_set
    output_df = benchmark_df[["scenario_id", "cluster", "silhouette_score", "input_set", "output_set"]].rename(
        columns={"cluster": "cluster_label"}
    )
    benchmark_csv_path = output_path / "environment_benchmark.csv"
    output_df.to_csv(benchmark_csv_path, index=False)
    print(f"\nSaved benchmark split assignments to {benchmark_csv_path}")

    # --- Save model artifacts ---
    kmeans_path = output_path / "kmeans_model.pkl"
    with kmeans_path.open("wb") as f:
        pickle.dump(kmeans, f)

    scaler_path = output_path / "scaler.pkl"
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved KMeans model to {kmeans_path}")
    print(f"Saved StandardScaler to {scaler_path}")

    # --- Print summary ---
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

    # --- Visualize (using sample for graph examples; all for scatter) ---
    sample_cluster_df = benchmark_df.iloc[: len(sample_ids)].copy()
    visualize_clusters(
        sample_cluster_df,
        np.stack(sample_scaled),  # pyright: ignore[reportArgumentType, reportCallIssue]
        input_data_path,
        output_path,
        n_examples,
        seed,
        test_clusters=test_cluster_ids,
        ego_centered=ego_centered,
        k_polylines=k_polylines,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_path",
        type=Path,
        default="/data/driving/waymo/processed/mini_causal/",
        help="Path to the processed scenario data with train/val/test splits.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="./out",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="Number of KMeans clusters.",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=30,
        help="Number of graph visualizations to save per cluster.",
    )
    parser.add_argument(
        "--sample_percentage",
        type=float,
        default=0.20,
        help="Fraction of all scenarios used to fit the scaler and KMeans model (e.g. 0.10 = 10%%).",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=None,
        help="Explicit number of scenarios to use for fitting. Overrides --sample_percentage when set.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to run.")
    parser.add_argument(
        "--ego_centered",
        action="store_true",
        default=False,
        help="Restrict each scenario's graph to the K map elements closest to the ego agent's position.",
    )
    parser.add_argument(
        "--k_polylines",
        type=int,
        default=384,
        help="Number of closest map elements to retain when --ego_centered is set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Recompute all descriptors and overwrite the cache, ignoring any previously cached embeddings.",
    )
    args = parser.parse_args()
    run(**vars(args))
