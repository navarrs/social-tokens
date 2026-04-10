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
import operator
import pickle  # nosec B403
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import netlsd
import networkx as nx
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


def _rotate_points_along_z(points: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """Rotates the (x, y) columns of an arbitrary-shape array by angle radians around the Z axis.

    Mirrors the rotation used in base_dataset.py's transform pipeline. All other columns are left unchanged.

    Args:
        points: Array whose last dimension is at least 2, with x at index 0 and y at index 1.
        angle: Rotation angle in radians (positive = counter-clockwise).

    Returns:
        Copy of points with (x, y) rotated in place.
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    result = points.copy()
    result[..., :2] = points[..., :2] @ rot.T
    return result


def _filter_map_elements_by_proximity(  # noqa: PLR0913
    map_infos: dict[str, Any],
    all_polylines: NDArray[np.float64],
    ego_xy: NDArray[np.float64],
    k: int,
    ego_heading: float = 0.0,
    map_range: float = 100.0,
) -> dict[str, Any]:
    """Returns a filtered copy of map_infos keeping map elements that are within range of the ego agent.

    Mirrors the selection logic in base_dataset.py's ``get_centered_map_data``:

    1. All polyline points are transformed to the ego-centric frame (translate by -ego_xy, rotate by -ego_heading).
    2. A map element passes the **range filter** if any of its points fall within the L∞ box
       ``|x| < map_range AND |y| < map_range``.
    3. Among passing elements, the top-K are kept by ascending **average L2 distance** of their points from the
       ego-frame origin.

    Node positions stored in the graph remain in world coordinates — only the filtering step uses the ego frame.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.
        all_polylines: Array of shape (N, >=2) with all polyline points in world coordinates.
        ego_xy: Ego position (x, y) at the current time step.
        k: Maximum number of map elements to retain.
        ego_heading: Ego heading in radians. Used to rotate polyline points to the ego-centric frame before the range
            check, matching base_dataset.py's rotation convention. Defaults to 0.0.
        map_range: L∞ half-width of the ego-centric range box in metres. Elements with at least one point inside the
            box are candidates. Defaults to 100.0.

    Returns:
        Shallow copy of map_infos with each element list filtered to the K nearest in-range elements.
    """
    if all_polylines.shape[0] == 0:
        return map_infos

    pts_ego: NDArray[np.float64] = all_polylines.copy()
    pts_ego[..., :2] -= ego_xy[:2]
    pts_ego = _rotate_points_along_z(pts_ego, -ego_heading)

    candidates: list[tuple[float, str, dict]] = []  # pyright: ignore[reportMissingTypeArgument]
    for etype in _MAP_ELEMENT_TYPES:
        for element in map_infos.get(etype, []):
            start, end = element.get("polyline_index", (0, 0))
            if end <= start:
                continue
            pts = pts_ego[start:end, :2]

            in_range = (np.abs(pts[:, 0]) < map_range) & (np.abs(pts[:, 1]) < map_range)
            if not in_range.any():
                continue

            avg_dist = float(np.linalg.norm(pts, axis=-1).mean())
            candidates.append((avg_dist, etype, element))

    candidates.sort(key=operator.itemgetter(0))
    filtered: dict[str, list] = {etype: [] for etype in _MAP_ELEMENT_TYPES}  # pyright: ignore[reportMissingTypeArgument]
    for _, etype, element in candidates[:k]:
        filtered[etype].append(element)

    return {**map_infos, **filtered}


def _map_infos_to_graph(
    map_infos: dict[str, Any],
    ego_xy: NDArray[np.float64] | None = None,
    k_polylines: int | None = None,
    ego_heading: float = 0.0,
    map_range: float = 100.0,
) -> nx.DiGraph:
    """Converts scenario map information into a directed NetworkX graph.

    Nodes represent lanes, road lines, road edges, crosswalks, speed bumps, and stop signs. Directed edges connect
    lanes via their entry/exit lane relationships. Each node stores the centroid (x, y, z) of its polyline as
    position attributes when available.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.
        ego_xy: If provided together with k_polylines, triggers ego-centric filtering via
            `_filter_map_elements_by_proximity`.
        k_polylines: Maximum number of map elements to retain when ego_xy is set.
        ego_heading: Ego heading in radians, used for the ego-frame range check. Defaults to 0.0.
        map_range: L∞ half-width of the ego-centric range box in metres. Defaults to 100.0.

    Returns:
        nx.DiGraph: Directed graph representing the road topology.
    """
    graph = nx.DiGraph()
    all_polylines: NDArray[np.float64] = np.asarray(map_infos.get("all_polylines", np.empty((0, 7))))

    if ego_xy is not None and k_polylines is not None:
        map_infos = _filter_map_elements_by_proximity(
            map_infos, all_polylines, ego_xy, k_polylines, ego_heading=ego_heading, map_range=map_range
        )

    def _node_pos(element: dict[str, Any]) -> dict[str, float]:
        start, end = element.get("polyline_index", (0, 0))
        if all_polylines.shape[0] > 0 and end > start:
            centroid = all_polylines[start:end, :3].mean(axis=0)
            return {"x": float(centroid[0]), "y": float(centroid[1]), "z": float(centroid[2])}
        return {}

    for lane in map_infos.get("lane", []):
        lane_id = lane["id"]
        graph.add_node(lane_id, type="lane", speed_limit_mph=lane.get("speed_limit_mph", 0.0), **_node_pos(lane))
        for entry_id in lane.get("entry_lanes", []):
            graph.add_edge(entry_id, lane_id)
        for exit_id in lane.get("exit_lanes", []):
            graph.add_edge(lane_id, exit_id)

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
    ego_heading: float = 0.0,
    map_range: float = 100.0,
) -> tuple[nx.DiGraph, dict[Any, NDArray[np.float64]]]:
    """Builds a map graph and extracts a matplotlib-compatible position dict from node attributes.

    Reads (x, y) from node attributes set by `_map_infos_to_graph`. When ego_xy and k_polylines are provided the
    graph is filtered using the ego-centric L∞ range box and simplified, matching the graph that was passed to NetLSD
    during encoding. Nodes without position attributes are placed via a spring layout fallback.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.
        ego_xy: Ego position (x, y). When provided together with k_polylines, triggers ego-centric filtering.
        k_polylines: Maximum number of map elements to retain when ego_xy is set.
        ego_heading: Ego heading in radians, forwarded to `_map_infos_to_graph`. Defaults to 0.0.
        map_range: L∞ half-width of the ego-centric range box in metres. Defaults to 100.0.

    Returns:
        The directed graph and a dict mapping node id to (x, y) position.
    """
    graph = _simplify_graph(
        _map_infos_to_graph(
            map_infos, ego_xy=ego_xy, k_polylines=k_polylines, ego_heading=ego_heading, map_range=map_range
        )
    )

    pos: dict[Any, NDArray[np.float64]] = {
        n: np.array([data["x"], data["y"]]) for n, data in graph.nodes(data=True) if "x" in data and "y" in data
    }

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
    graph = _simplify_graph(
        _map_infos_to_graph(
            map_infos,
            ego_xy=ego_xy,
            k_polylines=k_polylines if ego_centered else None,
            ego_heading=ego_heading,
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

    def _assign_output_set(row: pd.Series) -> str:  # pyright: ignore[reportMissingTypeArgument]
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
    map_range: float = 100.0,
) -> None:
    """Renders a scenario's road topology graph and saves it as a PNG.

    Nodes are drawn at their real-world (x, y) map coordinates and coloured by element type. Lane connectivity edges
    are shown as directed arrows. When ego_centered is True the graph is filtered and simplified to match exactly
    what was encoded by NetLSD.

    Args:
        filepath: Path to the scenario pickle file.
        output_dir: Directory in which to save the PNG.
        ego_centered: If True, restrict the graph using the ego-centric L∞ range filter, reproducing the encoding
            graph. Defaults to False.
        k_polylines: Maximum number of map elements to retain when ego_centered is True. Defaults to 100.
        map_range: L∞ half-width of the ego-centric range box in metres. Defaults to 100.0.
    """
    try:
        with filepath.open("rb") as f:
            scenario = pickle.load(f)  # nosec B301
    except (OSError, pickle.UnpicklingError):
        return

    scenario_id = scenario.get("scenario_id", filepath.stem)
    map_infos = scenario.get("map_infos", {})

    ego_xy: NDArray[np.float64] | None = None
    ego_heading: float = 0.0
    if ego_centered:
        sdc_track_index = scenario["sdc_track_index"]
        curr_time_index = scenario["current_time_index"]
        trajs = scenario["track_infos"]["trajs"][sdc_track_index, curr_time_index]
        ego_xy = trajs[:2]
        ego_heading = float(trajs[6])

    graph, pos = _build_positioned_graph(
        map_infos,
        ego_xy=ego_xy,
        k_polylines=k_polylines if ego_centered else None,
        ego_heading=ego_heading,
        map_range=map_range,
    )

    if graph.number_of_nodes() == 0:
        return

    node_colors = [_NODE_COLORS.get(graph.nodes[n].get("type", ""), _DEFAULT_NODE_COLOR) for n in graph.nodes]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors, node_size=20)
    nx.draw_networkx_edges(graph, pos=pos, ax=ax, edge_color="#CCCCCC", arrowsize=6, width=0.5)
    ax.set_axis_off()

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label)
        for label, color in _NODE_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)
    ax.set_title(f"{scenario_id}", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(output_dir / f"{scenario_id}.png"), dpi=100, bbox_inches="tight")
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
                _visualize_scenario_graph(
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
            num_scenarios, num_workers, ego_centered, k_polylines, seed, overwrite, map_range, reduction.

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
    benchmark_df = select_train_test_splits(clusters_df, sil_scores)

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
