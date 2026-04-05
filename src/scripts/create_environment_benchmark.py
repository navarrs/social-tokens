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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
    all_polylines: np.ndarray,
    ego_xy: np.ndarray,
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
    ego_xy: np.ndarray | None = None,
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
    all_polylines: np.ndarray = np.asarray(map_infos.get("all_polylines", np.empty((0, 7))))

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


def _build_positioned_graph(map_infos: dict[str, Any]) -> tuple[nx.DiGraph, dict[Any, np.ndarray]]:
    """Builds a map graph and extracts a matplotlib-compatible position dict from node attributes.

    Reads (x, y) from node attributes set by `_map_infos_to_graph`. Nodes without position attributes
    are placed via a spring layout fallback.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.

    Returns:
        The directed graph and a dict mapping node id to (x, y) position.
    """
    graph = _map_infos_to_graph(map_infos)

    pos: dict[Any, np.ndarray] = {
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
) -> tuple[str, str, np.ndarray] | None:
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

    ego_xy: np.ndarray | None = None
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

    descriptor: np.ndarray = netlsd.heat(graph)

    # The split is the parent directory name (e.g. training/validation/testing).
    split = filepath.parent.name

    return scenario_id, split, descriptor


def _visualize_scenario_graph(filepath: Path, output_dir: Path) -> None:
    """Renders a scenario's road topology graph and saves it as a PNG.

    Nodes are drawn at their real-world (x, y) map coordinates and coloured by element type. Lane connectivity edges
    are shown as directed arrows.

    Args:
        filepath: Path to the scenario pickle file.
        output_dir: Directory in which to save the PNG.
    """
    try:
        with filepath.open("rb") as f:
            scenario = pickle.load(f)  # nosec B301
    except (OSError, pickle.UnpicklingError):
        return

    scenario_id = scenario.get("scenario_id", filepath.stem)
    map_infos = scenario.get("map_infos", {})
    graph, pos = _build_positioned_graph(map_infos)

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


def visualize_clusters(  # noqa: PLR0913
    clusters_df: pd.DataFrame,
    descriptor_matrix: np.ndarray,
    input_data_path: Path,
    output_data_path: Path,
    n_examples: int = 30,
    seed: int = 42,
) -> None:
    """Saves per-cluster graph visualizations and a 2-D PCA scatter plot of all descriptors.

    For each cluster, up to `n_examples` scenario graphs are rendered as PNGs under
    `output_data_path/cluster_results/cluster_<id>/`. A single scatter plot coloured by cluster label is saved as
    `output_data_path/cluster_results/cluster_scatter.png`.

    Args:
        clusters_df: DataFrame with columns scenario_id, split, cluster.
        descriptor_matrix: Array of shape (N, D) — one descriptor row per scenario.
        input_data_path: Root path of the processed scenario data.
        output_data_path: Directory where cluster results will be written.
        n_examples: Number of example graphs per cluster. Defaults to 30.
        seed: Random seed for example sampling. Defaults to 42.
    """
    rng = np.random.default_rng(seed)
    cluster_results_path = output_data_path / "cluster_results"

    # Build scenario_id -> filepath lookup.
    filepath_map = {fp.stem: fp for fp in input_data_path.rglob("*.pkl") if "infos" not in fp.stem}

    #  Per-cluster graph examples
    for cluster_id, group in clusters_df.groupby("cluster"):
        cluster_dir = cluster_results_path / f"cluster_{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        scenario_ids = group["scenario_id"].tolist()
        n_chosen = min(n_examples, len(scenario_ids))
        chosen = rng.choice(scenario_ids, size=n_chosen, replace=False).tolist()

        for scenario_id in tqdm(chosen, desc=f"Visualizing cluster {cluster_id}", leave=False):
            filepath = filepath_map.get(str(scenario_id))
            if filepath is not None:
                _visualize_scenario_graph(filepath, cluster_dir)

    print(f"Saved per-cluster graph examples to {cluster_results_path}")

    # Scatter plot
    n_components = 2
    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(descriptor_matrix)

    labels = clusters_df["cluster"].to_numpy()
    n_clusters = int(labels.max()) + 1
    cmap = plt.get_cmap("tab20", n_clusters)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, s=10, alpha=0.6, linewidths=0)
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(n_clusters))
    cbar.set_label("Cluster", fontsize=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=10)
    ax.set_title("Environment clusters (PCA of NetLSD descriptors)", fontsize=12)
    fig.tight_layout()
    scatter_path = cluster_results_path / "cluster_scatter.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot to {scatter_path}")


def run(  # noqa: PLR0913
    input_data_path: Path,
    output_path: Path,
    n_clusters: int = 10,
    n_examples: int = 30,
    num_scenarios: int | None = None,
    num_workers: int = 8,
    ego_centered: bool = False,  # noqa: FBT001, FBT002
    k_polylines: int = 100,
    seed: int = 42,
) -> None:
    """Clusters scenarios by road topology using NetLSD graph descriptors and KMeans.

    For each scenario, the map information is converted into a directed NetworkX graph (lanes as nodes, entry/exit
    relationships as edges). NetLSD encodes each graph into a fixed-length descriptor. KMeans then clusters these
    descriptors and the results are saved as a CSV. Per-cluster graph visualizations and a scatter plot are written
    under `output_path/cluster_results/`.

    Args:
        input_data_path (Path): Path to the processed scenario data with train/val/test splits.
        output_path (Path): Path to the output directory.
        n_clusters (int, optional): Number of KMeans clusters. Defaults to 10.
        n_examples (int, optional): Number of graph visualizations to save per cluster. Defaults to 30.
        num_scenarios (int | None, optional): Maximum number of scenarios to process. If None, all are used.
            Defaults to None.
        num_workers (int, optional): Number of parallel workers. Defaults to 8.
        ego_centered (bool, optional): If True, restrict each scenario's graph to the K map elements closest to the
            ego agent's position at the current time step. Defaults to False.
        k_polylines (int, optional): Number of closest map elements to retain when ego_centered is True.
            Defaults to 100.
        seed (int, optional): Random seed for KMeans and sampling. Defaults to 42.

    Raises:
        ValueError: If no valid scenario descriptors could be computed.
    """
    rng = np.random.default_rng(seed)
    filepaths = [fp for fp in input_data_path.rglob("*.pkl") if "infos" not in fp.stem]
    if num_scenarios is not None and num_scenarios < len(filepaths):
        filepaths = rng.choice(filepaths, size=num_scenarios, replace=False).tolist()  # type: ignore[assignment]
    print(f"Found {len(filepaths)} scenario files. Computing graph descriptors...")

    worker_fn = functools.partial(_compute_graph_descriptor, ego_centered=ego_centered, k_polylines=k_polylines)

    if num_workers == 0:
        results = [worker_fn(fp) for fp in tqdm(filepaths, desc="Encoding map graphs")]
    else:
        with multiprocessing.Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(worker_fn, filepaths),
                    total=len(filepaths),
                    desc="Encoding map graphs",
                )
            )

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        error_message = "No valid scenario descriptors were computed. Check the input data path."
        raise ValueError(error_message)

    scenario_ids, splits, descriptors = zip(*valid_results, strict=False)
    descriptor_matrix = np.stack(descriptors)

    print(f"Clustering {len(scenario_ids)} scenarios into {n_clusters} environment clusters...")
    scaler = StandardScaler()
    descriptor_matrix_scaled = scaler.fit_transform(descriptor_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(descriptor_matrix_scaled)

    output_path.mkdir(parents=True, exist_ok=True)
    clusters_df = pd.DataFrame({"scenario_id": scenario_ids, "split": splits, "cluster": labels})
    clusters_csv_path = output_path / "environment_clusters.csv"
    clusters_df.to_csv(clusters_csv_path, index=False)
    print(f"Saved cluster assignments to {clusters_csv_path}")

    kmeans_path = output_path / "kmeans_model.pkl"
    with kmeans_path.open("wb") as f:
        pickle.dump(kmeans, f)
    print(f"Saved KMeans model to {kmeans_path}")

    cluster_counts = clusters_df["cluster"].value_counts().sort_index()
    print("Cluster sizes:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} scenarios")

    visualize_clusters(clusters_df, descriptor_matrix_scaled, input_data_path, output_path, n_examples, seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_path",
        type=Path,
        default="/datasets/driving/waymo/processed/mini_causal/",
        help="Path to the processed scenario data with train/val/test splits.",
    )
    # parser.add_argument(
    #     "--output_data_path",
    #     type=Path,
    #     default="/datasets/driving/waymo/processed/environment_benchmark/",
    #     help="Path to the output directory.",
    # )
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
        "--num_scenarios",
        type=int,
        default=None,
        help="Maximum number of scenarios to process. If unset, all scenarios are used.",
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
    args = parser.parse_args()
    run(**vars(args))
