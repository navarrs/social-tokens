"""Map graph utilities for road topology analysis and visualization.

Provides functions for converting scenario map data into NetworkX graphs, filtering map elements
by proximity to a reference point, and rendering graph visualizations.
"""

import operator
import pickle  # nosec B403
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from numpy.typing import NDArray


def _rotate_points_along_z(points: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """Rotates the (x, y) columns of an array by angle radians around the Z axis.

    Args:
        points: Array whose last dimension is at least 2, with x at index 0 and y at index 1.
        angle: Rotation angle in radians (positive = counter-clockwise).

    Returns:
        Copy of points with (x, y) rotated.
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    result = points.copy()
    result[..., :2] = points[..., :2] @ rot.T
    return result


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


def filter_map_elements_by_proximity(  # noqa: PLR0913
    map_infos: dict[str, Any],
    all_polylines: NDArray[np.float64],
    ref_xy: NDArray[np.float64],
    k: int,
    ref_heading: float = 0.0,
    map_range: float = 100.0,
) -> dict[str, Any]:
    """Returns a filtered copy of map_infos keeping map elements that are within range of the reference point.

    Mirrors the selection logic in base_dataset.py's ``get_centered_map_data``:

    1. All polyline points are transformed to the reference frame (translate by -ref_xy, rotate by -ref_heading).
    2. A map element passes the **range filter** if any of its points fall within the L∞ box
       ``|x| < map_range AND |y| < map_range``.
    3. Among passing elements, the top-K are kept by ascending **average L2 distance** of their points from the
       reference-frame origin.

    Node positions stored in the graph remain in world coordinates — only the filtering step uses the reference frame.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.
        all_polylines: Array of shape (N, >=2) with all polyline points in world coordinates.
        ref_xy: Reference position (x, y).
        k: Maximum number of map elements to retain.
        ref_heading: Reference heading in radians. Used to rotate polyline points to the reference frame before the
            range check, matching base_dataset.py's rotation convention. Defaults to 0.0.
        map_range: L∞ half-width of the reference-frame range box in metres. Elements with at least one point inside
            the box are candidates. Defaults to 100.0.

    Returns:
        Shallow copy of map_infos with each element list filtered to the K nearest in-range elements.
    """
    if all_polylines.shape[0] == 0:
        return map_infos

    pts_ref: NDArray[np.float64] = all_polylines.copy()
    pts_ref[..., :2] -= ref_xy[:2]
    pts_ref = _rotate_points_along_z(pts_ref, -ref_heading)

    candidates: list[tuple[float, str, dict]] = []  # pyright: ignore[reportMissingTypeArgument]
    for etype in _MAP_ELEMENT_TYPES:
        for element in map_infos.get(etype, []):
            start, end = element.get("polyline_index", (0, 0))
            if end <= start:
                continue
            pts = pts_ref[start:end, :2]

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


def map_infos_to_graph(
    map_infos: dict[str, Any],
    ref_xy: NDArray[np.float64] | None = None,
    k_polylines: int | None = None,
    ref_heading: float = 0.0,
    map_range: float = 100.0,
) -> nx.DiGraph:
    """Converts scenario map information into a directed NetworkX graph.

    Nodes represent lanes, road lines, road edges, crosswalks, speed bumps, and stop signs. Directed edges connect
    lanes via their entry/exit lane relationships. Each node stores the centroid (x, y, z) of its polyline as
    position attributes when available.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.
        ref_xy: If provided together with k_polylines, triggers proximity filtering via
            `filter_map_elements_by_proximity`.
        k_polylines: Maximum number of map elements to retain when ref_xy is set.
        ref_heading: Reference heading in radians, used for the range check. Defaults to 0.0.
        map_range: L∞ half-width of the reference-frame range box in metres. Defaults to 100.0.

    Returns:
        nx.DiGraph: Directed graph representing the road topology.
    """
    graph = nx.DiGraph()
    all_polylines: NDArray[np.float64] = np.asarray(map_infos.get("all_polylines", np.empty((0, 7))))

    if ref_xy is not None and k_polylines is not None:
        map_infos = filter_map_elements_by_proximity(
            map_infos, all_polylines, ref_xy, k_polylines, ref_heading=ref_heading, map_range=map_range
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


def simplify_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Returns a simplified copy of the road topology graph for more stable encoding.

    Two simplifications are applied in order:

    1. **Remove isolated nodes** — nodes with no edges (degree zero) do not contribute to the graph's Laplacian
        structure but inflate the heat-trace baseline, causing NetLSD descriptors to vary with the number of
        non-connected map elements rather than road topology.
    2. **Keep the largest weakly-connected component** — small disconnected lane fragments at the scene boundary add
        noise without representing the main road network.

    Args:
        graph: Full road topology graph produced by `map_infos_to_graph`.

    Returns:
        Simplified directed graph containing only the largest connected lane subgraph.
    """
    graph = graph.copy()
    graph.remove_nodes_from(list(nx.isolates(graph)))

    if graph.number_of_nodes() == 0:
        return graph

    largest_wcc = max(nx.weakly_connected_components(graph), key=len)
    return nx.DiGraph(graph.subgraph(largest_wcc))


def build_positioned_graph(
    map_infos: dict[str, Any],
    ref_xy: NDArray[np.float64] | None = None,
    k_polylines: int | None = None,
    ref_heading: float = 0.0,
    map_range: float = 100.0,
) -> tuple[nx.DiGraph, dict[Any, NDArray[np.float64]]]:
    """Builds a map graph and extracts a matplotlib-compatible position dict from node attributes.

    Reads (x, y) from node attributes set by `map_infos_to_graph`. When ref_xy and k_polylines are provided the
    graph is filtered using the reference-frame L∞ range box and simplified. Nodes without position attributes are
    placed via a spring layout fallback.

    Args:
        map_infos: Map information dictionary from a scenario pickle file.
        ref_xy: Reference position (x, y). When provided together with k_polylines, triggers proximity filtering.
        k_polylines: Maximum number of map elements to retain when ref_xy is set.
        ref_heading: Reference heading in radians, forwarded to `map_infos_to_graph`. Defaults to 0.0.
        map_range: L∞ half-width of the reference-frame range box in metres. Defaults to 100.0.

    Returns:
        The directed graph and a dict mapping node id to (x, y) position.
    """
    graph = simplify_graph(
        map_infos_to_graph(
            map_infos, ref_xy=ref_xy, k_polylines=k_polylines, ref_heading=ref_heading, map_range=map_range
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


def visualize_scenario_graph(
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
        ego_centered: If True, restrict the graph using the reference-frame L∞ range filter, reproducing the encoding
            graph. Defaults to False.
        k_polylines: Maximum number of map elements to retain when ego_centered is True. Defaults to 100.
        map_range: L∞ half-width of the reference-frame range box in metres. Defaults to 100.0.
    """
    try:
        with filepath.open("rb") as f:
            scenario = pickle.load(f)  # nosec B301
    except (OSError, pickle.UnpicklingError):
        return

    scenario_id = scenario.get("scenario_id", filepath.stem)
    map_infos = scenario.get("map_infos", {})

    ref_xy: NDArray[np.float64] | None = None
    ref_heading: float = 0.0
    if ego_centered:
        sdc_track_index = scenario["sdc_track_index"]
        curr_time_index = scenario["current_time_index"]
        trajs = scenario["track_infos"]["trajs"][sdc_track_index, curr_time_index]
        ref_xy = trajs[:2]
        ref_heading = float(trajs[6])

    graph, pos = build_positioned_graph(
        map_infos,
        ref_xy=ref_xy,
        k_polylines=k_polylines if ego_centered else None,
        ref_heading=ref_heading,
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
