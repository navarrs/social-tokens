import json
import multiprocessing
import pickle  # nosec B403
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from characterization.utils.common import MIN_VALID_POINTS
from numpy.random import Generator, default_rng
from tqdm import tqdm


def _remove_causal(scenario: dict[str, Any], causal_labels: dict[str, Any], output_filepath: Path) -> None:
    """Removes causal objects from a scenario by setting the last column of the trajectories to 0 for causal objects.

    Args:
        scenario (dict[str, Any]): Scenario dictionary.
        causal_labels (dict[str, Any]): Causal labels dictionary.
        output_filepath (Path): Path to the output file.
    """
    causal_ids = np.array(causal_labels["causal_ids"], dtype=np.int64)
    object_ids = np.array(scenario["track_infos"]["object_id"])

    # Identify causal objects
    causal_mask = np.isin(object_ids, causal_ids)

    # Add causal IDs to the scenario
    track_infos = scenario["track_infos"]
    track_infos["causal_ids"] = causal_labels["causal_ids"]

    # Mask out causal objects
    trajectories = track_infos["trajs"].copy()
    trajectories[..., -1][causal_mask] = 0
    track_infos["trajs"] = trajectories
    scenario["track_infos"] = track_infos

    # Remove causal agents if they are in within the 'tracks_to_predict'
    agent_idxs = np.arange(len(object_ids))
    causal_idxs = agent_idxs[causal_mask]

    tracks_to_predict = scenario["tracks_to_predict"]
    track_index = np.array(tracks_to_predict["track_index"])
    track_difficulty = np.array(tracks_to_predict["difficulty"])
    object_type = np.array(tracks_to_predict["object_type"])

    # Filter out causal tracks
    causal_track_index_mask = ~np.isin(track_index, causal_idxs)
    filtered_tracks_to_predict = {
        "track_index": track_index[causal_track_index_mask].tolist(),
        "track_difficulty": track_difficulty[causal_track_index_mask].tolist(),
        "object_type": object_type[causal_track_index_mask].tolist(),
    }
    scenario["tracks_to_predict"] = filtered_tracks_to_predict

    with output_filepath.open("wb") as f:
        pickle.dump(scenario, f)


def _remove_noncausal(scenario: dict[str, Any], causal_labels: dict[str, Any], output_filepath: Path) -> None:
    """Removes non-causal objects from a scenario by setting the last column of the trajectories to 0 for non-causal
    objects.

    Args:
        scenario (dict[str, Any]): Scenario dictionary.
        causal_labels (dict[str, Any]): Causal labels dictionary.
        output_filepath (Path): Path to the output file.
    """
    object_ids = np.array(scenario["track_infos"]["object_id"])
    ego_idx = scenario["sdc_track_index"]
    ego_id = object_ids[ego_idx]

    # Make sure not to delete the ego vehicle
    causal_ids = np.array(causal_labels["causal_ids"] + [ego_id], dtype=np.int64)

    # Identify noncausal objects
    noncausal_mask = ~np.isin(object_ids, causal_ids)

    # Add causal IDs to the scenario
    track_infos = scenario["track_infos"]
    track_infos["causal_ids"] = causal_labels["causal_ids"]

    # Mask out causal objects
    trajectories = track_infos["trajs"].copy()
    trajectories[..., -1][noncausal_mask] = 0
    track_infos["trajs"] = trajectories
    scenario["track_infos"] = track_infos

    # Remove causal agents if they are in within the 'tracks_to_predict'
    agent_idxs = np.arange(len(object_ids))
    noncausal_idxs = agent_idxs[noncausal_mask]

    tracks_to_predict = scenario["tracks_to_predict"]
    track_index = np.array(tracks_to_predict["track_index"])
    track_difficulty = np.array(tracks_to_predict["difficulty"])
    object_type = np.array(tracks_to_predict["object_type"])

    # Filter out non-causal tracks
    noncausal_track_index_mask = ~np.isin(track_index, noncausal_idxs)
    filtered_tracks_to_predict = {
        "track_index": track_index[noncausal_track_index_mask].tolist(),
        "track_difficulty": track_difficulty[noncausal_track_index_mask].tolist(),
        "object_type": object_type[noncausal_track_index_mask].tolist(),
    }
    scenario["tracks_to_predict"] = filtered_tracks_to_predict

    with output_filepath.open("wb") as f:
        pickle.dump(scenario, f)


def _remove_noncausalequal(
    scenario: dict[str, Any], causal_labels: dict[str, Any], output_filepath: Path, random_generator: Generator
) -> None:
    """Removes a subset of non-causal objects from a scenario by setting the last column of the trajectories to 0 for
    a subset of non-causal objects equal to the number of causal objects.

    Args:
        scenario (dict[str, Any]): Scenario dictionary.
        causal_labels (dict[str, Any]): Causal labels dictionary.
        output_filepath (Path): Path to the output file.
        random_generator (Generator): Random number generator.
    """
    object_ids = np.array(scenario["track_infos"]["object_id"])
    ego_idx = scenario["sdc_track_index"]
    ego_id = object_ids[ego_idx]

    # Make sure not to delete the ego vehicle
    causal_ids = np.array(causal_labels["causal_ids"] + [ego_id], dtype=np.int64)

    # Identify causal objects
    noncausal_mask = ~np.isin(object_ids, causal_ids)

    # Instead of removing all non-causal objects, we randomly remove a subset of them equals to the number of causal
    # objects.
    num_to_remove = len(causal_labels["causal_ids"])
    agent_idxs = np.arange(len(object_ids))
    noncausal_idxs = agent_idxs[noncausal_mask]
    noncausal_idxs_to_remove = random_generator.choice(
        noncausal_idxs, size=min(num_to_remove, len(noncausal_idxs)), replace=False
    )

    # Add causal IDs to the scenario
    track_infos = scenario["track_infos"]
    track_infos["causal_ids"] = causal_labels["causal_ids"]

    # Mask out causal objects
    trajectories = track_infos["trajs"].copy()
    trajectories[..., -1][noncausal_idxs_to_remove] = 0
    track_infos["trajs"] = trajectories
    scenario["track_infos"] = track_infos

    # Remove causal agents if they are in within the 'tracks_to_predict'
    tracks_to_predict = scenario["tracks_to_predict"]
    track_index = np.array(tracks_to_predict["track_index"])
    track_difficulty = np.array(tracks_to_predict["difficulty"])
    object_type = np.array(tracks_to_predict["object_type"])

    # Filter out non-causal tracks
    noncausal_track_index_mask = ~np.isin(track_index, noncausal_idxs_to_remove)
    filtered_tracks_to_predict = {
        "track_index": track_index[noncausal_track_index_mask].tolist(),
        "track_difficulty": track_difficulty[noncausal_track_index_mask].tolist(),
        "object_type": object_type[noncausal_track_index_mask].tolist(),
    }
    scenario["tracks_to_predict"] = filtered_tracks_to_predict

    with output_filepath.open("wb") as f:
        pickle.dump(scenario, f)


def _remove_static(scenario: dict[str, Any], output_filepath: Path, threshold_distance: float = 0.1) -> None:
    track_infos = scenario["track_infos"]
    track_infos["static_threshold_distance"] = threshold_distance

    # Mask out static objects
    trajectories = track_infos["trajs"].copy()

    static_mask = np.zeros(trajectories.shape[0], dtype=bool)
    for n, traj in enumerate(trajectories):
        valid_mask = traj[..., -1].astype(bool)
        if valid_mask.sum() < MIN_VALID_POINTS:
            continue
        pos = traj[..., :2][valid_mask]

        # Check if the trajectory is static
        static_mask[n] = np.linalg.norm(pos[-1] - pos[0], axis=-1) < threshold_distance
    ego_idx = scenario["sdc_track_index"]
    static_mask[ego_idx] = False

    # Remove static objects
    trajectories[..., -1][static_mask] = 0
    track_infos["trajs"] = trajectories
    scenario["track_infos"] = track_infos

    # Remove causal agents if they are in within the 'tracks_to_predict'
    tracks_to_predict = scenario["tracks_to_predict"]
    track_index = np.array(tracks_to_predict["track_index"])
    track_difficulty = np.array(tracks_to_predict["difficulty"])
    object_type = np.array(tracks_to_predict["object_type"])

    # Filter out non-causal tracks
    object_ids = np.array(scenario["track_infos"]["object_id"])
    agent_idxs = np.arange(len(object_ids))
    static_idxs = agent_idxs[static_mask]
    static_track_index_mask = ~np.isin(track_index, static_idxs)

    filtered_track_index = track_index[static_track_index_mask].tolist()
    if not filtered_track_index:
        return

    filtered_tracks_to_predict = {
        "track_index": track_index[static_track_index_mask].tolist(),
        "track_difficulty": track_difficulty[static_track_index_mask].tolist(),
        "object_type": object_type[static_track_index_mask].tolist(),
    }
    scenario["tracks_to_predict"] = filtered_tracks_to_predict

    with output_filepath.open("wb") as f:
        pickle.dump(scenario, f)


def _create_scenario(  # noqa: PLR0913
    input_filepath: Path,
    output_path: Path,
    causal_labels_path: Path,
    scenario_mapping: dict[str, str],
    benchmark: str,
    random_generator: Generator,
) -> None:
    """Creates a benchmark scenario info file from a processed Waymo scenario.

    Args:
        input_filepath (Path): Path to the input file.
        output_path (Path): Path to the output directory.
        causal_labels_path (Path): Path to the causal labels.
        scenario_mapping (dict[str, str]): Mapping from scenario_id to split and shard. Used to determine the output
            filepath.
        benchmark (str): Benchmark name.
        random_generator (Generator): Random number generator.
    """
    if not input_filepath.exists():
        return

    with input_filepath.open("rb") as f:
        scenario = pickle.load(f)

    scenario_id = scenario["scenario_id"]

    causal_labels_filepath = causal_labels_path / f"{scenario_id}.json"
    if not causal_labels_filepath.exists():
        return

    with causal_labels_filepath.open("r") as f:
        causal_labels = json.load(f)

    split = scenario_mapping[scenario_id]["split"]
    shard = scenario_mapping[scenario_id]["shard"]
    output_filepath = output_path / split / shard / f"{scenario_id}.pkl"

    match benchmark:
        case "remove_causal":
            _remove_causal(scenario, causal_labels, output_filepath)
        case "remove_noncausal":
            _remove_noncausal(scenario, causal_labels, output_filepath)
        case "remove_noncausalequal":
            _remove_noncausalequal(scenario, causal_labels, output_filepath, random_generator)
        case "remove_static":
            _remove_static(scenario, output_filepath)
        case _:
            error_message = f"Benchmark: {benchmark} not supported!"
            raise ValueError(error_message)


def run(  # noqa: PLR0913
    causal_data_path: Path,
    output_data_path: Path,
    causal_labels_path: Path,
    benchmark: str,
    num_workers: int = 8,
    seed: int = 42,
) -> None:
    """Creates benchmark scenarios for Waymo dataset following CausalAgents strategy.

    Args:
        causal_data_path (Path): Path to the causal data.
        output_data_path (Path): Path to the output data.
        causal_labels_path (Path): Path to the causal labels.
        benchmark (str): Benchmark name.
        num_workers (int, optional): Number of parallel workers. Defaults to 8.
        seed (int, optional): Random seed. Defaults to 42.

    Raises:
        ValueError: If the raw data path does not exist.
    """
    scenario_mapping = {}
    filepaths = []
    for filepath in causal_data_path.rglob("*.pkl"):
        if "infos" in filepath.stem:
            continue
        scenario_id = filepath.stem
        scenario_mapping[scenario_id] = {
            "shard": filepath.parent.stem,
            "split": filepath.parent.parent.stem,
        }
        filepaths.append(filepath)

    # Create the benchmark subdirectories.
    proc_data_path = output_data_path / benchmark
    print(f"Processing Waymo benchmark: {benchmark}")
    splits = ["training", "validation", "testing"]
    shards = [f"shard_{i}" for i in range(10)]
    for split, shard in product(splits, shards):
        benchmark_subdir = proc_data_path / split / shard
        benchmark_subdir.mkdir(parents=True, exist_ok=True)
        print(f"Creating benchmark subdir: {benchmark_subdir}")

    # Create the scenario benchmark from the original WOMD subset.
    random_generator: Generator = default_rng(seed)
    # _create_scenario(filepaths[0], proc_data_path, causal_labels_path, scenario_mapping, benchmark, random_generator)
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(
            partial(
                _create_scenario,
                output_path=proc_data_path,
                causal_labels_path=causal_labels_path,
                scenario_mapping=scenario_mapping,
                benchmark=benchmark,
                random_generator=random_generator,
            ),
            [(file,) for file in tqdm(filepaths, total=len(filepaths))],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--causal_data_path",
        type=Path,
        default="/data/driving/waymo/processed/mini_causal/",
        help="Paths to the raw input data.",
    )
    parser.add_argument(
        "--output_data_path", type=Path, default="/data/driving/waymo/processed/", help="Paths to the output data."
    )
    parser.add_argument(
        "--causal_labels_path",
        type=Path,
        default="/data/driving/waymo/causal_agents/processed_labels/",
        help="Path to the causal labels.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="remove_causal",
        choices=["remove_causal", "remove_noncausal", "remove_noncausalequal", "remove_static"],
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run(**vars(args))
