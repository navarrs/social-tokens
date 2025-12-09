import itertools
import pickle
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path
from time import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


def verify(data_path: Path, scores_path: Path, prefix: str) -> None:
    """The script verifies that the splits and corresponding scores from SafeShift were correctly organized.

    Args:
        data_path (Path): Path to the processed scenarios from SafeShift.
        scores_path (Path): Path to the SafeShift scores.
        output_path (Path): Path to save the output plot.
        prefix (str): Prefix of the SafeShift scores to analyze
    """
    # Verify there's no intersections between the train/val/test scenarios resulting from 'resplit_safeshift.py`.
    splits = ["training", "testing", "validation"]

    # Get the scenario IDs.
    split_data = {}
    for split in splits:
        split_path = data_path / split
        split_data[split] = {str(p).split("/")[-1] for p in split_path.rglob("*/*.pkl") if p.is_file()}

    # Check for intersections (ideally, we get 0).
    for set1, set2 in itertools.combinations(splits, 2):
        intersection = list(split_data[set1] & split_data[set2])
        print(f"Intersection between sets ({set1},{set2}) is: {len(intersection)}")

    # Verify the score density plots for the SafeShift set are correct. For the train/val sets the density over the
    # scores should be very similar, whereas for the testing set we should get a density over higher scores.
    scores_ac, scores_fe = {}, {}
    colors = {"training": "green", "testing": "red", "validation": "blue"}
    for split in splits:
        print(f"Split: {split}")
        split_infos = "test" if split == "testing" else "val" if split == "validation" else "training"
        scenario_metadata_filepath = scores_path / f"{prefix}extra_processed_scenarios_{split_infos}_infos.pkl"

        with scenario_metadata_filepath.open("rb") as f:
            scenario_metadata = pickle.load(f)

        scores_ac[split] = np.asarray([scenario["traj_scores_asym_combined"].max() for scenario in scenario_metadata])
        scores_fe[split] = np.asarray([scenario["traj_scores_fe"].max() for scenario in scenario_metadata])

        # Create a density plot over the loaded scores
        sns.kdeplot(scores_ac[split], fill=True, color=colors[split], label=split + "_ac")
        sns.kdeplot(scores_fe[split], fill=True, color=colors[split], label=split + "_fe", alpha=0.7)

    plt.title("Score density plot")
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig("scores_pdf.png")


def _process_scenario(scenario_id: str, input_filepath: Path, destination_path: Path) -> None:
    """Copies a file from a input path to an destination path and then unlinks the source one.

    Args:
        scenario_id (str): ID of the scenario in the source directory to be copied to the destination path.
        input_filepath (Path): Filepath to the input scenario.
        destination_path (Path): Path where the scenario will be copied to.

    """
    if not input_filepath.exists():
        # This will print in the parallel process
        print(f"Warning file: {input_filepath} not found!")
        return  # Skip if file not found

    output_filepath = destination_path / f"{scenario_id}.pkl"
    shutil.copy(input_filepath, output_filepath)
    input_filepath.unlink()


def _star_process_scenario(args) -> None:  # noqa: ANN001
    """Helper function to unpack arguments for multiprocessing.

    Args:
        args: Tuple of arguments to be passed to _process_scenario.
    """
    return _process_scenario(*args)


def run(  # noqa: PLR0913
    scores_path: Path,
    scenarios_path: Path,
    output_path: Path,
    prefix: str,
    num_shards: int,
    n_jobs: int = 2,
) -> None:
    """Preprocess Waymo scenario protos from SafeShift using multiprocessing.

    Args:
        scores_path: Path to the SafeShift score metadata.
        scenarios_path: Path to the scenarios to be sorted.
        output_path: Path to store the processed data.
        prefix: Prefix of the SafeShift score metadata. It indicates the type of scoring strategy utilized to sort the
            original WOMD dataset.
        num_shards: Number of shards to store the data.
        n_jobs: Number of worker processes. Defaults to all available CPU cores.
    """
    # Get the paths to all the data
    scenario_filepaths = {path.stem: path for path in scenarios_path.rglob("*.pkl")}

    # Prepare the list of tasks (arguments for _process_scenario)
    tasks = []

    # Setup paths and load metadata (Sequential part)
    for split in ["validation", "training", "testing"]:
        print(f"Processing split: {split}")
        split_infos = "test" if split == "testing" else "val" if split == "validation" else "training"

        scenario_metadata_filepath = scores_path / f"{prefix}processed_scenarios_{split_infos}_infos.pkl"
        if not scenario_metadata_filepath.exists():
            error_message = f"Scenario metadata file not found: {scenario_metadata_filepath}"
            raise FileNotFoundError(error_message)

        with scenario_metadata_filepath.open("rb") as f:
            scenario_metadata: list[dict[str, Any]] = pickle.load(f)
        print(f"\tLoaded {len(scenario_metadata)} scenarios from {scenario_metadata_filepath}")

        output_split_path = output_path / split
        output_split_path.mkdir(parents=True, exist_ok=True)

        shard_size = (len(scenario_metadata) + num_shards - 1) // num_shards
        split_tasks = []
        num_not_found = 0
        print("\tPreparing tasks for parallel execution...")
        for shard_idx in tqdm(range(num_shards), desc="\tCollecting tasks"):
            shard_dir = output_split_path / f"shard_{shard_idx}"
            shard_dir.mkdir(parents=True, exist_ok=True)  # Ensure shard directory exists

            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, len(scenario_metadata))

            # Create a tuple of arguments for each scenario
            for scenario in scenario_metadata[start_idx:end_idx]:
                # The arguments must be pickleable: scenario, scenarios_path, shard_dir
                scenario_id = scenario["scenario_id"]
                scenario_filepath = scenario_filepaths.get(scenario_id)
                if scenario_filepath is None:
                    # print(f"Warning: Scenario ID {scenario_id} not found in scenario filepaths.")
                    num_not_found += 1
                    continue
                split_tasks.append((scenario_id, scenario_filepath, shard_dir))

        print(f"\tTotal tasks prepared: {len(split_tasks)} for split {split}.")
        print(f"\tNumber of scenarios not found: {num_not_found} for split {split}.")
        tasks.extend(split_tasks)

    # Use all available cores if n_jobs is not specified
    if n_jobs == -1:
        n_jobs = cpu_count()

    print(f"Starting parallel processing with {n_jobs} workers...")
    start_time = time()
    with Pool(processes=n_jobs) as pool:
        list(
            tqdm(
                pool.imap_unordered(_star_process_scenario, tasks),
                total=len(tasks),
                desc="Processing scenarios in parallel",
            )
        )
    print(f"Finished processing and sharding data in {time() - start_time:.2f} seconds.")

    # Verify the splits were correctly created
    verify(output_path, scores_path, prefix)


if __name__ == "__main__":
    """Entry point for preprocessing Waymo scenario protos from SafeShift. """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores_path", type=Path, default="/datasets/waymo/mtr_process_splits", help="Path to the SafeShift data."
    )
    parser.add_argument(
        "--scenarios_path", type=Path, default="/datasets/scenarios/safeshift_all", help="Path to the SafeShift data."
    )
    parser.add_argument(
        "--output_path", type=Path, default="/datasets/waymo/processed/safeshift", help="Path to the output data."
    )
    parser.add_argument("--prefix", type=str, default="score_asym_combined_80_", help="Prefix for the input files.")
    parser.add_argument("--num_shards", type=int, default=10, help="number of shards to store the data")
    parser.add_argument("--n_jobs", type=int, default=20, help="Number of parallel worker processes to use.")
    args = parser.parse_args()
    run(**vars(args))
