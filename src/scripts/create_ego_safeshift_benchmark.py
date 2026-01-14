import multiprocessing
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng
from tqdm import tqdm


def _get_scenario_mapping(
    scenario_ids: list[str],
    output_data_path: Path,
    split: str,
) -> dict[str, Path]:
    """Creates a mapping from scenario IDs to output file paths.

    Args:
        scenario_ids (list[str]): List of scenario IDs.
        output_data_path (Path): Path to the output data.
        split (str): Data split (e.g., 'training', 'validation', 'testing').

    Returns:
        dict[str, Path]: Mapping from scenario IDs to output file paths.
    """
    scenario_mapping = {}
    for scenario_id in scenario_ids:
        output_filepath = output_data_path / split / f"{scenario_id}"
        scenario_mapping[scenario_id] = output_filepath
    return scenario_mapping


def _copy_scenario(
    scenario_id: str, input_scenario_mapping: dict[str, Path], output_scenario_mapping: dict[str, Path]
) -> None:
    """Copies a scenario from the input mapping to the output mapping.

    Args:
        scenario_id (str): Scenario ID.
        input_scenario_mapping (dict[str, Path]): Input scenario mapping.
        output_scenario_mapping (dict[str, Path]): Output scenario mapping.
    """
    if scenario_id not in input_scenario_mapping:
        print(f"Scenario {scenario_id} not found in input mapping.")
        return
    input_filepath = input_scenario_mapping[scenario_id]
    output_filepath = output_scenario_mapping[scenario_id]
    shutil.copy2(input_filepath, output_filepath)


def run(  # noqa: PLR0913
    causal_data_path: Path,
    output_data_path: Path,
    scenario_score_mapping_filepath: Path,
    score_type: str = "gt_critical_continuous_safeshift",
    cutoff_percentile: float = 80.0,
    validation_percentage: float = 10.0,
    num_workers: int = 8,
    seed: int = 42,
) -> None:
    """Creates benchmark scenarios for Waymo dataset following CausalAgents strategy.

    Args:
        causal_data_path (Path): Path to the causal data.
        output_data_path (Path): Path to the output data.
        scenario_score_mapping_filepath (Path): Path to the file with scenario score mappings.
        score_type (str, optional): Type of score to use for filtering scenarios. Defaults to
            'gt_critical_continuous_safeshift'.
        cutoff_percentile (float, optional): Cutoff percentile for filtering scenarios. Defaults to 80.0.
        validation_percentage (float, optional): Percentage of data to use for validation. Defaults to 10.0.
        num_workers (int, optional): Number of parallel workers. Defaults to 8.
        seed (int, optional): Random seed. Defaults to 42.

    Raises:
        ValueError: If the raw data path does not exist.
    """
    random_generator: Generator = default_rng(seed)
    input_scenario_mapping = {}
    for filepath in causal_data_path.rglob("*.pkl"):
        if "infos" in filepath.stem:
            continue
        scenario_id = filepath.name
        input_scenario_mapping[scenario_id] = filepath

    # Create the benchmark subdirectories.
    print("Processing Ego-SafeShift benchmark")
    splits = ["training", "validation", "testing"]
    for split in splits:
        benchmark_subdir = output_data_path / split
        benchmark_subdir.mkdir(parents=True, exist_ok=True)
        print(f"Creating benchmark subdir: {benchmark_subdir}")

    scenario_scores_df = pd.read_csv(scenario_score_mapping_filepath)
    cutoff_score = np.percentile(scenario_scores_df[score_type], cutoff_percentile)

    output_scenario_mapping = {}

    # Get the training and validation scenarios as those below the cutoff score.
    train_val_scenarios = scenario_scores_df[scenario_scores_df[score_type] < cutoff_score]["scenario_ids"].tolist()
    random_generator.shuffle(train_val_scenarios)
    num_validation_scenarios = int(len(train_val_scenarios) * (validation_percentage / 100.0))
    validation_scenarios = train_val_scenarios[:num_validation_scenarios]
    validation_scenario_mapping = _get_scenario_mapping(validation_scenarios, output_data_path, "validation")
    output_scenario_mapping.update(validation_scenario_mapping)

    training_scenarios = train_val_scenarios[num_validation_scenarios:]
    training_scenario_mapping = _get_scenario_mapping(training_scenarios, output_data_path, "training")
    output_scenario_mapping.update(training_scenario_mapping)

    # Get the testing scenarios as those above the cutoff score.
    testing_scenarios = scenario_scores_df[scenario_scores_df[score_type] >= cutoff_score]["scenario_ids"].tolist()
    testing_scenario_mapping = _get_scenario_mapping(testing_scenarios, output_data_path, "testing")
    output_scenario_mapping.update(testing_scenario_mapping)

    # Create the scenario benchmark from the original WOMD subset.
    with multiprocessing.Pool(num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    partial(
                        _copy_scenario,
                        input_scenario_mapping=input_scenario_mapping,
                        output_scenario_mapping=output_scenario_mapping,
                    ),
                    list(output_scenario_mapping.keys()),
                ),
                total=len(output_scenario_mapping),
                desc="Copying scenarios",
            )
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--causal_data_path",
        type=Path,
        default="/datasets/driving/waymo/processed/mini_causal/",
        help="Paths to the raw input data.",
    )
    parser.add_argument(
        "--output_data_path",
        type=Path,
        default="/datasets/driving/waymo/processed/causal_ego_safeshift",
        help="Paths to the output data.",
    )
    parser.add_argument(
        "--scenario_score_mapping_filepath",
        type=Path,
        default="../../meta/scenario_to_scores_mapping.csv",
        help="Path to the file with scenario score mappings.",
    )
    parser.add_argument(
        "--score_type",
        type=str,
        default="gt_critical_continuous_safeshift",
        help="Type of score to use for filtering scenarios.",
    )
    parser.add_argument(
        "--cutoff_percentile",
        type=float,
        default=80.0,
        help="Cutoff percentile for filtering scenarios.",
    )
    parser.add_argument(
        "--validation_percentage", type=float, default=10.0, help="Percentage of data to use for validation"
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run(**vars(args))
