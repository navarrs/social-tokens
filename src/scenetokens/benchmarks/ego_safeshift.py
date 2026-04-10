r"""Benchmark creation for the Ego-SafeShift benchmark.

Filters scenarios by safety score percentile and copies them into training/validation/testing splits. Scenarios below
the cutoff percentile form the train/val pool; scenarios above form the test set.

Example usage:

    uv run -m scenetokens.create_benchmark benchmark=ego_safeshift \\
        input_data_path=/datasets/waymo/processed/mini_causal \\
        output_data_path=/datasets/waymo/processed/causal_ego_safeshift \\
        scenario_score_mapping_filepath=meta/scenario_to_scores_mapping.csv

See configs/benchmark/ego_safeshift.yaml for all available options.
"""

import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng
from omegaconf import DictConfig
from tqdm import tqdm

from scenetokens.benchmarks.common import (
    collect_scenario_filepaths,
    copy_scenario,
    create_split_dirs,
    get_scenario_mapping,
)


def create_ego_safeshift_benchmark(config: DictConfig) -> None:
    """Creates benchmark scenarios for the Ego-SafeShift benchmark.

    Splits the dataset by safety score percentile: scenarios below cutoff_percentile go into the training/validation
    pool (split 90/10 by default); scenarios above go into the test set. Files are copied from causal_data_path into the
    appropriate split subdirectory.

    Args:
        config: Hydra config.
            Expected keys: input_data_path, output_data_path, scenario_score_mapping_filepath, score_type,
            cutoff_percentile, validation_percentage, num_workers, seed.
    """
    output_data_path = Path(config.output_data_path)
    random_generator: Generator = default_rng(config.seed)

    filepaths = collect_scenario_filepaths(Path(config.input_data_path))
    input_scenario_mapping = {fp.name: fp for fp in filepaths}

    print("Processing Ego-SafeShift benchmark")
    create_split_dirs(output_data_path)

    scenario_scores_df = pd.read_csv(Path(config.scenario_score_mapping_filepath))
    cutoff_score = np.percentile(scenario_scores_df[config.score_type], config.cutoff_percentile)

    output_scenario_mapping: dict[str, Path] = {}

    train_val_scenarios = scenario_scores_df[scenario_scores_df[config.score_type] < cutoff_score][
        "scenario_ids"
    ].tolist()
    random_generator.shuffle(train_val_scenarios)
    num_validation_scenarios = int(len(train_val_scenarios) * (config.validation_percentage / 100.0))

    validation_scenarios = train_val_scenarios[:num_validation_scenarios]
    output_scenario_mapping.update(get_scenario_mapping(validation_scenarios, output_data_path, "validation"))

    training_scenarios = train_val_scenarios[num_validation_scenarios:]
    output_scenario_mapping.update(get_scenario_mapping(training_scenarios, output_data_path, "training"))

    testing_scenarios = scenario_scores_df[scenario_scores_df[config.score_type] >= cutoff_score][
        "scenario_ids"
    ].tolist()
    output_scenario_mapping.update(get_scenario_mapping(testing_scenarios, output_data_path, "testing"))

    with multiprocessing.Pool(config.num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    partial(
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
