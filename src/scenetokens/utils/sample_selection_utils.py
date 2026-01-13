"""Utility functions to perform model analysis. See 'docs/ANALYSIS.md' for details on usage."""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from omegaconf import DictConfig

from scenetokens.schemas import output_schemas as output
from scenetokens.utils.constants import SampleSelection
from scenetokens.utils.model_analysis_utils import (
    compute_alignment_scores,
    get_group_modes,
    get_scenario_classes_best_mode,
    get_tokenization_groups,
)


def random_selection(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """A sample selection strategy that randomly keeps a specified percentage of all scenarios.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.

    Returns:
        selected_samples (dict[str, Any]): a dictionary containing the IDs of the samples (scenarios) to keep or drop
            for training.
    """
    scenario_ids = list(model_outputs.keys())
    num_scenarios = len(scenario_ids)
    random.seed(config.seed)
    random.shuffle(scenario_ids)

    min_scenarios_to_keep = int(config.percentage_to_keep * num_scenarios)
    keep = scenario_ids[:min_scenarios_to_keep]
    drop = scenario_ids[min_scenarios_to_keep:]
    return {"keep": keep, "num_to_keep": len(keep), "drop": drop, "num_to_drop": len(drop)}


def random_selection_per_token(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """A sample selection strategy that randomly keeps a specified percentage of the scenarios for each class that has
    more than a desired minimum percentage.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.

    Returns:
        selected_samples (dict[str, Any]): a dictionary containing the IDs of the samples (scenarios) to keep or drop
            for training.
    """
    scenario_ids, scenario_classes, _, _ = get_scenario_classes_best_mode(model_outputs)
    num_scenarios, num_modes = scenario_classes.shape
    classes_dict = {"scenario_id": scenario_ids, "scenario_class": scenario_classes[:, 0]}
    classes_df = pd.DataFrame(classes_dict)

    percentage_per_class = (classes_df["scenario_class"].value_counts() / num_scenarios).to_frame(name="percentage")
    # NOTE: we drop less than num_scenarios_to_drop, because we don't ceil the number of samples to keep per class, to
    # favor tokens that are heavily underrepresented.
    num_scenarios_to_drop = int((1 - config.percentage_to_keep) * num_scenarios)

    min_percentage_per_class = config.min_percentage_per_class
    valid_percentages_per_class = percentage_per_class[percentage_per_class["percentage"] > min_percentage_per_class]
    total_valid_percentage = valid_percentages_per_class["percentage"].sum()

    selected_samples = {}
    for _, row in percentage_per_class.iterrows():
        scenario_class = row.name
        scenario_ids_in_class = classes_df["scenario_id"][classes_df["scenario_class"] == scenario_class].tolist()

        num_to_drop = (
            int(row.percentage * num_scenarios_to_drop / total_valid_percentage)
            if row.percentage > min_percentage_per_class
            else 0
        )

        random.seed(config.seed)
        random.shuffle(scenario_ids_in_class)
        if num_to_drop > 0:
            drop = scenario_ids_in_class[:num_to_drop]
            keep = scenario_ids_in_class[num_to_drop:]
            selected_samples[scenario_class] = {
                "keep": keep,
                "num_to_keep": len(keep),
                "drop": drop,
                "num_to_drop": len(drop),
            }
        else:
            selected_samples[scenario_class] = {
                "keep": scenario_ids_in_class,
                "num_to_keep": len(scenario_ids_in_class),
                "drop": [],
                "num_to_drop": 0,
            }

    # Get combined list
    keep = []
    drop = []
    for samples in selected_samples.values():
        keep += samples["keep"]
        drop += samples["drop"]
    selected_samples["keep"] = keep
    selected_samples["drop"] = drop
    selected_samples["num_to_keep"] = len(selected_samples["keep"])
    selected_samples["num_to_drop"] = len(selected_samples["drop"])
    return selected_samples


def weighted_sorting(
    samples: NDArray[Any], weights: NDArray[np.float64], *, sort_ascending: bool = True
) -> tuple[NDArray[np.int32], ...]:
    """Sorts the samples of an array using based on their weight values.

    Args:
        samples (NDArray[Any]): a numpy array containing samples.
        weights (NDArray[np.float64]): weights values in [0.0, 1.0] corresponding to each sample.
        sort_ascending (bool): if 'True' it sorts the samples in ascending order, based on the key values.

    Returns:
        samples (NDArray[np.int32]): the sorted samples.
        weights (NDArray[np.int32]): the sorted weights.
    """
    num_samples = len(samples)
    if num_samples != len(weights):
        error_messsage = f"Size of samples {num_samples} and weights {len(weights)} must be the same."
        raise ValueError(error_messsage)

    # Sort the sample indices based on the key values. If 'sort_ascending=True' higher priority values will show first.
    sorted_indices = np.argsort(weights) if sort_ascending else np.argsort(weights)[::-1]

    # Return the samples and weights sorted
    return samples[sorted_indices], weights[sorted_indices]


def weighted_sorting_gumbel(
    samples: NDArray[Any],
    weights: NDArray[np.float64],
    generator: Generator,
    *,
    sort_ascending: bool = True,
    large_exponent: np.float64 = np.inf,
) -> tuple[NDArray[np.int32], ...]:
    """Sorts the samples of an array using the Gumbel Max weighted sampling trick:
        https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/. Weights are assumed to be in [0, 1].

    Args:
        samples (NDArray[Any]): a numpy array containing samples.
        weights (NDArray[np.float64]): weights values in [0.0, 1.0] corresponding to each sample.
        generator (Generator): a random generator instance.
        sort_ascending (bool): if 'True' it sorts the samples in ascending order, based on the key values.
        large_exponent (np.float64): exponent value to use for samples whose weights are zero.

    Returns:
        samples (NDArray[np.int32]): the sorted samples.
        weights (NDArray[np.int32]): the sorted weights.
    """
    num_samples = len(samples)
    if num_samples != len(weights):
        error_messsage = f"Size of samples {num_samples} and weights {len(weights)} must be the same."
        raise ValueError(error_messsage)

    # Generate random numbers in [0, 1]
    uniform = generator.random(num_samples)

    # Calculate the exponent term (1 / W_i), if the weight of a sample is low its exponent to will be high.
    exponent = np.where(weights > 0.0, 1.0 / weights, large_exponent)

    # Calculate the priority values (uniform ** (1 / W_i)). Elements in 'uniform' raised to a large power (inf) will
    # result in 0.0.
    priority = uniform**exponent

    # Sort the sample indices based on the key values. If 'sort_ascending=False' higher priority values will show first.
    sorted_indices = np.argsort(priority) if sort_ascending else np.argsort(priority)[::-1]

    # Return the samples and weights sorted
    return samples[sorted_indices], weights[sorted_indices]


def alignment_based_selection_per_token(
    config: DictConfig, model_outputs: dict[str, output.ModelOutput]
) -> dict[str, Any]:
    """A sample selection strategy that randomly keeps a specified percentage of the scenarios for each class that has
    more than a desired minimum percentage.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.

    Returns:
        dict[str, Any]: a dictionary containing the IDs of the samples (scenarios) to keep or drop for training.
    """
    num_scenarios = len(model_outputs)
    # Get the groups by token and compute the each group's mode
    tokenization_groups, group_scenario_ids = get_tokenization_groups(config, model_outputs)
    group_modes = get_group_modes(tokenization_groups)

    # Compute the group percentages and get the valid percentages
    group_percentages = {
        base_token: len(token_group) / num_scenarios
        for base_token, token_group in tokenization_groups.items()
        if token_group is not None
    }
    min_percentage_per_class = config.min_percentage_per_class
    valid_percentages_per_class = {k: v for k, v in group_percentages.items() if v > min_percentage_per_class}
    total_valid_percentage = sum(valid_percentages_per_class.values())

    num_scenarios_to_drop = int((1 - config.percentage_to_keep) * num_scenarios)

    selected_samples = {}
    for base_token, token_group in tokenization_groups.items():
        if token_group is None:
            continue
        group_percentage = group_percentages[base_token]
        scenario_ids = group_scenario_ids[base_token].squeeze(axis=1)
        num_to_drop = (
            int(group_percentage * num_scenarios_to_drop / total_valid_percentage)
            if group_percentage > min_percentage_per_class
            else 0
        )

        if num_to_drop > 0:
            # generator instance
            generator = default_rng(config.seed)
            scores = compute_alignment_scores(group_modes[base_token], token_group, config.alignment_strategy)
            # Get the scenario IDs (pseudo-randomly) sorted by their score, prioritizing samples with lower alignment to
            # the target value, i.e., samples that are potentially rare w.r.t their group.
            if config.sorting_strategy == "gumbel":
                # Highly aligned instances will be given a score of (1-score) to deprioritize them for selection, but we
                # don't want to be too harsh so as to completely zero-out-them (large_exponent=8).
                # Since we want to drop the samples at the beginning of the sorted list and we're using the inverse of
                # the score, we sort in ascending order.
                sorted_scenario_ids, _ = weighted_sorting_gumbel(
                    scenario_ids, 1.0 - scores, generator, sort_ascending=True, large_exponent=8.0
                )
            else:
                # Since we want to drop the samples at the beginning of the sorted list and we're using the scores
                # directly, we sort in descending order.
                sorted_scenario_ids, _ = weighted_sorting(scenario_ids, scores, sort_ascending=False)

            drop = sorted_scenario_ids[:num_to_drop].tolist()
            keep = sorted_scenario_ids[num_to_drop:].tolist()
            selected_samples[base_token] = {
                "keep": keep,
                "num_to_keep": len(keep),
                "drop": drop,
                "num_to_drop": len(drop),
            }
        else:
            selected_samples[base_token] = {
                "keep": scenario_ids.tolist(),
                "num_to_keep": len(scenario_ids),
                "drop": [],
                "num_to_drop": 0,
            }

    keep = []
    drop = []
    for samples in selected_samples.values():
        keep += samples["keep"]
        drop += samples["drop"]
    selected_samples["keep"] = keep
    selected_samples["drop"] = drop
    selected_samples["num_to_keep"] = len(selected_samples["keep"])
    selected_samples["num_to_drop"] = len(selected_samples["drop"])
    return selected_samples


def run_sample_selection(config: DictConfig, model_outputs: dict[str, output.ModelOutput], output_path: Path) -> None:
    """Wrapper function which runs a specified sample selection strategy. A sample selection strategy produces a
    dictionary containing the a set of training scenarios to keep and to drop.

    Args:
        config (DictConfig): encapsulates model analysis configuration parameters.
        model_outputs (dict[str, output.ModelOutput]): a dictionary containing model outputs per scenario.
        output_path (Path): output path where visualization will be saved to.
    """
    selection_strategy = SampleSelection(config.selection_strategy)
    match selection_strategy:
        case SampleSelection.RANDOM_DROP:
            sample_selection = random_selection(config, model_outputs)
        case SampleSelection.TOKEN_RANDOM_DROP:
            sample_selection = random_selection_per_token(config, model_outputs)
        case SampleSelection.SIMPLE_TOKEN_JACCARD_DROP:
            config.sorting_strategy = "simple"
            config.alignment_strategy = "jaccard"
            sample_selection = alignment_based_selection_per_token(config, model_outputs)
        case SampleSelection.SIMPLE_TOKEN_HAMMING_DROP:
            config.sorting_strategy = "simple"
            config.alignment_strategy = "hamming"
            sample_selection = alignment_based_selection_per_token(config, model_outputs)
        case SampleSelection.GUMBEL_TOKEN_JACCARD_DROP:
            config.sorting_strategy = "gumbel"
            config.alignment_strategy = "jaccard"
            sample_selection = alignment_based_selection_per_token(config, model_outputs)
        case SampleSelection.GUMBEL_TOKEN_HAMMING_DROP:
            config.sorting_strategy = "gumbel"
            config.alignment_strategy = "hamming"
            sample_selection = alignment_based_selection_per_token(config, model_outputs)
        case _:
            error_message = f"Unsupported selection strategy: {selection_strategy}"
            raise ValueError(error_message)

    output_filepath = output_path / f"sample_selection_{selection_strategy.value}_{config.percentage_to_keep}.json"
    with output_filepath.open("w") as f:
        json.dump(sample_selection, f, indent=2)
