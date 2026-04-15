"""Token-based sample selection strategies."""

import random
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from scenetokens.sample_selection.common import (
    aggregate_selected_samples,
    allocate_removal_budget,
    compute_proportional_number_to_drop,
    greedy_select_from_sim_matrix,
    make_group_result,
    sort_ids_by_score,
)
from scenetokens.schemas import output_schemas as output
from scenetokens.utils import metrics as metrics_utils
from scenetokens.utils.model_analysis_utils import (
    compute_alignment_scores,
    get_group_modes,
    get_scenario_classes_best_mode,
    get_tokenization_groups,
)


def random_selection_per_token(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """A sample selection strategy that randomly keeps a specified percentage of the scenarios for each class that has
    more than a desired minimum percentage.

    Args:
        config: encapsulates model analysis configuration parameters.
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        selected_samples: a dictionary containing the IDs of the samples (scenarios) to keep or drop for training.
    """
    scenario_ids, scenario_classes, _, _ = get_scenario_classes_best_mode(model_outputs)
    num_scenarios, _ = scenario_classes.shape
    classes_df = pd.DataFrame({"scenario_id": scenario_ids, "scenario_class": scenario_classes[:, 0]})

    percentage_per_class = (classes_df["scenario_class"].value_counts() / num_scenarios).to_frame(name="percentage")

    # NOTE: we drop less than num_scenarios_to_drop, because we don't ceil the number of samples to keep per class to
    # favor tokens that are heavily underrepresented.
    num_scenarios_to_drop = int((1 - config.percentage_to_keep) * num_scenarios)

    min_percentage_per_class = config.min_percentage_per_class
    valid_percentages_per_class = percentage_per_class[percentage_per_class["percentage"] > min_percentage_per_class]
    total_valid_percentage = valid_percentages_per_class["percentage"].sum()

    selected_samples = {}
    for _, row in percentage_per_class.iterrows():
        scenario_class = row.name
        scenario_ids_in_class = classes_df["scenario_id"][classes_df["scenario_class"] == scenario_class].tolist()

        num_to_drop = compute_proportional_number_to_drop(
            num_scenarios_to_drop, row.percentage, min_percentage_per_class, total_valid_percentage
        )

        random.seed(config.seed)
        random.shuffle(scenario_ids_in_class)
        if num_to_drop > 0:
            selected_samples[scenario_class] = make_group_result(
                keep=scenario_ids_in_class[num_to_drop:],
                drop=scenario_ids_in_class[:num_to_drop],
            )
        else:
            selected_samples[scenario_class] = make_group_result(keep=scenario_ids_in_class, drop=[])

    aggregate_selected_samples(selected_samples)
    return selected_samples


def alignment_based_selection_per_token(
    config: DictConfig, model_outputs: dict[str, output.ModelOutput]
) -> dict[str, Any]:
    """A sample selection strategy that keeps a specified percentage of the scenarios for each tokenization group,
    prioritizing retention of samples that are least aligned to the group mode. Scenarios with high alignment
    (typical/redundant w.r.t. the group) are dropped first. Supports both Gumbel-weighted (stochastic) and simple
    deterministic sorting strategies.

    Args:
        config: encapsulates model analysis configuration parameters.
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        a dictionary containing the IDs of the samples (scenarios) to keep or drop for training.
    """
    num_scenarios = len(model_outputs)
    # Get the groups by token and compute each group's mode
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
        num_to_drop = compute_proportional_number_to_drop(
            num_scenarios_to_drop, group_percentage, min_percentage_per_class, total_valid_percentage
        )

        if num_to_drop > 0:
            scores = compute_alignment_scores(
                group_modes[base_token].tolist(), token_group.astype(np.int32), config.alignment_strategy
            )
            sorted_scenario_ids, _ = sort_ids_by_score(scenario_ids, scores, config.sorting_strategy, config.seed)
            selected_samples[base_token] = make_group_result(
                keep=sorted_scenario_ids[num_to_drop:].tolist(),
                drop=sorted_scenario_ids[:num_to_drop].tolist(),
            )
        else:
            selected_samples[base_token] = make_group_result(keep=scenario_ids.tolist(), drop=[])

    aggregate_selected_samples(selected_samples)
    return selected_samples


def vocab_token_selection(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """Sample selection using tokenization groups and greedy diversity-based selection.

    This is the token-based analogue of vocab_cluster_selection. Instead of fitting KMeans on embeddings to produce
    cluster groups, it uses the model's learned token vocabulary directly, grouping scenarios by their best-mode token.

    Algorithm:
        1. Use get_tokenization_groups() to group scenarios by best-mode token. Each scenario's token sequence across
            all modes forms a pseudo-vocabulary of length M (one token per mode, sorted by descending mode probability).
        2. Allocate a removal budget per token group proportional to group size, so larger groups are pruned more
            aggressively.
        3. Within each group, compute a pairwise similarity matrix over token sequences using the configured alignment
            strategy ('hamming' or 'jaccard') and apply greedy submodular selection to retain the most diverse
            scenarios.

    Args:
        config: encapsulates model analysis configuration parameters. Requires: percentage_to_keep (float),
            alignment_strategy (str, 'hamming' or 'jaccard'), seed (int).
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        selected_samples: a dictionary containing the IDs of the samples to keep or drop.
    """
    num_scenarios = len(model_outputs)
    tokenization_groups, group_scenario_ids = get_tokenization_groups(config, model_outputs)

    group_sizes = {token: ids.shape[0] for token, ids in group_scenario_ids.items() if ids is not None}
    total_removal = int((1 - config.percentage_to_keep) * num_scenarios)
    removal_budgets = allocate_removal_budget(group_sizes, total_removal)

    alignment_strategy = config.get("alignment_strategy", "hamming")
    match alignment_strategy:
        case "hamming":
            compute_sim_matrix = metrics_utils.compute_pairwise_hamming_similarity
        case "jaccard":
            compute_sim_matrix = metrics_utils.compute_pairwise_jaccard_similarity
        case _:
            error_message = f"Unsupported alignment_strategy '{alignment_strategy}'. Choose 'hamming' or 'jaccard'."
            raise ValueError(error_message)

    selected_samples: dict[Any, Any] = {}
    for token_id, token_group in tokenization_groups.items():
        if token_group is None:
            continue
        scenario_ids_arr = group_scenario_ids[token_id].squeeze(axis=1)
        num_to_remove = removal_budgets.get(token_id, 0)
        num_to_keep = len(scenario_ids_arr) - num_to_remove

        if num_to_remove == 0:
            selected_samples[token_id] = make_group_result(keep=scenario_ids_arr.tolist(), drop=[])
            continue

        sim_matrix = compute_sim_matrix(token_group)
        keep, drop = greedy_select_from_sim_matrix(scenario_ids_arr, sim_matrix, num_to_keep)
        selected_samples[token_id] = make_group_result(keep=keep, drop=drop)

    aggregate_selected_samples(selected_samples)
    return selected_samples
