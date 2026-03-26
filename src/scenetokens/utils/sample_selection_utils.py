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
from sklearn.cluster import KMeans

from scenetokens.schemas import output_schemas as output
from scenetokens.utils.constants import INVALID_AGENT_ID, SampleSelection
from scenetokens.utils.metric_utils import compute_cosine_similarity, compute_pairwise_cosine_similarity
from scenetokens.utils.model_analysis_utils import (
    compute_alignment_scores,
    get_group_modes,
    get_scenario_classes_best_mode,
    get_scenario_dec_embeddings,
    get_tokenization_groups,
)


# Exponent used in Gumbel sorting for zero-weight samples — large enough to strongly deprioritize
# them without completely zeroing out their probability via uniform**inf.
_GUMBEL_LARGE_EXPONENT: float = 8.0


def _aggregate_selected_samples(selected_samples: dict[Any, Any]) -> None:
    """Helper function to aggregate the sample IDs to keep and drop across groups (tokens or clusters) into a single
    list of samples to keep and drop. Mutates the input dictionary in place.

    Args:
        selected_samples: a dictionary containing the sample selection results per group.
    """
    keep = []
    drop = []
    for samples in selected_samples.values():
        keep += samples["keep"]
        drop += samples["drop"]
    selected_samples["keep"] = keep
    selected_samples["drop"] = drop
    selected_samples["num_to_keep"] = len(keep)
    selected_samples["num_to_drop"] = len(drop)


def _make_group_result(keep: list[Any], drop: list[Any]) -> dict[str, Any]:
    """Constructs a per-group selection result dict."""
    return {"keep": keep, "num_to_keep": len(keep), "drop": drop, "num_to_drop": len(drop)}


def _compute_proportional_number_to_drop(
    total_number_to_drop: int, percentage: float, min_percentage: float, total_valid_percentage: float
) -> int:
    """Computes the proportional number of samples to drop for a group.

    Args:
        total_number_to_drop: the total number of samples to drop across all groups.
        percentage: the percentage of samples in the group.
        min_percentage: the min percentage threshold for a group to be considered valid for dropping samples.
        total_valid_percentage: the total percentage of samples across all valid groups.

    Returns:
        0 if percentage does not exceed min_percentage, favoring underrepresented groups by flooring rather than
        ceiling the per-group drop count.
    """
    return int(percentage * total_number_to_drop / total_valid_percentage) if percentage > min_percentage else 0


def random_selection(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """A sample selection strategy that randomly keeps a specified percentage of all scenarios.

    Args:
        config: encapsulates model analysis configuration parameters.
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        A dictionary containing the IDs of the samples (scenarios) to keep or drop for training.
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

        num_to_drop = _compute_proportional_number_to_drop(
            num_scenarios_to_drop, row.percentage, min_percentage_per_class, total_valid_percentage
        )

        random.seed(config.seed)
        random.shuffle(scenario_ids_in_class)
        if num_to_drop > 0:
            selected_samples[scenario_class] = _make_group_result(
                keep=scenario_ids_in_class[num_to_drop:],
                drop=scenario_ids_in_class[:num_to_drop],
            )
        else:
            selected_samples[scenario_class] = _make_group_result(keep=scenario_ids_in_class, drop=[])

    _aggregate_selected_samples(selected_samples)
    return selected_samples


def weighted_sorting(
    samples: NDArray[Any], weights: NDArray[np.float64], *, sort_ascending: bool = True
) -> tuple[NDArray[Any], NDArray[np.float64]]:
    """Sorts the samples of an array using based on their weight values.

    Args:
        samples: a numpy array containing samples.
        weights: weights values in [0.0, 1.0] corresponding to each sample.
        sort_ascending: if 'True' it sorts the samples in ascending order so the lowest weight values appear first.

    Returns:
        samples: the sorted samples.
        weights: the sorted weights.
    """
    if len(samples) != len(weights):
        error_message = f"Size of samples {len(samples)} and weights {len(weights)} must be the same."
        raise ValueError(error_message)

    # Sort the sample indices based on the key values. When sort_ascending=True, the lowest weight values appear first.
    sorted_indices = np.argsort(weights) if sort_ascending else np.argsort(weights)[::-1]

    return samples[sorted_indices], weights[sorted_indices]


def weighted_sorting_gumbel(
    samples: NDArray[Any],
    weights: NDArray[np.float64],
    generator: Generator,
    *,
    sort_ascending: bool = True,
    large_exponent: float = np.inf,
) -> tuple[NDArray[Any], NDArray[np.float64]]:
    """Sorts the samples of an array using the Gumbel Max weighted sampling trick:
        https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/. Weights are assumed to be in [0, 1].

    Args:
        samples: a numpy array containing samples.
        weights: weights values in [0.0, 1.0] corresponding to each sample.
        generator: a random generator instance.
        sort_ascending: if 'True' it sorts the samples in ascending order, based on the key values.
        large_exponent: exponent value to use for samples whose weights are zero.

    Returns:
        samples: the sorted samples.
        weights: the sorted weights.
    """
    if len(samples) != len(weights):
        error_message = f"Size of samples {len(samples)} and weights {len(weights)} must be the same."
        raise ValueError(error_message)

    # Generate random numbers in [0, 1]
    uniform = generator.random(len(samples))

    # Calculate the exponent term (1 / W_i), if the weight of a sample is low its exponent to will be high.
    exponent = np.where(weights > 0.0, 1.0 / weights, large_exponent)

    # Calculate the priority values (uniform ** (1 / W_i)). Elements in 'uniform' raised to a large power (inf) will
    # result in 0.0.
    priority = uniform**exponent

    # Sort the sample indices based on the key values. If 'sort_ascending=False' higher priority values will show first.
    sorted_indices = np.argsort(priority) if sort_ascending else np.argsort(priority)[::-1]

    return samples[sorted_indices], weights[sorted_indices]


def _sort_ids_by_score(
    ids: NDArray[Any],
    scores: NDArray[np.float64],
    sorting_strategy: str,
    seed: int,
) -> tuple[NDArray[Any], NDArray[np.float64]]:
    """Sorts IDs by score so that the lowest-priority candidates (to drop) appear first.

    For 'gumbel': uses (1 - scores) with the Gumbel Max trick so high-scoring (typical) samples are softly
    deprioritized. For other strategies: sorts by raw scores descending so the highest-scoring IDs appear last.
    """
    if sorting_strategy == "gumbel":
        generator = default_rng(seed)
        return weighted_sorting_gumbel(
            ids, 1.0 - scores, generator, sort_ascending=True, large_exponent=_GUMBEL_LARGE_EXPONENT
        )
    return weighted_sorting(ids, scores, sort_ascending=False)


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
        num_to_drop = _compute_proportional_number_to_drop(
            num_scenarios_to_drop, group_percentage, min_percentage_per_class, total_valid_percentage
        )

        if num_to_drop > 0:
            scores = compute_alignment_scores(
                group_modes[base_token].tolist(), token_group.astype(np.int32), config.alignment_strategy
            )
            sorted_scenario_ids, _ = _sort_ids_by_score(scenario_ids, scores, config.sorting_strategy, config.seed)
            selected_samples[base_token] = _make_group_result(
                keep=sorted_scenario_ids[num_to_drop:].tolist(),
                drop=sorted_scenario_ids[:num_to_drop].tolist(),
            )
        else:
            selected_samples[base_token] = _make_group_result(keep=scenario_ids.tolist(), drop=[])

    _aggregate_selected_samples(selected_samples)
    return selected_samples


def _fit_kmeans(
    embeddings: NDArray[np.float64],
    num_clusters: int,
    seed: int,
) -> tuple[KMeans, NDArray[np.int32]]:
    """Fits a KMeans model and returns both the fitted model and the cluster labels.

    The fitted model is returned so callers can access cluster_centers_ (centroids).
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init="auto")
    cluster_labels: NDArray[np.int32] = kmeans.fit_predict(embeddings)
    return kmeans, cluster_labels


def _cluster_embeddings(config: DictConfig, embeddings: NDArray[np.float64]) -> tuple[Any, NDArray[np.int32]]:
    """Resolves the clustering strategy from config, validates inputs, and fits the model.

    Args:
        config: must contain `clustering_strategy` and `seed`; optionally `num_clusters` (default 100).
        embeddings: array of shape (num_embeddings, embedding_dim).

    Returns:
        A (kmeans, cluster_labels) tuple from the fitted model.

    Raises:
        ValueError: if num_embeddings <= num_clusters or the clustering strategy is unsupported.
    """
    match config.clustering_strategy:
        case "kmeans":
            num_clusters = config.get("num_clusters", 100)
            if len(embeddings) <= num_clusters:
                error_message = f"num_embeddings ({len(embeddings)}) must be greater than num_clusters ({num_clusters})"
                raise ValueError(error_message)
            return _fit_kmeans(embeddings, num_clusters, config.seed)
        case _:
            error_message = f"Unsupported clustering strategy: {config.clustering_strategy}"
            raise ValueError(error_message)


def random_selection_per_cluster(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """A sample selection strategy that clusters scenario_dec embeddings using a clustering algorithm (currently only
    K-Means is supported) and randomly drops samples per cluster proportional to each cluster's size, mirroring the
    logic of random_selection_per_token.

    Args:
        config: encapsulates model analysis configuration parameters.
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        A dictionary containing the IDs of the samples to keep or drop.
    """
    scenario_ids, embeddings = get_scenario_dec_embeddings(model_outputs)
    num_scenarios = len(scenario_ids)

    _, cluster_labels = _cluster_embeddings(config, embeddings)

    clusters_df = pd.DataFrame({"scenario_id": scenario_ids, "cluster": cluster_labels})
    percentage_per_cluster = (clusters_df["cluster"].value_counts() / num_scenarios).to_frame(name="percentage")

    num_scenarios_to_drop = int((1 - config.percentage_to_keep) * num_scenarios)
    min_percentage_per_class = config.min_percentage_per_class
    valid_percentages = percentage_per_cluster[percentage_per_cluster["percentage"] > min_percentage_per_class]
    total_valid_percentage = valid_percentages["percentage"].sum()

    selected_samples = {}
    for _, row in percentage_per_cluster.iterrows():
        cluster_id = row.name
        cluster_scenario_ids = clusters_df["scenario_id"][clusters_df["cluster"] == cluster_id].tolist()

        num_to_drop = _compute_proportional_number_to_drop(
            num_scenarios_to_drop, row.percentage, min_percentage_per_class, total_valid_percentage
        )

        random.seed(config.seed)
        random.shuffle(cluster_scenario_ids)
        if num_to_drop > 0:
            selected_samples[cluster_id] = _make_group_result(
                keep=cluster_scenario_ids[num_to_drop:],
                drop=cluster_scenario_ids[:num_to_drop],
            )
        else:
            selected_samples[cluster_id] = _make_group_result(keep=cluster_scenario_ids, drop=[])

    _aggregate_selected_samples(selected_samples)
    return selected_samples


def cosine_selection_per_cluster(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """A sample selection strategy that clusters scenario_dec embeddings using K-Means and drops samples based on
    cosine similarity to the cluster centroid, mirroring the logic of alignment_based_selection_per_token.

    Samples with high cosine similarity to their cluster centroid (most typical/redundant) are prioritized for
    dropping. Supports both simple (deterministic) and Gumbel-weighted (stochastic) sorting strategies.

    Args:
        config: encapsulates model analysis configuration parameters. Requires:
            num_clusters (int), percentage_to_keep (float), min_percentage_per_class (float), seed (int),
            sorting_strategy (str, "simple" or "gumbel").
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        selected_samples: A dictionary containing the IDs of the samples to keep or drop.
    """
    scenario_ids_list, embeddings = get_scenario_dec_embeddings(model_outputs)
    scenario_ids_arr = np.array(scenario_ids_list)
    num_scenarios = len(scenario_ids_list)

    kmeans, cluster_labels = _cluster_embeddings(config, embeddings)

    centroids = kmeans.cluster_centers_
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    cluster_percentages = {
        int(c): count / num_scenarios for c, count in zip(unique_clusters, cluster_counts, strict=True)
    }
    min_percentage_per_class = config.min_percentage_per_class
    valid_percentages = {k: v for k, v in cluster_percentages.items() if v > min_percentage_per_class}
    total_valid_percentage = sum(valid_percentages.values())

    num_scenarios_to_drop = int((1 - config.percentage_to_keep) * num_scenarios)

    selected_samples = {}
    for cluster_id in unique_clusters.tolist():
        cluster_mask = cluster_labels == cluster_id
        cluster_scenario_ids = scenario_ids_arr[cluster_mask]
        cluster_embeddings = embeddings[cluster_mask]
        cluster_percentage = cluster_percentages[cluster_id]

        num_to_drop = _compute_proportional_number_to_drop(
            num_scenarios_to_drop, cluster_percentage, min_percentage_per_class, total_valid_percentage
        )

        if num_to_drop > 0:
            scores = compute_cosine_similarity(cluster_embeddings, centroids[cluster_id])
            sorted_ids, _ = _sort_ids_by_score(cluster_scenario_ids, scores, config.sorting_strategy, config.seed)
            selected_samples[cluster_id] = _make_group_result(
                keep=sorted_ids[num_to_drop:].tolist(),
                drop=sorted_ids[:num_to_drop].tolist(),
            )
        else:
            selected_samples[cluster_id] = _make_group_result(keep=cluster_scenario_ids.tolist(), drop=[])

    _aggregate_selected_samples(selected_samples)
    return selected_samples


def _get_agent_count(model_output: output.ModelOutput) -> int:
    """Returns the number of agent slots for a scenario, used as the Den-TP density proxy."""
    agent_ids = np.array(model_output.agent_ids.value)
    return int(np.count_nonzero(agent_ids != INVALID_AGENT_ID))


def _greedy_submodular_select(
    scenario_ids: NDArray[np.str_],
    embeddings: NDArray[np.float64],
    num_to_keep: int,
) -> tuple[list[str], list[str]]:
    """Greedy submodular selection using Den-TP's P(S_j) objective.

    Iteratively selects samples that minimise:
        P(S_j) = Σ_{i ∈ C_k} cos(emb_i, emb_j)  -  Σ_{i ∈ D_k/C_k} cos(emb_i, emb_j)
                        selected similarity     -      unselected similarity

    Minimising P(S_j) simultaneously rewards diversity w.r.t. already-selected samples (low similarity to C_k) and
    coverage of the unselected pool (high similarity to D_k/C_k).

    NOTE: This implementation uses the model embeddings as a proxy for the gradient features described in the paper,
    since gradient computation requires live model access.

    Args:
        scenario_ids: array of scenario IDs.
        embeddings: array of shape (N, d) — one embedding per scenario.
        num_to_keep: number of samples to keep.

    Returns:
        (keep, drop): lists of scenario IDs.
    """
    num_scenarios = len(scenario_ids)
    if num_to_keep >= num_scenarios:
        return scenario_ids.tolist(), []
    if num_to_keep <= 0:
        return [], scenario_ids.tolist()

    # Precompute full pairwise cosine similarity matrix (N x N)
    sim_matrix = compute_pairwise_cosine_similarity(embeddings)

    selected_mask = np.zeros(num_scenarios, dtype=bool)

    # selected_sim[j] = Σ_{i ∈ C_k} cos(emb_i, emb_j) — starts at zero (C_k is empty)
    selected_sim = np.zeros(num_scenarios, dtype=np.float64)

    # unselected_sim[j] = Σ_{i ∈ D_k\C_k} cos(emb_i, emb_j) — initially all are unselected
    unselected_sim = sim_matrix.sum(axis=0).copy()

    for _ in range(num_to_keep):
        # P(S_j) = selected_sim[j] - unselected_sim[j]. We want to select the sample with the lowest P(S_j) to maximize
        # coverage and diversity.
        p_scores = selected_sim - unselected_sim

        # Exclude already selected samples by setting their P(S_j) to infinity so they won't be selected again.
        p_scores[selected_mask] = np.inf

        # Select the sample with the lowest P(S_j) score.
        best_j = int(np.argmin(p_scores))
        selected_mask[best_j] = True

        # Incremental update: best_j leaves D_k\C_k and joins C_k
        selected_sim += sim_matrix[best_j]
        unselected_sim -= sim_matrix[best_j]

    return scenario_ids[selected_mask].tolist(), scenario_ids[~selected_mask].tolist()


def dentp_selection(config: DictConfig, model_outputs: dict[str, output.ModelOutput]) -> dict[str, Any]:
    """Sample selection implementing the Den-TP algorithm: https://arxiv.org/pdf/2409.17385.

    Partitions scenarios into density bins by agent count and applies greedy submodular selection within each bin using
    decoder embeddings as feature representations. High-density partitions are processed first and given priority via a
    dynamic budget allocation rule: n_k = min(|D_k|, floor(B / k)) where B is the remaining budget and k is the
    partition index (high → low density).

    Args:
        config: encapsulates model analysis configuration parameters. Requires: percentage_to_keep (float),
            density_interval (int, default 4), seed (int).
        model_outputs: a dictionary containing model outputs per scenario.

    Returns:
        selected_samples: a dictionary containing the IDs of the samples to keep or drop.
    """
    scenario_ids, embeddings = get_scenario_dec_embeddings(model_outputs)
    num_scenarios = len(scenario_ids)

    agent_counts = np.array([_get_agent_count(model_outputs[sid]) for sid in scenario_ids], dtype=np.int64)

    tau = int(config.get("density_interval", 4))
    rho_min = agent_counts.min().item()

    # Assign each scenario to a 1-indexed density bin
    k_indices = ((agent_counts - rho_min) // tau + 1).astype(int)
    max_k = k_indices.max().item()

    # Build mapping: bin index → array of scenarios in scenario_ids_list
    scenario_partitions: dict[int, NDArray[np.intp]] = {}
    for k in range(1, max_k + 1):
        scenario_idxs = np.where(k_indices == k)[0]
        if len(scenario_idxs) > 0:
            scenario_partitions[k] = scenario_idxs

    keep_budget = int(config.percentage_to_keep * num_scenarios)
    selected_samples: dict[Any, Any] = {}

    # Process partitions in descending order (high density first)
    for k in range(max_k, 0, -1):
        if k not in scenario_partitions:
            continue

        partition_idxs = scenario_partitions[k]
        partition_scenario_ids = scenario_ids[partition_idxs]
        partition_embeddings = embeddings[partition_idxs]
        partition_size = len(partition_idxs)

        # Allocate budget for this partition: n_k = min(|D_k|, floor(B / k)) (DynamicSelect in Algorithm 1)
        num_to_keep = min(partition_size, keep_budget // k)

        keep, drop = _greedy_submodular_select(partition_scenario_ids, partition_embeddings, num_to_keep)
        selected_samples[k] = _make_group_result(keep=keep, drop=drop)
        keep_budget -= len(keep)
    _aggregate_selected_samples(selected_samples)
    return selected_samples


def run_sample_selection(config: DictConfig, model_outputs: dict[str, output.ModelOutput], output_path: Path) -> None:
    """Wrapper function which runs a specified sample selection strategy. A sample selection strategy produces a
    dictionary containing the a set of training scenarios to keep and to drop.

    Args:
        config: encapsulates model analysis configuration parameters.
        model_outputs: a dictionary containing model outputs per scenario.
        output_path: output path where visualization will be saved to.
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
        case SampleSelection.KMEANS_RANDOM_DROP:
            config.clustering_strategy = "kmeans"
            sample_selection = random_selection_per_cluster(config, model_outputs)
        case SampleSelection.SIMPLE_KMEANS_COSINE_DROP:
            config.sorting_strategy = "simple"
            sample_selection = cosine_selection_per_cluster(config, model_outputs)
        case SampleSelection.GUMBEL_KMEANS_COSINE_DROP:
            config.sorting_strategy = "gumbel"
            sample_selection = cosine_selection_per_cluster(config, model_outputs)
        case SampleSelection.DEN_TP:
            sample_selection = dentp_selection(config, model_outputs)
        case _:
            error_message = f"Unsupported selection strategy: {selection_strategy}"
            raise ValueError(error_message)

    output_filepath = output_path / f"sample_selection_{selection_strategy.value}_{config.percentage_to_keep}.json"
    with output_filepath.open("w") as f:
        json.dump(sample_selection, f, indent=2)
