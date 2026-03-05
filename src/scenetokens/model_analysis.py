"""Scenario Analysis Script.

Example usage:

    uv run -m scenetokens.model_analysis analysis=tokenization num_batches=[x] num_scenarios=[y]

See `docs/ANALYSIS.md` and 'configs/analysis.yaml' for more argument details.
"""

import copy
import random
from pathlib import Path
from time import time

import hydra
import pyrootutils
from omegaconf import DictConfig

from scenetokens import utils


_LOGGER = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _run_experiment_analysis(config: DictConfig, output_path: Path) -> None:
    """Runs analyses over the tokenized scenarios for a given experiment.

    Args:
        config (DictConfig): The configuration object containing all necessary parameters for running the analyses.
        output_path (Path): The path where the analysis results should be saved.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Only load the batches if any of the analyses are set to true
    if not any(
        [
            config.run_scenario_class_distribution,
            config.run_token_consistency_analysis,
            config.run_group_uniqueness_analysis,
            config.run_intergroup_analysis,
            config.run_dim_reduction_analysis,
            config.run_score_analysis,
        ]
    ):
        _LOGGER.info("No analyses selected to run. Skipping batch loading and analysis.")
        return

    # Loads scenario tokenized information
    _LOGGER.info("Loading batches from %s", config.paths.batch_cache_path)
    batches = utils.load_batches(
        config.paths.batch_cache_path, config.num_batches, config.num_scenarios, config.seed, config.split
    )

    # Run analyses

    # Produces a histogram over the tokenized scenarios, representing token-utilization.
    if config.run_scenario_class_distribution:
        _LOGGER.info("Running tokenization distribution analysis...")
        utils.plot_scenario_class_distribution(config, batches, output_path)

    # Produces a consistency analysis over tokenized groups.
    if config.run_token_consistency_analysis:
        _LOGGER.info("Running tokenization consistency distribution analysis...")
        utils.compute_token_consistency_matrix(config, batches, output_path)

    # Producess a distribution analysis over tokenized groups.
    if config.run_group_uniqueness_analysis:
        _LOGGER.info("Running group uniqueness distribution analysis...")
        utils.compute_group_uniqueness(config, batches, output_path)

    if config.run_intergroup_analysis:
        _LOGGER.info("Running intergroup distribution analysis...")
        utils.compute_intergroup_uniqueness(config, batches, output_path)

    # Produces a visualization of the scenario embeddings using dimensionality reduction methods.
    if config.run_dim_reduction_analysis:
        _LOGGER.info("Running dimensionality reduction analysis...")
        dim_reduction_result = utils.compute_dimensionality_reduction(config, batches, output_path)
        utils.plot_manifold_by_tokens(config, dim_reduction_result, batches, output_path)

    # This analysis produces a per-class score analysis over the scenarios.
    if config.run_score_analysis:
        _LOGGER.info("Running score analysis...")
        score_analysis = utils.compute_score_analysis(config, batches, output_path)
        if config.viz_scored_scenarios:
            utils.plot_tokenized_scenarios_by_score_percentile(config, batches, score_analysis, output_path)


@hydra.main(version_base="1.3", config_path="configs", config_name="analysis.yaml")
def main(config: DictConfig) -> float | None:
    """Hydra's entrypoint for running scenario analysis training."""
    analysis_paths = {}

    # Run each analysis for each experiment specified in the config, and save the results to the output path.
    for experiment_dir in config.experiment_dirs:
        random.seed(config.seed)
        _LOGGER.info("Running analysis for experiment: %s", experiment_dir)
        start = time()

        # Print the configuration, just to validate that the correct experiment_dir is being used for the analysis.
        experiment_config = copy.deepcopy(config)
        experiment_config.paths.experiment_dir = experiment_dir
        utils.print_config_tree(
            experiment_config,
            resolve=True,
            save_to_file=False,
            print_order=["analysis", "paths"],
            print_field_not_in_print_order=False,
        )

        # Run the tokenization analysis for the given experiment.
        output_path = Path(experiment_config.paths.analysis_path) / f"{experiment_config.split}_model-analysis"
        _run_experiment_analysis(experiment_config, output_path)
        analysis_paths[experiment_dir] = output_path

        _LOGGER.info("Total time: %s second", time() - start)
        _LOGGER.info("Process completed!")

    # If more than one experiment is being analyzed, we also run a comparative analysis across the experiments.
    if len(analysis_paths) > 1:
        _LOGGER.info("Running comparative analysis across experiments...")
        analysis_path = Path(config.output_path) / "tokenization" / "comparative-analysis"
        utils.run_comparative_analysis(analysis_paths, analysis_path)


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
