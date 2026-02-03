"""Models Metrics Analysis Script.

Example usage:

    uv run -m scenetokens.model_metric_analysis group_name=[name]

See `docs/ANALYSIS.md` and 'configs/model_metric_analysis.yaml' for more argument details.
"""

import copy
import random
from pathlib import Path
from time import time

import hydra
import pyrootutils
from omegaconf import DictConfig

from scenetokens import utils


log = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="analysis.yaml")
def main(config: DictConfig) -> float | None:
    """Hydra's entrypoint for running scenario analysis training."""
    random.seed(config.seed)

    start = time()
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if config.run_sample_selection_sweep_lineplot_analysis:
        utils.plot_sample_selection_sweep_lineplot(config, log, output_path)

    if config.run_sample_selection_sweep_heatmap_analysis:
        utils.plot_sample_selection_sweep_heatmap(config, log, output_path)

    if config.run_ego_safeshift_analysis:
        ego_benchmark_config = copy.deepcopy(config)
        ego_benchmark_config.benchmark = config.ego_safeshift_benchmark
        ego_benchmark_config.benchmark_filepath = config.ego_safeshift_filepath
        ego_benchmark_config.benchmark_colormap = config.ego_safeshift_colormap
        ego_benchmark_config.benchmark_splits_to_compare = config.ego_safeshift_splits_to_compare
        utils.run_benchmark_analysis(ego_benchmark_config, log, output_path)

    if config.run_causal_benchmark_analysis:
        causal_benchmark_config = copy.deepcopy(config)
        causal_benchmark_config.benchmark = config.causal_benchmark
        causal_benchmark_config.benchmark_filepath = config.causal_benchmark_filepath
        causal_benchmark_config.benchmark_colormap = config.causal_benchmark_colormap
        causal_benchmark_config.benchmark_splits_to_compare = config.causal_benchmark_splits_to_compare
        utils.run_benchmark_analysis(causal_benchmark_config, log, output_path)

    log.info("Total time: %s second", time() - start)
    log.info("Process completed!")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
