"""Models Metrics Analysis Script.

Example usage:

    uv run -m scenetokens.model_metric_analysis group_name=[name]

See `docs/ANALYSIS.md` and 'configs/model_metric_analysis.yaml' for more argument details.
"""

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

    if config.run_ego_safeshift_analysis:
        config.benchmark = config.ego_safeshift_benchmark
        config.benchmark_filepath = config.ego_safeshift_filepath
        config.benchmark_colormap = config.ego_safeshift_colormap
        config.benchmark_splits_to_compare = config.ego_safeshift_splits_to_compare
        utils.run_benchmark_analysis(config, log, output_path)

    if config.run_causal_benchmark_analysis:
        config.benchmark = config.causal_benchmark
        config.benchmark_filepath = config.causal_benchmark_filepath
        config.benchmark_colormap = config.causal_benchmark_colormap
        config.benchmark_splits_to_compare = config.causal_benchmark_splits_to_compare
        utils.run_benchmark_analysis(config, log, output_path)

    log.info("Total time: %s second", time() - start)
    log.info("Process completed!")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
