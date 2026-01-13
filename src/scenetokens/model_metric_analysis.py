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

    if config.run_sample_selection_sweep_lineplot_analysis:
        utils.plot_sample_selection_sweep_lineplot(config, log, output_path)

    if config.run_sample_selection_sweep_heatmap_analysis:
        utils.plot_sample_selection_sweep_heatmap(config, log, output_path)

    if config.run_egosafeshift_analysis:
        utils.plot_egosafeshift(config, log, output_path)

    if config.run_causal_benchmark_analysis:
        utils.plot_causal_benchmark(config, log, output_path)

    log.info("Total time: %s second", time() - start)
    log.info("Process completed!")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
