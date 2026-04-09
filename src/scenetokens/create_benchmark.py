r"""Script used for creating benchmark dataset splits.

Example usage:

    # Causal Agents benchmark
    uv run -m scenetokens.create_benchmark benchmark=causal_agents \\
        input_data_path=/data/driving/waymo/processed/mini_causal \\
        output_data_path=/data/driving/waymo/processed/causal_agents \\
        causal_labels_path=/data/driving/waymo/causal_agents/processed_labels \\
        strategy=remove_causal

    # Ego-SafeShift benchmark
    uv run -m scenetokens.create_benchmark benchmark=ego_safeshift \\
        scenario_score_mapping_filepath=meta/scenario_to_scores_mapping.csv

    # Environments benchmark
    uv run -m scenetokens.create_benchmark benchmark=environments

See `configs/create_benchmark.yaml` and the per-benchmark configs under `configs/benchmark/` for all options.
"""

from pathlib import Path

import hydra
import pyrootutils
from omegaconf import DictConfig

from scenetokens import benchmarks, utils
from scenetokens.benchmarks import Benchmark


_LOGGER = utils.get_pylogger(__name__)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="create_benchmark.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for creating benchmark dataset splits."""
    _LOGGER.info("Printing cfg tree with Rich! <cfg.extras.print_cfg=True>")
    utils.print_config_tree(cfg, resolve=True, save_to_file=False)

    benchmark = Benchmark(cfg.benchmark_name)
    match benchmark:
        case Benchmark.CAUSAL_AGENTS:
            benchmarks.create_causal_agents(
                causal_data_path=Path(cfg.input_data_path),
                output_data_path=Path(cfg.output_data_path),
                causal_labels_path=Path(cfg.causal_labels_path),
                strategy=cfg.strategy,
                num_workers=cfg.num_workers,
                seed=cfg.seed,
            )
        case Benchmark.EGO_SAFESHIFT:
            benchmarks.create_ego_safeshift(
                causal_data_path=Path(cfg.input_data_path),
                output_data_path=Path(cfg.output_data_path),
                scenario_score_mapping_filepath=Path(cfg.scenario_score_mapping_filepath),
                score_type=cfg.score_type,
                cutoff_percentile=cfg.cutoff_percentile,
                validation_percentage=cfg.validation_percentage,
                num_workers=cfg.num_workers,
                seed=cfg.seed,
            )
        case Benchmark.ENVIRONMENTS:
            benchmarks.create_environments(
                input_data_path=Path(cfg.input_data_path),
                output_path=Path(cfg.output_data_path),
                n_clusters=cfg.n_clusters,
                n_examples=cfg.n_examples,
                sample_percentage=cfg.sample_percentage,
                num_scenarios=cfg.num_scenarios,
                num_workers=cfg.num_workers,
                ego_centered=cfg.ego_centered,
                k_polylines=cfg.k_polylines,
                seed=cfg.seed,
                overwrite=cfg.overwrite,
                map_range=cfg.map_range,
                reduction=cfg.reduction,
            )
        case _:
            _LOGGER.error("Unsupported benchmark: %s", cfg.benchmark_name)


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
