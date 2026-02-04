"""Script used for performing sample selection for training experiments.

Example usage:

    uv run -m scenetokens.sample_selection \
        ckpt_path=/path/to/scenetokens/checkpoint.ckpt
        model=scenetokens_student
        selection_strategy='token_random_drop'

See `docs/ANALYSIS.md` and `configs/sample_selection.yaml` for more argument details.
"""

import copy
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from scenetokens import utils
from scenetokens.utils.constants import DataSplits


if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.loggers.logger import Logger

log = utils.get_pylogger(__name__)

torch.set_float32_matmul_precision("medium")
root_path = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

utils.disable_mlflow_tls_verification()


@utils.task_wrapper
def evaluate_and_cache_dataset(cfg: DictConfig) -> tuple[dict, dict]:
    """Evaluates and caches the training set for sample selection purposes.

    Args:
        cfg (Dictcfg): configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    if not Path(cfg.ckpt_path).exists():
        error_message = f"Checkpoint path: {cfg.ckpt_path} does not exist!"
        raise ValueError(error_message)

    log.info("Instantiating model <%s>", cfg.model._target_)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {"cfg": cfg, "model": model, "logger": logger, "trainer": trainer}
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Instantiating training dataset <%s>", cfg.dataset._target_)
    train_dataset_config = copy.deepcopy(cfg.dataset)
    train_dataset_config.split = DataSplits.TRAINING
    train_dataset: Dataset = hydra.utils.get_class(train_dataset_config._target_)(config=train_dataset_config)
    test_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model.config.eval_batch_size,
        num_workers=cfg.model.config.load_num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=train_dataset.collate_fn,  # pyright: ignore[reportAttributeAccessIssue]
    )
    log.info("Starting test process to cache training set.")
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=cfg.ckpt_path)
    metric_dict = trainer.callback_metrics
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="sample_selection.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra's entrypoint for running the sample selection experiment."""
    log.info("Printing cfg tree with Rich! <cfg.extras.print_cfg=True>")
    utils.print_config_tree(cfg, resolve=True, save_to_file=False)

    # Run model evaluation to cache the training set embeddings
    if cfg.create_training_batch_cache:
        evaluate_and_cache_dataset(cfg)

    log.info("Loading batches from %s", cfg.paths.batch_cache_path)
    batches = utils.load_batches(cfg.paths.batch_cache_path, cfg.num_batches, cfg.num_scenarios, cfg.seed, cfg.split)

    # Run sample selection/
    output_path = Path(cfg.paths.meta_path)
    output_path.mkdir(parents=True, exist_ok=True)
    log.info("Saving sample selection lists to %s", str(output_path))
    for selection_strategy, percentage_to_keep in product(cfg.selection_strategies, cfg.percentages_to_keep):
        cfg.selection_strategy = selection_strategy
        cfg.percentage_to_keep = percentage_to_keep
        log.info(
            "Running sample selection with strategy: %s, percentage_to_keep: %s", selection_strategy, percentage_to_keep
        )
        utils.run_sample_selection(cfg, batches, output_path)


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
