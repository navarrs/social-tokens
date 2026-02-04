"""Training script for SocialTokens.

Example usage:

    uv run -m scenetokens.train model=[model_name]

See `docs/MODEL_TRAINING.md` and `configs/train.yaml` for more argument details.
"""

import copy
from typing import TYPE_CHECKING

import hydra
import pyrootutils
import pytorch_lightning
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, Dataset

from scenetokens import utils
from scenetokens.utils.constants import DataSplits


if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.loggers.logger import Logger

log = utils.get_pylogger(__name__)

torch.set_float32_matmul_precision("medium")
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

utils.disable_mlflow_tls_verification()


@utils.task_wrapper
def train(cfg: DictConfig) -> tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during training. This
    method is wrapped in optional @task_wrapper decorator, that controls the behavior during failure. Useful for
    multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    metric_dict, object_dict = {}, {}
    print("Check CUDA", torch.cuda.is_available())

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pytorch_lightning.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating loggers...")
    logger: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Instantiating model <%s>", cfg.model._target_)
    model_config = copy.deepcopy(cfg.model)
    model: LightningModule = hydra.utils.get_class(model_config._target_)(config=model_config)

    log.info("Instantiating training dataset <%s>", cfg.dataset._target_)
    train_dataset_config = copy.deepcopy(cfg.dataset)
    train_dataset_config.split = DataSplits.TRAINING
    train_dataset: Dataset = hydra.utils.get_class(train_dataset_config._target_)(config=train_dataset_config)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.model.train_batch_size,
        num_workers=cfg.model.load_num_workers,
        drop_last=False,
        collate_fn=train_dataset.collate_fn,  # pyright: ignore[reportAttributeAccessIssue]
    )

    log.info("Instantiating validation dataset <%s>", cfg.dataset._target_)
    val_dataset_config = copy.deepcopy(cfg.dataset)
    val_dataset_config.split = DataSplits.VALIDATION
    val_dataset: Dataset = hydra.utils.get_class(val_dataset_config._target_)(config=val_dataset_config)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.model.eval_batch_size,
        num_workers=cfg.model.load_num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=val_dataset.collate_fn,  # pyright: ignore[reportAttributeAccessIssue]
    )

    object_dict = {
        "cfg": cfg,
        "training_dataset": train_dataloader,
        "validation_dataset": val_dataloader,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # if logger:
    #     log.info("Logging hyperparameters!")
    #     utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)  # pyright: ignore[reportAssignmentType]

    log.info("Starting training!")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.get("ckpt_path"),
    )
    train_metrics = trainer.callback_metrics
    log.info("Done training!")

    if cfg.get("test"):
        log.info("Instantiating testing dataset <%s>", cfg.dataset._target_)
        test_dataset_config = copy.deepcopy(cfg.dataset)
        test_dataset_config.split = DataSplits.TESTING
        test_dataset: Dataset = hydra.utils.get_class(test_dataset_config._target_)(config=test_dataset_config)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.model.eval_batch_size,
            num_workers=cfg.model.load_num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=test_dataset.collate_fn,  # pyright: ignore[reportAttributeAccessIssue]
        )

        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        log.info("Best ckpt path: %s", ckpt_path)
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, dataloaders=test_loader, ckpt_path=ckpt_path)
        log.info("Done testing!")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Hydra's entrypoint for running model training."""
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))
    log.info("Retrieved metrics: \n%s", metric_value)
    log.info("Process completed!")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
