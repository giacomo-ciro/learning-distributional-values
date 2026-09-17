from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import lightning as pl
from aim.pytorch_lightning import AimLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from omegaconf import DictConfig, OmegaConf


class StepProgressBar(TQDMProgressBar):
    """Single training bar counting global steps up to max_steps, instead of batches per epoch."""

    def on_train_start(self, trainer: pl.Trainer, *_: Any) -> None:
        super().on_train_start()
        bar = self.train_progress_bar
        bar.reset(total=trainer.max_steps)
        bar.initial = bar.n = trainer.global_step  # nonzero when resuming
        bar.set_description("Step")

    def on_train_epoch_start(self, *_: Any) -> None:
        pass  # don't reset the bar every epoch

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *_: Any
    ) -> None:
        bar = self.train_progress_bar
        n = trainer.global_step
        if not bar.disable and self._should_update(n, bar.total):
            bar.n = n
            bar.refresh()
            bar.set_postfix(self.get_metrics(trainer, pl_module))


class ConfigCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that writes config.yaml next to the checkpoints, read back by BaseModel.load_from_checkpoint."""

    def __init__(self, cfg: DictConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg

    def on_save_checkpoint(self, *_: Any) -> None:
        # called right before a checkpoint is written, so the config never exists without one
        assert self.dirpath is not None
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, Path(self.dirpath) / "config.yaml", resolve=True)


def load_trainer(cfg: DictConfig) -> pl.Trainer:
    logger = AimLogger(run_name=cfg.run_name)
    params = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    logger.log_hyperparams(params)
    callbacks: list[pl.Callback] = [
        StepProgressBar(),
        LearningRateMonitor(logging_interval="step"),
    ]
    if cfg.trainer.enable_checkpointing:
        # a single best.ckpt, overwritten whenever val/loss improves
        checkpoint = ConfigCheckpoint(
            cfg,
            dirpath=Path("checkpoints") / cfg.run_name,
            # filename="last",
            # monitor=None,
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            enable_version_counter=False,
        )
        callbacks.append(checkpoint)
    return pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        max_steps=cfg.trainer.max_steps,
        # val_check_interval counts batches, not optimizer steps: scale so validation stays every n steps
        val_check_interval=cfg.trainer.val_every_n_steps
        * cfg.trainer.accumulate_grad_batches,
        # count val_check_interval across epochs, not per epoch (needed when epochs are shorter)
        check_val_every_n_epoch=None,
        overfit_batches=cfg.trainer.overfit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        accelerator="auto",
        devices="auto",
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        num_sanity_val_steps=0,
    )
