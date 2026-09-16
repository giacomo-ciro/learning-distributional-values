from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import get_class, instantiate
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchvision.models import ResNet101_Weights, resnet101
from transformers import SiglipVisionModel


class BaseModel(pl.LightningModule):
    """Distributional value function over n_bins value bins spanning [-1, 0].

    Children implement forward: X of shape (B, 3, 3, H, W) -> logits of shape (B, n_bins).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.n_bins = cfg.data.n_bins
        self.lr = cfg.trainer.lr
        self.scheduler_cfg = cfg.scheduler

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str | Path) -> BaseModel:
        cfg = OmegaConf.load(Path(checkpoint_path).parent / "config.yaml")
        assert isinstance(cfg, DictConfig)
        model_cls = cast(type[BaseModel], get_class(cfg.model.name))
        return cast(
            BaseModel,
            super(BaseModel, model_cls).load_from_checkpoint(checkpoint_path, cfg=cfg),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _common_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        # entropy of the batch-averaged prediction: near 0 = same prediction for every sample
        batch_probs = logits.float().softmax(dim=-1).mean(dim=0)
        batch_entropy = -(batch_probs * batch_probs.clamp_min(1e-12).log()).sum()

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/acc", acc)
        self.log(f"{stage}/batch_entropy", batch_entropy)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._common_step(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._common_step(batch, "val")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._common_step(batch, "test")

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        return self(x)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        # frozen parameters (e.g. a pretrained backbone) are excluded
        optimizer = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad), lr=self.lr
        )
        scheduler = instantiate(self.scheduler_cfg)(optimizer)
        return {
            "optimizer": optimizer,
            # stepped per optimizer step; name is the metric logged by LearningRateMonitor
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "train/lr",
            },
        }


class GradientFreeBaseline(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.automatic_optimization = False
        # Lightning only advances global_step on optimizer steps, so without one fit never reaches max_steps
        self.dummy = nn.Parameter(torch.zeros(()))

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = super().training_step(batch, batch_idx)
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.step()  # no-op (no grads), only advances global_step
        return loss


class RandomBaseline(GradientFreeBaseline):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.shape[0], self.n_bins, device=x.device)


class MajorityBaseline(GradientFreeBaseline):
    """Counts the bins of the training labels seen so far and always predicts the most frequent one."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # add-one smoothing: finite loss for unseen bins, uniform prediction before training
        self.register_buffer("counts", torch.ones(self.n_bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.counts).expand(x.shape[0], -1)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        _, y = batch
        self.counts += torch.bincount(y, minlength=self.n_bins)
        return super().training_step(batch, batch_idx)


class ResNet101Scratch(BaseModel):
    """ResNet-101 from scratch on the 3 camera frames stacked channel-wise (early fusion)."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.encoder = resnet101(weights=None, num_classes=self.n_bins)
        # 9 input channels: 3 RGB frames
        self.encoder.conv1 = nn.Conv2d(
            9, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1, 2)  # (B, 3, 3, H, W) -> (B, 9, H, W)
        return self.encoder(x)


class Backbone(nn.Module, ABC):
    """Embeds camera frames: (N, 3, H, W) in [0, 1] -> (N, embed_dim), preprocessing included."""

    name: str  # identifies the embeddings, also names the embedding cache folder
    embed_dim: int

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor: ...


class SigLIP(Backbone):
    name = "google/siglip-so400m-patch14-384"

    def __init__(self):
        super().__init__()
        self.model = SiglipVisionModel.from_pretrained(self.name)
        self.embed_dim = self.model.config.hidden_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # replicates SiglipImageProcessor on [0, 1] float frames: bicubic resize, normalize to [-1, 1]
        size = self.model.config.image_size
        images = F.interpolate(images, size=size, mode="bicubic").clamp(0, 1)
        images = (images - 0.5) / 0.5
        return self.model(pixel_values=images).pooler_output


class ResNet101(Backbone):
    name = "torchvision/resnet101-IMAGENET1K_V2"

    def __init__(self):
        super().__init__()
        self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        self.embed_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # frames are already 224x224, the ImageNet resolution
        return self.model((images - self.mean) / self.std)


class Head(nn.Module, ABC):
    """Maps the concatenated camera embeddings to logits: (B, in_dim) -> (B, out_dim)."""

    def __init__(self, in_dim: int, out_dim: int, cfg: DictConfig):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class Linear(Head):
    def __init__(self, in_dim: int, out_dim: int, cfg: DictConfig):
        super().__init__(in_dim, out_dim, cfg)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FFNN(Head):
    """One hidden layer of width cfg.model.hidden_dim."""

    def __init__(self, in_dim: int, out_dim: int, cfg: DictConfig):
        super().__init__(in_dim, out_dim, cfg)
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.model.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.model.hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BackbonePlusHead(BaseModel):
    """Each camera frame is embedded independently by BACKBONE,
    the 3 embeddings are concatenated and mapped to logits by HEAD.

    Abstract: subclasses pick the combination by setting BACKBONE and HEAD.

    With cfg.model.freeze the backbone is a frozen feature extractor whose embeddings are cached
    on disk, otherwise it is finetuned together with the head.
    """

    BACKBONE: type[Backbone]
    HEAD: type[Head]

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # not `self.freeze`: that would shadow LightningModule.freeze()
        self.freeze_backbone: bool = cfg.model.freeze
        self.backbone = self.BACKBONE()
        self.backbone.requires_grad_(not self.freeze_backbone)
        self.head = self.HEAD(3 * self.backbone.embed_dim, self.n_bins, cfg)
        # float16 embeddings, one file per camera image; delete when changing a backbone's forward
        self.cache_dir = Path("cache") / self.backbone.name

    def train(self, mode: bool = True) -> BackbonePlusHead:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # frozen: always in inference mode, e.g. BatchNorm statistics
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        images = x.flatten(0, 1)  # (B * 3, 3, H, W)
        if self.freeze_backbone:
            emb = self._encode_cached(images)
        else:
            emb = self.backbone(images)
        return self.head(emb.view(x.shape[0], -1))

    @torch.no_grad()
    def _encode_cached(self, images: torch.Tensor) -> torch.Tensor:
        # cache key: content hash of the raw frame, so it is independent of dataset indices and splits
        keys = [
            hashlib.blake2b(image.numpy().tobytes(), digest_size=16).hexdigest()
            for image in images.cpu()
        ]
        paths = [self.cache_dir / key[:2] / f"{key}.npy" for key in keys]
        hits = [i for i, path in enumerate(paths) if path.exists()]
        misses = [i for i, path in enumerate(paths) if not path.exists()]

        emb = torch.empty(len(paths), self.backbone.embed_dim, device=images.device)
        if hits:
            emb[hits] = torch.from_numpy(
                np.stack([np.load(paths[i]) for i in hits])
            ).to(emb)
        if misses:
            new = self.backbone(images[misses])
            emb[misses] = new.to(emb)
            for i, e in zip(misses, new.half().cpu().numpy(), strict=True):
                paths[i].parent.mkdir(parents=True, exist_ok=True)
                tmp = paths[i].with_suffix(".tmp")
                with tmp.open("wb") as f:
                    np.save(
                        f, e
                    )  # .npy has a 128-byte header, torch.save a much larger zip container
                tmp.rename(
                    paths[i]
                )  # atomic: an interrupted write never leaves a corrupt entry
        return emb


class SigLIPLinear(BackbonePlusHead):
    BACKBONE = SigLIP
    HEAD = Linear


class SigLIPFFNN(BackbonePlusHead):
    BACKBONE = SigLIP
    HEAD = FFNN


class ResNet101Linear(BackbonePlusHead):
    BACKBONE = ResNet101
    HEAD = Linear


class ResNet101FFNN(BackbonePlusHead):
    BACKBONE = ResNet101
    HEAD = FFNN


def load_model(cfg: DictConfig) -> BaseModel:
    if cfg.resume_from_ckpt is not None:
        return BaseModel.load_from_checkpoint(
            Path("checkpoints") / cfg.resume_from_ckpt
        )
    return get_class(cfg.model.name)(cfg)
