from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data import ValueDataModule, ValueDataset
from model import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints"
CONFIG_PATH = REPO_ROOT / "configs" / "train.yaml"  # defines the data split
DEVICE = "cuda"


def load_data_config(config_path: Path) -> DictConfig:
    # only the data section and seed are used: the hydra defaults list is not resolved
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    # the data root in the config is relative to the repo root
    cfg.data.root = str(REPO_ROOT / cfg.data.root)
    return cfg


def load_split(cfg: DictConfig, split: str) -> tuple[ValueDataModule, ValueDataset]:
    datamodule = ValueDataModule(cfg)
    datamodule.setup()
    return datamodule, datamodule.datasets[split]


@torch.no_grad()
def predict_logits(
    model: BaseModel, dataset: ValueDataset, batch_size: int, num_workers: int, device: str
) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    logits = []
    with torch.autocast(device, dtype=torch.bfloat16):
        for x, _ in loader:
            logits.append(model(x.to(device)).float().cpu())
    return torch.cat(logits)  # (N, n_bins)


def expected_value(logits: torch.Tensor) -> np.ndarray:
    # bin k has centre -1 + k / (n_bins - 1), see ValueDataModule.setup
    n_bins = logits.shape[-1]
    bin_centres = torch.linspace(-1, 0, n_bins)
    probs = logits.softmax(dim=-1)
    return (probs @ bin_centres).numpy()


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def bin_cross_entropy(logits: torch.Tensor, bins: np.ndarray) -> float:
    return F.cross_entropy(logits, torch.from_numpy(bins)).item()


def success_timestep_correlation(
    pred: np.ndarray, indices: np.ndarray, episode: np.ndarray, success: np.ndarray
) -> float:
    # Pearson correlation computed within each successful episode, then averaged.
    # Pooling episodes would mix different lengths: the target itself is not linear in t across episodes.
    # Frames of an episode are contiguous, so the dataset index is the timestep up to an offset.
    correlations = []
    for episode_id in np.unique(episode[success]):
        in_episode = episode == episode_id
        correlations.append(np.corrcoef(pred[in_episode], indices[in_episode])[0, 1])

    # NaN if any episode has constant predictions (e.g. the majority baseline)
    return float(np.mean(correlations))


def evaluate_run(
    checkpoint_path: Path,
    datamodule: ValueDataModule,
    dataset: ValueDataset,
    cfg: DictConfig,
    device: str,
) -> dict[str, float]:
    model = BaseModel.load_from_checkpoint(checkpoint_path).to(device).eval()

    # Distributional prediction -> expected value over the bin centres.
    logits = predict_logits(model, dataset, cfg.data.batch_size, cfg.data.num_workers, device)
    assert logits.shape[-1] == cfg.data.n_bins, "model trained with a different number of bins"
    pred = expected_value(logits)

    # Per-frame labels of the evaluated frames.
    indices = dataset.indices
    target = datamodule.values[indices]
    bins = dataset.bins[indices]
    episode = datamodule.episode[indices]
    success = datamodule.success[indices]

    return {
        "r2": r2_score(pred, target),
        "cross_entropy": bin_cross_entropy(logits, bins),
        "success_timestep_corr": success_timestep_correlation(pred, indices, episode, success),
    }


def main(run_names: list[str], split: str) -> None:
    # The same split for every run, independent of the config each model was trained with.
    cfg = load_data_config(CONFIG_PATH)
    datamodule, dataset = load_split(cfg, split)

    metrics = {}
    for run_name in run_names:
        checkpoint_path = CHECKPOINTS_ROOT / run_name / "best.ckpt"
        metrics[run_name] = evaluate_run(checkpoint_path, datamodule, dataset, cfg, DEVICE)

    print(f"Metrics on the {split} split:")
    print(pd.DataFrame(metrics).T.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    RUN_NAMES = [
        "random",
        "majority",
        "siglip_linear",
        "resnet101_linear",
        "resnet101_scratch"
    ]
    SPLIT = "test"  # frames subsampled by data.eval_frame_stride

    main(
        run_names=RUN_NAMES,
        split=SPLIT,
    )
