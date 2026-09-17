from __future__ import annotations

from pathlib import Path
from typing import cast

import lightning as pl
import numpy as np
import pandas as pd
import torch
from lerobot.datasets import LeRobotDataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

CAMERAS = (
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.top",
)


def compute_values(episodes: pd.DataFrame) -> np.ndarray:
    # per-frame values indexed by absolute dataset index, NaN for frames of other episodes
    lengths = episodes["length"].to_numpy()
    from_idx = episodes["dataset_from_index"].to_numpy()
    # frames of episode e are contiguous and ordered: [from_idx[e], from_idx[e] + lengths[e])
    assert (episodes["dataset_to_index"].to_numpy() - from_idx == lengths).all()

    c_fail = lengths.max() + 1

    T = np.repeat(lengths, lengths)
    t = np.arange(T.size) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    success = np.repeat(episodes["success"].to_numpy().astype(np.int64), lengths)
    values = np.full(episodes["dataset_to_index"].max(), np.nan)
    values[np.repeat(from_idx, lengths) + t] = -(T - 1 - t) - c_fail * (1 - success)

    # no clip, rescale by overall min value
    # value_norm = (lengths.max() - 1) + c_fail
    # return (values / value_norm).astype(np.float32)

    # with clip, all bad frames map to -1 exactly
    value_norm = lengths.max() - 1

    return (values / value_norm).astype(np.float32).clip(min=-1)


class ValueDataset(Dataset):
    def __init__(self, frames: LeRobotDataset, indices: np.ndarray, bins: np.ndarray):
        self.frames = frames
        self.indices = indices  # absolute dataset indices
        self.bins = bins  # indexed by absolute dataset index

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[i])
        frame = cast(dict[str, torch.Tensor], self.frames[idx])
        x = torch.stack([frame[camera] for camera in CAMERAS])
        y = torch.tensor(self.bins[idx])
        return x, y


class ValueDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.repo_id = "local/data"
        self.root = cfg.data.root
        # {train,val,test}.txt, one episode index per line
        self.split_dir = Path(cfg.data.split_dir)
        self.n_bins = cfg.data.n_bins
        self.eval_frame_stride = cfg.data.eval_frame_stride
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers
        self.datasets: dict[str, ValueDataset] = {}
        self.split_episodes: dict[str, np.ndarray] = {}  # sorted episode ids per split
        # per-frame arrays, indexed by absolute dataset index
        self.values = np.empty(
            0, dtype=np.float32
        )  # normalised value, NaN for dropped frames
        self.success = np.empty(0, dtype=bool)  # episode outcome
        self.progress = np.empty(0)  # fraction of the episode elapsed, t / T in [0, 1)
        self.task = np.empty(
            0, dtype=object
        )  # episode tasks joined by " + ", "" for dropped frames
        self.episode = np.empty(
            0, dtype=np.int64
        )  # episode index, -1 for dropped frames

    def setup(self, stage: str | None = None) -> None:

        # called before each of validate/fit/test: load once
        if self.datasets:
            return

        frames = LeRobotDataset(self.repo_id, root=self.root)
        episode_table = frames.meta.episodes
        assert episode_table is not None
        episodes = cast(pd.DataFrame, episode_table.to_pandas())

        # nearest value bin; bin k has centre -1 + k / (n_bins - 1); -1 for dropped frames
        self.values = compute_values(episodes)
        bins = np.where(
            np.isnan(self.values), -1, np.rint((self.values + 1) * (self.n_bins - 1))
        ).astype(np.int64)
        lengths = episodes["length"].to_numpy()
        from_idx = episodes["dataset_from_index"].to_numpy()
        frame_idx = np.concatenate(
            [
                np.arange(start, start + length)
                for start, length in zip(from_idx, lengths)
            ]
        )
        self.success = np.zeros(len(self.values), dtype=bool)
        self.success[frame_idx] = np.repeat(
            episodes["success"].to_numpy().astype(bool), lengths
        )
        self.progress = np.full(len(self.values), np.nan)
        self.progress[frame_idx] = np.concatenate(
            [np.arange(length) / length for length in lengths]
        )
        episode_task = episodes["tasks"].map(
            lambda tasks: " + ".join(cast(np.ndarray, tasks))
        )
        self.task = np.full(len(self.values), "", dtype=object)
        self.task[frame_idx] = np.repeat(episode_task.to_numpy(), lengths)
        self.episode = np.full(len(self.values), -1, dtype=np.int64)
        self.episode[frame_idx] = np.repeat(
            episodes["episode_index"].to_numpy(), lengths
        )

        # fixed episode split read from disk: every episode is in exactly one split
        episode_index = episodes["episode_index"].to_numpy()
        split = {
            name: np.loadtxt(self.split_dir / f"{name}.txt", dtype=np.int64, ndmin=1)
            for name in ("train", "val", "test")
        }
        all_split_episodes = np.concatenate(list(split.values()))
        assert np.array_equal(np.sort(all_split_episodes), np.sort(episode_index)), (
            f"splits in {self.split_dir} do not partition the dataset episodes"
        )

        to_idx = episodes["dataset_to_index"].to_numpy()
        for name, split_episodes in split.items():
            # positions of the split episodes in the episode table
            split_episode_ids = np.flatnonzero(np.isin(episode_index, split_episodes))
            self.split_episodes[name] = episode_index[split_episode_ids]
            stride = 1 if name == "train" else self.eval_frame_stride
            indices = np.concatenate(
                [np.arange(from_idx[e], to_idx[e], stride) for e in split_episode_ids]
            )
            self.datasets[name] = ValueDataset(frames, indices, bins)

    def _dataloader(self, name: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.datasets[name],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader("test", shuffle=False)


def load_datamodule(cfg: DictConfig) -> ValueDataModule:
    return ValueDataModule(cfg)
