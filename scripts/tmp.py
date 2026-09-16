from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lerobot.datasets import LeRobotDataset

from data import CAMERAS

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_episodes(data_root: Path, tasks: list[str]) -> tuple[LeRobotDataset, pd.DataFrame]:
    frames = LeRobotDataset("local/data", root=data_root)
    episode_table = frames.meta.episodes
    assert episode_table is not None
    episodes = cast(pd.DataFrame, episode_table.to_pandas())

    # keep episodes whose tasks are all in `tasks`
    keep = episodes["tasks"].map(lambda episode_tasks: set(episode_tasks) <= set(tasks))
    return frames, episodes[keep].reset_index(drop=True)


def sample_episodes(
    episodes: pd.DataFrame, success: bool, n_episodes: int, rng: np.random.Generator
) -> pd.DataFrame:
    candidates = episodes[episodes["success"] == success]
    rows = rng.choice(len(candidates), size=n_episodes, replace=False)
    return candidates.iloc[rows]


def load_early_frame(
    frames: LeRobotDataset, episode: pd.Series, horizon: float, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    # random frame t in the first `horizon` fraction of the episode (at least frame 0)
    n_early_frames = max(1, int(horizon * episode["length"]))
    t = int(rng.integers(n_early_frames))
    frame = cast(dict[str, torch.Tensor], frames[int(episode["dataset_from_index"]) + t])

    # cameras side by side: (H, n_cameras * W, 3)
    images = [frame[camera].permute(1, 2, 0).numpy() for camera in CAMERAS]
    return np.concatenate(images, axis=1), t


def load_samples(
    frames: LeRobotDataset,
    episodes: pd.DataFrame,
    horizon: float,
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    samples = []
    for _, episode in episodes.iterrows():
        image, t = load_early_frame(frames, episode, horizon, rng)
        title = f"episode {episode['episode_index']}, t = {t} / {episode['length']}"
        samples.append((title, image))
    return samples


def plot_grid(
    success_samples: list[tuple[str, np.ndarray]],
    failure_samples: list[tuple[str, np.ndarray]],
    path: Path,
) -> None:
    # one row per pair: success on the left, failure on the right
    n_rows = len(success_samples)
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 3.6 * n_rows), squeeze=False)
    for row in range(n_rows):
        for col, (label, samples) in enumerate(
            [("SUCCESS", success_samples), ("FAILURE", failure_samples)]
        ):
            title, image = samples[row]
            ax = axes[row, col]
            ax.imshow(image.clip(0, 1))
            ax.set_title(f"{label}  {title}", fontsize=11)
            ax.axis("off")

    # camera order is the same in every image
    fig.suptitle(" | ".join(CAMERAS), fontsize=13, y=1, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))  # 0.4 in for the suptitle
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print(f"Saved to {path}")


def main(
    data_root: Path,
    tasks: list[str],
    n_episodes: int,
    horizon: float,
    seed: int,
    output_path: Path,
) -> None:
    rng = np.random.default_rng(seed)
    frames, episodes = load_episodes(data_root, tasks)

    # sample episodes of each outcome and one early frame from each
    success_episodes = sample_episodes(episodes, True, n_episodes, rng)
    failure_episodes = sample_episodes(episodes, False, n_episodes, rng)
    success_samples = load_samples(frames, success_episodes, horizon, rng)
    failure_samples = load_samples(frames, failure_episodes, horizon, rng)

    plot_grid(success_samples, failure_samples, output_path)


if __name__ == "__main__":
    DATA_ROOT = REPO_ROOT / "data" / "actuator_unboxing_recap_mix_v1_success"
    TASKS = ["Take an actuator from the box and place it in the tray"]
    N_EPISODES = 16  # per outcome
    HORIZON = 0.01  # sample within the first 1% of each episode
    SEED = 0
    OUTPUT_PATH = REPO_ROOT / "outputs" / "early_frames_success_vs_failure_task2.png"

    main(
        data_root=DATA_ROOT,
        tasks=TASKS,
        n_episodes=N_EPISODES,
        horizon=HORIZON,
        seed=SEED,
        output_path=OUTPUT_PATH,
    )
