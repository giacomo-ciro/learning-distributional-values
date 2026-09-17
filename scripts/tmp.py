import hashlib
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from data import CAMERAS, ValueDataModule

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_train_split(config_path: Path) -> ValueDataModule:
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    datamodule = ValueDataModule(cfg)
    datamodule.setup()
    return datamodule


def cache_path(cache_dir: Path, image: torch.Tensor) -> Path:
    # same content-hash key as BackbonePlusHead._encode_cached
    key = hashlib.blake2b(image.numpy().tobytes(), digest_size=16).hexdigest()
    return cache_dir / key[:2] / f"{key}.npy"


def build_pool(
    datamodule: ValueDataModule,
    cache_dir: Path,
    pool_size: int,
    num_workers: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Decodes pool_size random train frames and keeps the camera images with a cached embedding.

    The cache is keyed by frame content, so the frames must be decoded to find their embeddings.
    Returns the (M, D) embeddings and the (dataset index, camera id) of each.
    """
    # random train frames, loaded without the value labels
    train_indices = datamodule.datasets["train"].indices
    frame_indices = rng.choice(
        train_indices, size=min(pool_size, len(train_indices)), replace=False
    )
    frames = datamodule.datasets["train"].frames
    loader = DataLoader(
        Subset(frames, frame_indices.tolist()),
        batch_size=32,
        num_workers=num_workers,
        collate_fn=lambda batch: batch,
    )

    # keep only the camera images that were embedded during training
    embeddings = []
    keys = []
    for batch in loader:
        for frame in batch:
            for camera_id, camera in enumerate(CAMERAS):
                path = cache_path(cache_dir, frame[camera])
                if not path.exists():
                    continue
                embeddings.append(np.load(path))
                keys.append((int(frame["index"]), camera_id))

    print(f"{len(keys)} cached images out of {3 * len(frame_indices)} pool images")
    return torch.from_numpy(np.stack(embeddings)).float(), keys


def find_neighbors(
    embeddings: torch.Tensor,
    episodes: np.ndarray,
    n_queries: int,
    k: int,
    exclude_same_episode: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine k-NN of n_queries random pool images. Returns queries (Q,) and neighbors (Q, K)."""
    queries = rng.choice(len(embeddings), size=n_queries, replace=False)

    # cosine similarity of each query against the whole pool
    normed = F.normalize(embeddings.cuda(), dim=-1)
    similarity = normed[queries] @ normed.T

    # never match the query itself, optionally nothing from its episode
    similarity[torch.arange(n_queries), torch.from_numpy(queries)] = -torch.inf
    if exclude_same_episode:
        same_episode = torch.from_numpy(episodes[queries][:, None] == episodes[None])
        similarity[same_episode.cuda()] = -torch.inf

    neighbors = similarity.topk(k, dim=-1).indices
    return queries, neighbors.cpu().numpy()


def load_image(datamodule: ValueDataModule, key: tuple[int, int]) -> np.ndarray:
    index, camera_id = key
    frame = cast(dict[str, torch.Tensor], datamodule.datasets["train"].frames[index])
    return frame[CAMERAS[camera_id]].permute(1, 2, 0).numpy().clip(0, 1)


def plot_grid(
    datamodule: ValueDataModule,
    keys: list[tuple[int, int]],
    queries: np.ndarray,
    neighbors: np.ndarray,
    path: Path,
) -> None:
    # one row per query: the query in column 0, its neighbors by decreasing similarity after it
    n_rows, k = neighbors.shape
    fig, axes = plt.subplots(
        n_rows, k + 1, figsize=(2.2 * (k + 1), 2.2 * n_rows), squeeze=False
    )
    for row, query in enumerate(queries):
        ax = axes[row, 0]
        ax.imshow(load_image(datamodule, keys[query]))
        ax.axis("off")

        for col in range(k):
            neighbor = neighbors[row, col]
            ax = axes[row, col + 1]
            ax.imshow(load_image(datamodule, keys[neighbor]))
            ax.axis("off")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print(f"Saved to {path}")


def main(
    config_path: Path,
    cache_dir: Path,
    pool_size: int,
    n_queries: int,
    k: int,
    rows_per_figure: int,
    exclude_same_episode: bool,
    num_workers: int,
    seed: int,
    output_path: Path,
) -> None:
    rng = np.random.default_rng(seed)
    datamodule = load_train_split(config_path)

    # embeddings of the cached images among a random subset of train frames
    embeddings, keys = build_pool(datamodule, cache_dir, pool_size, num_workers, rng)
    episodes = np.array([datamodule.episode[index] for index, _ in keys])

    queries, neighbors = find_neighbors(
        embeddings, episodes, n_queries, k, exclude_same_episode, rng
    )

    # at most rows_per_figure queries per figure, saved as <stem>_<i>.png
    for i, start in enumerate(range(0, n_queries, rows_per_figure)):
        rows = slice(start, start + rows_per_figure)
        path = output_path.with_name(f"{output_path.stem}_{i}{output_path.suffix}")
        plot_grid(datamodule, keys, queries[rows], neighbors[rows], path)


if __name__ == "__main__":
    CONFIG_PATH = REPO_ROOT / "configs" / "train.yaml"  # only data.* is used
    CACHE_DIR = REPO_ROOT / "cache" / "google" / "siglip-so400m-patch14-384"
    POOL_SIZE = 30_000  # train frames decoded to search; the cache is keyed by content, huge value = whole train set (slow)
    N_QUERIES = 6
    K = 5  # columns: the query + K neighbors
    ROWS_PER_FIGURE = 3  # queries beyond this spill into further figures
    EXCLUDE_SAME_EPISODE = False  # True: neighbors come from other episodes only
    NUM_WORKERS = 24
    SEED = 0
    OUTPUT_PATH = (
        REPO_ROOT
        / "outputs"
        / f"siglip_knn{'_other_episodes' if EXCLUDE_SAME_EPISODE else ''}.png"
    )

    main(
        config_path=CONFIG_PATH,
        cache_dir=CACHE_DIR,
        pool_size=POOL_SIZE,
        n_queries=N_QUERIES,
        k=K,
        rows_per_figure=ROWS_PER_FIGURE,
        exclude_same_episode=EXCLUDE_SAME_EPISODE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        output_path=OUTPUT_PATH,
    )
