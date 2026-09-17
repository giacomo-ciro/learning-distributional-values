"""Blind pairwise annotation of the advantages predicted by a value function.

The advantage of the 1 s interval starting at frame t is
    A_t = sum of the rewards over [t, t + N) + V(t + N) - V(t),
so it is a score of that interval, and every frame with t + N <= T - 1 has one.

For each test episode the script samples two pairs of intervals: the highest and the
lowest advantage of the episode, and two random intervals as a control. The two clips of
a pair are shown one above the other with the scored interval highlighted, and the only
question asked is which of the two went better. Nothing about the model, the advantages
or the episode outcome is shown, so the answers are independent of what is being scored.
Agreement with the sign of the advantage difference is the metric; chance is 50%.

Only the extreme intervals depend on the model: the episodes, the control intervals, which
clip goes on top and the order of the pairs are fixed by the seed. Answers are keyed by the
two intervals and shared by all runs, so a pair judged for one model is never asked again.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder
from tqdm import tqdm

from data import CAMERAS, ValueDataModule, ValueDataset, value_scale
from model import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
# defines the dataset and the test split
CONFIG_PATH = REPO_ROOT / "configs" / "train.yaml"
# per-frame predictions, one csv per run
VALUES_DIR = REPO_ROOT / "outputs" / "values"
# sampled pairs, one csv per run
PAIRS_DIR = REPO_ROOT / "outputs" / "advantage"
# answers, shared by all runs
ANSWERS_PATH = PAIRS_DIR / "answers.csv"
# tiled clips, safe to delete
CLIP_DIR = REPO_ROOT / "cache" / "annotate"
PAIR_COLUMNS = [
    "pair_id",
    "kind",  # extreme = best vs worst advantage of the episode, control = two random intervals
    "episode_index",
    "top_frame",
    "bottom_frame",
    "top_advantage",
    "bottom_advantage",
    "top_value",
    "top_value_ahead",  # V(t + N), to tell a sustained drop from a single-frame spike
    "bottom_value",
    "bottom_value_ahead",
]
# an answer is about the two clips only, so it is keyed by the intervals and not by the run
ANSWER_KEY = ["episode_index", "top_frame", "bottom_frame"]
ANSWER_COLUMNS = [
    *ANSWER_KEY,
    "choice",  # top, bottom or tie
    "top_error",  # 1 when tagged as clearly going wrong
    "bottom_error",
]


def load_data_config(config_path: Path) -> DictConfig:
    # only the data section is used: the hydra defaults list is not resolved
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    # the data root and split dir in the config are relative to the repo root
    cfg.data.root = str(REPO_ROOT / cfg.data.root)
    cfg.data.split_dir = str(REPO_ROOT / cfg.data.split_dir)
    # the advantage needs a value at every frame, not at every eval_frame_stride-th one
    cfg.data.eval_frame_stride = 1
    return cfg


def load_datamodule(cfg: DictConfig) -> ValueDataModule:
    # the split is computed by the datamodule itself, so it matches training exactly
    datamodule = ValueDataModule(cfg)
    datamodule.setup()
    return datamodule


def load_episodes(datamodule: ValueDataModule) -> pd.DataFrame:
    episode_table = datamodule.datasets["test"].frames.meta.episodes
    assert episode_table is not None
    return cast(pd.DataFrame, episode_table.to_pandas()).set_index("episode_index")


@torch.no_grad()
def predict_values(
    model: BaseModel, dataset: ValueDataset, cfg: DictConfig, device: str
) -> np.ndarray:
    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers
    )

    # distributional prediction -> expected value over the bin centres; only the forward
    # pass runs in bf16, since the advantage compares values a few thousandths apart
    bin_centres = torch.linspace(-1, 0, cfg.data.n_bins)
    values = []
    for x, _ in tqdm(loader, desc="frames", unit="batch"):
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(x.to(device))
        probs = logits.float().cpu().softmax(dim=-1)
        values.append(probs @ bin_centres)
    return torch.cat(values).numpy()


def load_or_predict_values(
    run_name: str,
    datamodule: ValueDataModule,
    cfg: DictConfig,
    device: str,
    csv_path: Path,
) -> pd.DataFrame:
    # inference over the whole test split takes minutes, so it is cached per checkpoint
    if csv_path.exists():
        return pd.read_csv(csv_path)

    checkpoint_path = REPO_ROOT / "checkpoints" / run_name / "best.ckpt"
    model = BaseModel.load_from_checkpoint(checkpoint_path).to(device).eval()
    dataset = datamodule.datasets["test"]
    print(f"predicting {len(dataset)} test frames with {checkpoint_path}")

    # frames of an episode are contiguous and ordered, so the frame number is an offset
    values = pd.DataFrame(
        {
            "episode_index": datamodule.episode[dataset.indices],
            "index": dataset.indices,
            "value": predict_values(model, dataset, cfg, device),
        }
    )
    first_index = values.groupby("episode_index")["index"].transform("min")
    values["frame"] = values["index"] - first_index

    values = values[["episode_index", "frame", "value"]]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(csv_path, index=False, float_format="%.6f")
    return values


def values_per_episode(values: pd.DataFrame) -> dict[int, np.ndarray]:
    return {
        int(episode_index): group.sort_values("frame")["value"].to_numpy()
        for episode_index, group in values.groupby("episode_index")
    }


def advantages(values: np.ndarray, n: int, scale: float) -> np.ndarray:
    # A_t for t in [0, T - 1 - n]: n steps of reward -1 plus the change in value over them
    return values[n:] - values[:-n] - n / scale


def orient(episode_index: int, frames: tuple[int, int], seed: int) -> tuple[int, int]:
    # which interval is shown on top is random, so the position carries no signal, but it is
    # fixed for the two intervals: every run shows the same pair the same way up
    first, second = sorted(frames)
    rng = np.random.default_rng([seed, episode_index, first, second])
    return (first, second) if rng.random() < 0.5 else (second, first)


def pair_row(
    kind: str,
    episode_index: int,
    frames: tuple[int, int],
    values: np.ndarray,
    advantage: np.ndarray,
    n: int,
    seed: int,
    order: float,
) -> dict[str, object]:
    top, bottom = orient(episode_index, frames, seed)
    return {
        "order": order,
        "kind": kind,
        "episode_index": episode_index,
        "top_frame": top,
        "bottom_frame": bottom,
        "top_advantage": advantage[top],
        "bottom_advantage": advantage[bottom],
        "top_value": values[top],
        "top_value_ahead": values[top + n],
        "bottom_value": values[bottom],
        "bottom_value_ahead": values[bottom + n],
    }


def sample_pairs(
    episode_values: dict[int, np.ndarray],
    n: int,
    scale: float,
    n_episodes: int,
    seed: int,
) -> pd.DataFrame:
    episode_indices = np.random.default_rng(seed).permutation(sorted(episode_values))

    rows = []
    for episode_index in episode_indices[:n_episodes]:
        episode_index = int(episode_index)
        values = episode_values[episode_index]
        advantage = advantages(values, n, scale)

        # every random draw below depends on the seed and the episode only, never on the
        # values, so all runs get the same control intervals and the same order
        rng = np.random.default_rng([seed, episode_index])
        extreme_order, control_order = rng.random(2)

        # two random intervals of the same episode: the base rate of a visible difference
        first = int(rng.integers(len(advantage)))
        far_enough = np.flatnonzero(np.abs(np.arange(len(advantage)) - first) >= n)
        second = int(rng.choice(far_enough))
        rows.append(
            pair_row(
                "control",
                episode_index,
                (first, second),
                values,
                advantage,
                n,
                seed,
                control_order,
            )
        )

        # the best and the worst interval of the episode, unless they overlap
        best, worst = int(advantage.argmax()), int(advantage.argmin())
        if abs(best - worst) >= n:
            rows.append(
                pair_row(
                    "extreme",
                    episode_index,
                    (best, worst),
                    values,
                    advantage,
                    n,
                    seed,
                    extreme_order,
                )
            )

    # pairs of both kinds are interleaved, so the kind cannot be guessed from the order; an
    # extreme pair that coincides with the control one is shown once
    pairs = pd.DataFrame(rows).sort_values("order").drop_duplicates(ANSWER_KEY)
    pairs = pairs.reset_index(drop=True)
    pairs["pair_id"] = pairs.index
    return pairs[PAIR_COLUMNS]


def save_csv(table: pd.DataFrame, csv_path: Path) -> None:
    # write to a temporary file first so a crash never leaves a truncated csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(".tmp")
    table.to_csv(tmp_path, index=False, float_format="%.6f")
    tmp_path.replace(csv_path)


def load_answers(csv_path: Path) -> pd.DataFrame:
    # an empty table is typed like a loaded file, so it merges with the pairs the same way
    dtypes = {column: "int64" for column in ANSWER_COLUMNS} | {"choice": "object"}
    if not csv_path.exists():
        return pd.DataFrame(columns=ANSWER_COLUMNS).astype(dtypes)
    return pd.read_csv(csv_path, dtype=dtypes)


def attach_answers(pairs: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    # pairs without an answer get an empty choice and no tags
    pairs = pairs.merge(answers, on=ANSWER_KEY, how="left")
    pairs["choice"] = pairs["choice"].astype("object")
    pairs[["top_error", "bottom_error"]] = (
        pairs[["top_error", "bottom_error"]].fillna(0).astype(int)
    )
    return pairs


def report_agreement(pairs: pd.DataFrame) -> None:
    annotated = pairs[pairs["choice"].notna()].copy()
    print(f"\n{len(annotated)} / {len(pairs)} pairs annotated")
    if annotated.empty:
        return

    # the annotator agrees when the clip they picked is the one with the higher advantage
    decided = annotated[annotated["choice"] != "tie"].copy()
    decided["agrees"] = (decided["choice"] == "top") == (
        decided["top_advantage"] > decided["bottom_advantage"]
    )
    print(f"ties: {1 - len(decided) / len(annotated):.0%}")
    if decided.empty:
        return
    print(
        f"agreement: {decided['agrees'].mean():.0%} of {len(decided)} pairs (chance 50%)"
    )

    print("\nagreement by kind:")
    print(decided.groupby("kind")["agrees"].agg(["mean", "size"]).to_string())

    # the tags are the second question: does the worse interval actually contain a failure
    lower_is_top = annotated["top_advantage"] < annotated["bottom_advantage"]
    lower = np.where(lower_is_top, annotated["top_error"], annotated["bottom_error"])
    higher = np.where(lower_is_top, annotated["bottom_error"], annotated["top_error"])
    print(
        f"\ntagged as going wrong: {lower.mean():.0%} of the lower-advantage clips, "
        f"{higher.mean():.0%} of the higher-advantage ones"
    )


def clip_spans(
    pairs: pd.DataFrame, episodes: pd.DataFrame, n: int, context: int
) -> dict[tuple[int, str], tuple[int, int, int]]:
    # (pair, side) -> the episode and the frame range of the clip; the scored interval is
    # [frame, frame + n] and the clip extends up to context frames beyond it on both sides
    spans = {}
    for pair in pairs.itertuples():
        length = int(episodes.loc[pair.episode_index, "length"])
        frames = {side: int(getattr(pair, f"{side}_frame")) for side in ("top", "bottom")}

        # both clips of a pair get the same context, cut to what the episode has on either
        # side, so they enter and leave the scored interval at the same moment
        before = min(context, *frames.values())
        after = min(context, *(length - 1 - (frame + n) for frame in frames.values()))

        for side, frame in frames.items():
            spans[(int(pair.pair_id), side)] = (
                int(pair.episode_index),
                frame - before,
                frame + n + after,
            )
    return spans


def build_clip(
    episode: pd.Series,
    start: int,
    end: int,
    data_root: Path,
    video_path: str,
    fps: int,
    clip_path: Path,
) -> None:
    # decode frames [start, end] from each camera and tile the cameras side by side
    views = []
    for camera in CAMERAS:
        file_path = data_root / video_path.format(
            video_key=camera,
            chunk_index=int(episode[f"videos/{camera}/chunk_index"]),
            file_index=int(episode[f"videos/{camera}/file_index"]),
        )
        # the range is inset by a quarter frame on both sides, so rounding cannot add or drop one
        from_timestamp = float(episode[f"videos/{camera}/from_timestamp"])
        decoded = VideoDecoder(str(file_path)).get_frames_played_in_range(
            from_timestamp + (start + 0.25) / fps,
            from_timestamp + (end + 0.75) / fps,
        )
        assert len(decoded.data) == end - start + 1, "decoded frames != requested range"
        views.append(decoded.data)
    tiled = torch.cat(views, dim=-1)  # (T, 3, H, n_cameras * W)

    # short GOP keeps frame-by-frame seeking responsive in the browser
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = clip_path.with_suffix(".tmp.mp4")
    VideoEncoder(tiled, frame_rate=fps).to_file(
        tmp_path,
        codec="libx264",
        pixel_format="yuv420p",
        crf=18,
        extra_options={"g": 10},
    )
    tmp_path.replace(clip_path)


def record_answer(answers: pd.DataFrame, answer: dict[str, object]) -> pd.DataFrame:
    # a new answer for the same two intervals replaces the old one
    key = [answer[column] for column in ANSWER_KEY]
    same = (answers[ANSWER_KEY] == key).all(axis=1)
    return pd.concat([answers[~same], pd.DataFrame([answer])], ignore_index=True)


def make_handler(
    pairs: pd.DataFrame,
    spans: dict[tuple[int, str], tuple[int, int, int]],
    episodes: pd.DataFrame,
    answers: pd.DataFrame,
    answers_path: Path,
    data_root: Path,
    video_path: str,
    fps: int,
    n: int,
    clip_dir: Path,
) -> type[BaseHTTPRequestHandler]:
    clip_lock = (
        threading.Lock()
    )  # the page pre-fetches the next pair while the current one loads
    answer_lock = threading.Lock()  # requests are served on concurrent threads

    def clip_state(pair_id: int, side: str) -> dict[str, object]:
        _, start, end = spans[(pair_id, side)]
        frame = int(pairs.loc[pair_id, f"{side}_frame"])
        # everything the page needs to draw the clip, and nothing that reveals the score
        return {
            "length": end - start + 1,
            "highlight": [frame - start, frame - start + n],
        }

    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self.send(PAGE.encode(), "text/html")

            elif self.path == "/state":
                state = {
                    "fps": fps,
                    "pairs": [
                        {
                            "pair_id": int(pair.pair_id),
                            "task": " + ".join(
                                episodes.loc[pair.episode_index, "tasks"]
                            ),
                            "top": clip_state(int(pair.pair_id), "top"),
                            "bottom": clip_state(int(pair.pair_id), "bottom"),
                            "choice": None if pd.isna(pair.choice) else pair.choice,
                            "top_error": int(pair.top_error),
                            "bottom_error": int(pair.bottom_error),
                        }
                        for pair in pairs.itertuples()
                    ],
                }
                self.send(json.dumps(state).encode(), "application/json")

            elif self.path.startswith("/clip/"):
                pair_id, side = self.path.removeprefix("/clip/").split("/")
                episode_index, start, end = spans[(int(pair_id), side)]
                clip_path = clip_dir / f"{episode_index:06d}_{start:06d}_{end:06d}.mp4"
                with clip_lock:
                    if not clip_path.exists():
                        build_clip(
                            episodes.loc[episode_index],
                            start,
                            end,
                            data_root,
                            video_path,
                            fps,
                            clip_path,
                        )
                self.send(clip_path.read_bytes(), "video/mp4")

            else:
                self.send_error(404)

        def do_POST(self) -> None:
            nonlocal answers
            assert self.path == "/annotate"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            pair = pairs.loc[body["pair_id"]]
            answer = {column: int(pair[column]) for column in ANSWER_KEY}
            for column in ("choice", "top_error", "bottom_error"):
                answer[column] = body[column]
                pairs.loc[body["pair_id"], column] = body[column]  # for a page reload
            with answer_lock:
                answers = record_answer(answers, answer)
                save_csv(answers, answers_path)
            self.send(b"{}", "application/json")

        def log_message(self, format: str, *args) -> None:
            pass

    return Handler


def main(
    run_name: str,
    lookahead_seconds: float,
    context_seconds: float,
    n_episodes: int,
    seed: int,
    device: str,
    port: int,
) -> None:
    cfg = load_data_config(CONFIG_PATH)
    datamodule = load_datamodule(cfg)
    episodes = load_episodes(datamodule)
    fps = datamodule.datasets["test"].frames.meta.fps
    n = round(fps * lookahead_seconds)

    # per-frame values of the test split, then the advantage of every 1 s interval
    values = load_or_predict_values(
        run_name, datamodule, cfg, device, VALUES_DIR / f"{run_name}.csv"
    )
    episode_values = values_per_episode(values)

    # two pairs of intervals per episode; the draw is deterministic, so it is only saved for
    # inspection and never read back
    pairs = sample_pairs(episode_values, n, value_scale(episodes), n_episodes, seed)
    save_csv(pairs, PAIRS_DIR / f"{run_name}.csv")

    # answers given for any run cover the pairs of this one with the same two intervals
    answers = load_answers(ANSWERS_PATH)
    pairs = attach_answers(pairs, answers)
    report_agreement(pairs)

    frames = datamodule.datasets["test"].frames
    spans = clip_spans(pairs, episodes, n, round(fps * context_seconds))
    handler = make_handler(
        pairs.set_index("pair_id", drop=False),
        spans,
        episodes,
        answers,
        ANSWERS_PATH,
        frames.root,
        frames.meta.video_path,
        fps,
        n,
        CLIP_DIR / Path(cfg.data.root).name,
    )
    print(f"\nAnnotate at http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Advantage annotation</title>
<style>
  body { margin: 0; padding: 16px; background: #16181c; color: #e6e6e6; font: 14px system-ui, sans-serif; }
  /* both clips must fit the viewport: the width follows the height left after the text
     around the videos (~300px), times the aspect ratio of the tiled clip */
  main { --aspect: 3; max-width: min(1344px, calc((100vh - 300px) / 2 * var(--aspect))); margin: 0 auto; }
  header { display: flex; gap: 16px; align-items: baseline; margin-bottom: 8px; }
  h1 { font-size: 18px; margin: 0; }
  .muted { color: #8a8f98; } .tagged { color: #e5645e; }
  .clip { margin-bottom: 12px; }
  .clip video { width: 100%; background: #000; display: block; image-rendering: pixelated; border: 2px solid transparent; box-sizing: border-box; }
  .clip.scored video { border-color: #e6c15f; }
  .clip.chosen video { border-color: #5fbf77; }
  canvas { width: 100%; height: 12px; display: block; }
  .label { display: flex; gap: 12px; margin: 4px 0; font-variant-numeric: tabular-nums; }
  kbd { background: #2a2d33; border-radius: 3px; padding: 1px 5px; }
  .help { line-height: 2; }
</style>
</head>
<body>
<main>
  <header>
    <h1 id="task"></h1><span id="position" class="muted"></span><span id="saved" class="muted"></span>
  </header>
  <div class="clip" id="clip-top">
    <div class="label"><span>TOP</span><span id="tag-top" class="tagged"></span></div>
    <video id="video-top" muted playsinline loop></video>
    <canvas id="timeline-top" height="12"></canvas>
  </div>
  <div class="clip" id="clip-bottom">
    <div class="label"><span>BOTTOM</span><span id="tag-bottom" class="tagged"></span></div>
    <video id="video-bottom" muted playsinline loop></video>
    <canvas id="timeline-bottom" height="12"></canvas>
  </div>
  <div class="help muted">
    Which clip made more progress <em>during</em> the highlighted interval? Judge the progress made
    over that second, not how far along the task already is: a clip near the end of the task is not
    better for being near the end. Lost time counts as worse, and so does anything that puts the
    outcome at risk, which outweighs mere slowness. &nbsp;
    <kbd>↑</kbd> top &nbsp; <kbd>↓</kbd> bottom &nbsp; <kbd>0</kbd> tie or can't tell &nbsp;
    <kbd>1</kbd>/<kbd>2</kbd> tag top/bottom as clearly going wrong &nbsp;
    <kbd>Space</kbd> play/pause &nbsp; <kbd>R</kbd> replay &nbsp; <kbd>P</kbd> previous pair
  </div>
</main>
<script>
const SIDES = ["top", "bottom"];
let fps, pairs;
let current = 0, tags = { top: 0, bottom: 0 };

const pair = () => pairs[current];
const video = side => document.getElementById(`video-${side}`);
// pairs answered in an earlier session, or for another run, are skipped
const nextUnanswered = after => pairs.findIndex((p, i) => i > after && p.choice === null);

async function init() {
  ({ fps, pairs } = await (await fetch("/state")).json());
  const next = nextUnanswered(-1);
  await show(next === -1 ? 0 : next);
  requestAnimationFrame(draw);
}

async function show(index) {
  current = index;
  tags = { top: pair().top_error, bottom: pair().bottom_error };
  render();

  const blobs = await Promise.all(SIDES.map(side =>
    fetch(`/clip/${pair().pair_id}/${side}`).then(r => r.blob())));
  if (index !== current) return;  // the user moved on while the clips were loading
  SIDES.forEach((side, k) => {
    URL.revokeObjectURL(video(side).src);
    video(side).src = URL.createObjectURL(blobs[k]);
    video(side).onloadedmetadata = () => document.querySelector("main").style.setProperty(
      "--aspect", video(side).videoWidth / video(side).videoHeight);
    video(side).play();
  });

  // build the next pair's clips in the background
  const next = nextUnanswered(current);
  if (next !== -1)
    SIDES.forEach(side => fetch(`/clip/${pairs[next].pair_id}/${side}`));
}

async function answer(choice) {
  const p = pair();
  p.choice = choice; p.top_error = tags.top; p.bottom_error = tags.bottom;
  await fetch("/annotate", { method: "POST", body: JSON.stringify(
    { pair_id: p.pair_id, choice, top_error: tags.top, bottom_error: tags.bottom }) });
  const next = nextUnanswered(current);
  if (next !== -1) await show(next); else render();
}

function render() {
  document.getElementById("task").textContent = pair().task;
  document.getElementById("position").textContent = `${current + 1} / ${pairs.length}`;
  const answered = pair().choice !== null;
  document.getElementById("saved").textContent = answered ? `answered: ${pair().choice}` : "";
  SIDES.forEach(side => {
    document.getElementById(`tag-${side}`).textContent = tags[side] ? "goes wrong" : "";
    document.getElementById(`clip-${side}`).classList.toggle("chosen", pair().choice === side);
  });
}

function draw() {
  for (const side of SIDES) {
    const clip = pair()[side];
    const canvas = document.getElementById(`timeline-${side}`);
    const width = canvas.width = canvas.clientWidth;
    const x = f => f / clip.length * width;
    const frame = Math.min(clip.length - 1, video(side).currentTime * fps);

    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#2a2d33"; ctx.fillRect(0, 0, width, 12);
    ctx.fillStyle = "#e6c15f";
    ctx.fillRect(x(clip.highlight[0]), 0, x(clip.highlight[1]) - x(clip.highlight[0]), 12);
    ctx.fillStyle = "#fff"; ctx.fillRect(x(frame) - 1, 0, 2, 12);

    // the scored interval is also outlined on the video, so it is visible while watching
    const scored = frame >= clip.highlight[0] && frame <= clip.highlight[1];
    document.getElementById(`clip-${side}`).classList.toggle("scored", scored);
  }
  requestAnimationFrame(draw);
}

document.onkeydown = ev => {
  const key = ev.key.toLowerCase();
  if (key === "arrowup") { answer("top"); }
  else if (key === "arrowdown") { answer("bottom"); }
  else if (key === "0") { answer("tie"); }
  else if (key === "1" || key === "2") { const side = SIDES[Number(key) - 1]; tags[side] = 1 - tags[side]; }
  else if (key === " ") { SIDES.forEach(s => video(s).paused ? video(s).play() : video(s).pause()); }
  else if (key === "r") { SIDES.forEach(s => { video(s).currentTime = 0; video(s).play(); }); }
  else if (key === "p" && current > 0) { show(current - 1); }
  else return;
  ev.preventDefault();
  render();
};

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # RUN_NAME = "resnet101_ffnn_hlgauss_0.75"  # checkpoints/<run_name>/best.ckpt
    RUN_NAME = "resnet101_ffnn"
    LOOKAHEAD_SECONDS = 1.0  # N in the advantage, 50 frames at 50 fps as in the paper
    CONTEXT_SECONDS = 1.0  # shown before and after the scored interval
    N_EPISODES = 100  # test episodes to sample from, up to two pairs each
    SEED = 42
    DEVICE = "cuda"
    PORT = 8765

    main(
        run_name=RUN_NAME,
        lookahead_seconds=LOOKAHEAD_SECONDS,
        context_seconds=CONTEXT_SECONDS,
        n_episodes=N_EPISODES,
        seed=SEED,
        device=DEVICE,
        port=PORT,
    )
