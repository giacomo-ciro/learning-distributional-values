import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder

from data import CAMERAS, ValueDataModule

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_COLUMNS = ["episode_index", "start_frame", "end_frame"]


def load_data_config(config_path: Path) -> DictConfig:
    # only the data section and seed are used: the hydra defaults list is not resolved
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    # the data root and split dir in the config are relative to the repo root
    cfg.data.root = str(REPO_ROOT / cfg.data.root)
    cfg.data.split_dir = str(REPO_ROOT / cfg.data.split_dir)
    return cfg


def load_val_episodes(cfg: DictConfig) -> tuple[pd.DataFrame, Path, str, int]:
    # the split is computed by the datamodule itself, so it matches training exactly
    datamodule = ValueDataModule(cfg)
    datamodule.setup()
    frames = datamodule.datasets["val"].frames

    episode_table = frames.meta.episodes
    assert episode_table is not None
    episodes = cast(pd.DataFrame, episode_table.to_pandas()).set_index("episode_index")
    val_episodes = episodes.loc[datamodule.split_episodes["val"]]
    return val_episodes, frames.root, frames.meta.video_path, frames.meta.fps


def presentation_order(episodes: pd.DataFrame, seed: int) -> list[int]:
    # shuffle each outcome, then alternate success / failure so any prefix is balanced
    rng = np.random.default_rng(seed)
    successes = rng.permutation(episodes.index[episodes["success"]]).tolist()
    failures = rng.permutation(episodes.index[~episodes["success"]]).tolist()

    order = []
    for i in range(max(len(successes), len(failures))):
        order += successes[i : i + 1] + failures[i : i + 1]
    return order


def load_annotations(csv_path: Path) -> dict[int, list[list[int]]]:
    # episode -> windows; an episode with a single empty row was reviewed and has no failure
    if not csv_path.exists():
        return {}
    table = pd.read_csv(csv_path, dtype="Int64")

    annotations: dict[int, list[list[int]]] = {}
    for row in table.itertuples():
        windows = annotations.setdefault(int(row.episode_index), [])
        if not pd.isna(row.start_frame):
            windows.append([int(row.start_frame), int(row.end_frame)])
    return annotations


def save_annotations(annotations: dict[int, list[list[int]]], csv_path: Path) -> None:
    rows = []
    for episode_index in sorted(annotations):
        windows = sorted(annotations[episode_index])
        if not windows:
            rows.append([episode_index, pd.NA, pd.NA])
        for start_frame, end_frame in windows:
            rows.append([episode_index, start_frame, end_frame])

    # write to a temporary file first so a crash never leaves a truncated csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(".tmp")
    pd.DataFrame(rows, columns=CSV_COLUMNS).astype("Int64").to_csv(
        tmp_path, index=False
    )
    tmp_path.replace(csv_path)


def build_clip(
    episode: pd.Series, data_root: Path, video_path: str, fps: int, clip_path: Path
) -> None:
    # decode the episode from each camera and tile the cameras side by side
    views = []
    for camera in CAMERAS:
        file_path = data_root / video_path.format(
            video_key=camera,
            chunk_index=int(episode[f"videos/{camera}/chunk_index"]),
            file_index=int(episode[f"videos/{camera}/file_index"]),
        )
        decoded = VideoDecoder(str(file_path)).get_frames_played_in_range(
            float(episode[f"videos/{camera}/from_timestamp"]),
            float(episode[f"videos/{camera}/to_timestamp"]),
        )
        assert len(decoded.data) == episode["length"], (
            "decoded frames != episode length"
        )
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


def make_handler(
    episodes: pd.DataFrame,
    order: list[int],
    annotations: dict[int, list[list[int]]],
    csv_path: Path,
    data_root: Path,
    video_path: str,
    fps: int,
    clip_dir: Path,
) -> type[BaseHTTPRequestHandler]:
    clip_lock = (
        threading.Lock()
    )  # the page pre-fetches the next clip while the current one loads

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
                    "episodes": [
                        {
                            "episode_index": episode_index,
                            "success": bool(episodes.loc[episode_index, "success"]),
                            "length": int(episodes.loc[episode_index, "length"]),
                        }
                        for episode_index in order
                    ],
                    "annotations": annotations,
                }
                self.send(json.dumps(state).encode(), "application/json")

            elif self.path.startswith("/clip/"):
                episode_index = int(self.path.removeprefix("/clip/"))
                clip_path = clip_dir / f"episode_{episode_index:06d}.mp4"
                with clip_lock:
                    if not clip_path.exists():
                        build_clip(
                            episodes.loc[episode_index],
                            data_root,
                            video_path,
                            fps,
                            clip_path,
                        )
                self.send(clip_path.read_bytes(), "video/mp4")

            else:
                self.send_error(404)

        def do_POST(self) -> None:
            assert self.path == "/annotate"
            # replace every window of the episode with the submitted ones
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            annotations[int(body["episode_index"])] = body["windows"]
            save_annotations(annotations, csv_path)
            self.send(b"{}", "application/json")

        def log_message(self, format: str, *args) -> None:
            pass

    return Handler


def main(config_path: Path, csv_path: Path, clip_dir: Path, port: int) -> None:
    cfg = load_data_config(config_path)
    episodes, data_root, video_path, fps = load_val_episodes(cfg)
    order = presentation_order(episodes, cfg.seed)

    # previous annotations are loaded so they can be revised
    annotations = load_annotations(csv_path)
    print(f"{len(annotations)} / {len(order)} val episodes already annotated")

    # clips are cached per dataset
    clip_dir = clip_dir / Path(cfg.data.root).name
    handler = make_handler(
        episodes, order, annotations, csv_path, data_root, video_path, fps, clip_dir
    )
    print(f"Annotate at http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Failure annotation</title>
<style>
  body { margin: 0; padding: 16px; background: #16181c; color: #e6e6e6; font: 14px system-ui, sans-serif; }
  main { max-width: 1344px; margin: 0 auto; }
  header { display: flex; gap: 16px; align-items: baseline; margin-bottom: 8px; }
  h1 { font-size: 18px; margin: 0; }
  .success { color: #5fbf77; } .failure { color: #e5645e; } .muted { color: #8a8f98; }
  video { width: 100%; background: #000; display: block; image-rendering: pixelated; }
  canvas { width: 100%; height: 32px; display: block; margin-top: 8px; cursor: pointer; }
  #status { display: flex; gap: 16px; margin: 8px 0; font-variant-numeric: tabular-nums; }
  ul { list-style: none; padding: 0; margin: 8px 0; }
  li { display: inline-flex; gap: 6px; align-items: center; margin: 0 8px 8px 0; padding: 4px 8px; background: #2a2d33; border-radius: 4px; cursor: pointer; }
  li button { background: none; border: 0; color: #e5645e; cursor: pointer; font-size: 14px; }
  kbd { background: #2a2d33; border-radius: 3px; padding: 1px 5px; }
  .help { line-height: 2; }
</style>
</head>
<body>
<main>
  <header>
    <h1 id="title"></h1><span id="position" class="muted"></span><span id="saved"></span>
    <span id="counts" class="muted" style="margin-left:auto"></span>
  </header>
  <video id="video" muted playsinline></video>
  <canvas id="timeline" height="32"></canvas>
  <div id="status"><span id="frame"></span><span id="speed"></span><span id="pending"></span></div>
  <ul id="windows"></ul>
  <div class="help muted">
    <kbd>Space</kbd> play/pause &nbsp; <kbd>←</kbd><kbd>→</kbd> ±1 frame (<kbd>Shift</kbd> ±10) &nbsp;
    <kbd>I</kbd> window start &nbsp; <kbd>O</kbd> window end &nbsp; <kbd>X</kbd> delete window at playhead &nbsp;
    <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> speed 1×/2×/4× &nbsp;
    <kbd>Enter</kbd> submit and next &nbsp; <kbd>P</kbd>/<kbd>N</kbd> previous/next without saving
  </div>
</main>
<script>
const video = document.getElementById("video");
const timeline = document.getElementById("timeline");
let fps, episodes, annotations;
let current = 0, windows = [], pendingStart = null, frame = 0, dirty = false;

const episode = () => episodes[current];
const lastFrame = () => episode().length - 1;

async function init() {
  ({ fps, episodes, annotations } = await (await fetch("/state")).json());
  // resume at the first episode without a submitted annotation
  const next = episodes.findIndex(e => !(e.episode_index in annotations));
  await show(next === -1 ? 0 : next);
  requestAnimationFrame(draw);
}

async function show(index) {
  current = index;
  const saved = annotations[episode().episode_index];
  windows = saved ? saved.map(w => [...w]) : [];
  pendingStart = null; frame = 0; dirty = false;
  render();

  video.pause();
  const blob = await (await fetch(`/clip/${episode().episode_index}`)).blob();
  if (index !== current) return;  // the user moved on while the clip was loading
  URL.revokeObjectURL(video.src);
  video.src = URL.createObjectURL(blob);
  seek(0);

  // build the next clip in the background
  if (current + 1 < episodes.length) fetch(`/clip/${episodes[current + 1].episode_index}`);
}

function seek(f) {
  frame = Math.max(0, Math.min(lastFrame(), f));
  video.currentTime = (frame + 0.5) / fps;  // mid-frame, so the displayed frame is unambiguous
  render();
}

function trackFrame(now, meta) {
  if (!video.paused) { frame = Math.min(lastFrame(), Math.round(meta.mediaTime * fps)); render(); }
  video.requestVideoFrameCallback(trackFrame);
}
video.requestVideoFrameCallback(trackFrame);

function addWindow(endFrame) {
  if (pendingStart === null) return;
  windows.push([Math.min(pendingStart, endFrame), Math.max(pendingStart, endFrame)]);
  windows.sort((a, b) => a[0] - b[0]);
  pendingStart = null; dirty = true;
}

async function submit() {
  await fetch("/annotate", { method: "POST", body: JSON.stringify({ episode_index: episode().episode_index, windows }) });
  annotations[episode().episode_index] = windows.map(w => [...w]);
  if (current + 1 < episodes.length) await show(current + 1); else { dirty = false; render(); }
}

function render() {
  const e = episode();
  const title = document.getElementById("title");
  title.textContent = `Episode ${e.episode_index} · ${e.success ? "success" : "failure"}`;
  title.className = e.success ? "success" : "failure";
  document.getElementById("position").textContent = `${current + 1} / ${episodes.length}`;
  const submitted = e.episode_index in annotations;
  document.getElementById("saved").textContent = dirty ? "unsaved changes" : submitted ? "submitted" : "not annotated";
  document.getElementById("saved").className = dirty ? "failure" : submitted ? "success" : "muted";

  let nSuccess = 0, nFailure = 0;
  for (const ep of episodes) if (ep.episode_index in annotations) ep.success ? nSuccess++ : nFailure++;
  document.getElementById("counts").textContent = `annotated: ${nSuccess} successes, ${nFailure} failures`;

  document.getElementById("frame").textContent = `frame ${frame} / ${lastFrame()}  (${(frame / fps).toFixed(2)} s)`;
  document.getElementById("speed").textContent = `${video.playbackRate}×`;
  document.getElementById("pending").textContent = pendingStart === null ? "" : `window start at ${pendingStart}, press O to close`;

  const list = document.getElementById("windows");
  list.replaceChildren(...windows.map((w, k) => {
    const item = document.createElement("li");
    item.textContent = `${w[0]} – ${w[1]}`;
    item.onclick = () => seek(w[0]);
    const remove = document.createElement("button");
    remove.textContent = "×";
    remove.onclick = ev => { ev.stopPropagation(); windows.splice(k, 1); dirty = true; render(); };
    item.append(remove);
    return item;
  }));
  if (!windows.length) list.innerHTML = '<li class="muted" style="cursor:default">no failure windows</li>';
}

function draw() {
  const width = timeline.width = timeline.clientWidth;
  const ctx = timeline.getContext("2d");
  const x = f => (f + 0.5) / episode().length * width;
  ctx.fillStyle = "#2a2d33"; ctx.fillRect(0, 0, width, 32);
  ctx.fillStyle = "rgba(229, 100, 94, 0.6)";
  for (const [s, e] of windows) ctx.fillRect(x(s), 0, Math.max(2, x(e) - x(s)), 32);
  if (pendingStart !== null) {
    ctx.fillStyle = "rgba(229, 100, 94, 0.3)";
    ctx.fillRect(Math.min(x(pendingStart), x(frame)), 0, Math.abs(x(frame) - x(pendingStart)), 32);
  }
  ctx.fillStyle = "#fff"; ctx.fillRect(x(frame) - 1, 0, 2, 32);
  requestAnimationFrame(draw);
}

timeline.onclick = ev => {
  video.pause();
  seek(Math.floor(ev.offsetX / timeline.clientWidth * episode().length));
};

document.onkeydown = ev => {
  const step = ev.shiftKey ? 10 : 1;
  const key = ev.key.toLowerCase();
  if (key === " ") { video.paused ? video.play() : video.pause(); }
  else if (key === "arrowleft") { video.pause(); seek(frame - step); }
  else if (key === "arrowright") { video.pause(); seek(frame + step); }
  else if (key === "i") { pendingStart = frame; }
  else if (key === "o") { addWindow(frame); }
  else if (key === "x") {
    const k = windows.findIndex(([s, e]) => s <= frame && frame <= e);
    if (k !== -1) { windows.splice(k, 1); dirty = true; }
  }
  else if (key in { 1: 0, 2: 0, 3: 0 }) { video.playbackRate = 2 ** (Number(key) - 1); }
  else if (key === "enter") { submit(); }
  else if (key === "p" && current > 0 && (!dirty || confirm("Discard unsaved changes?"))) { show(current - 1); }
  else if (key === "n" && current + 1 < episodes.length && (!dirty || confirm("Discard unsaved changes?"))) { show(current + 1); }
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
    CONFIG_PATH = (
        REPO_ROOT / "configs" / "train.yaml"
    )  # defines the dataset and the val split
    CSV_PATH = REPO_ROOT / "outputs" / "annotations.csv"
    CLIP_DIR = (
        REPO_ROOT / "cache" / "annotate"
    )  # tiled per-episode clips, safe to delete
    PORT = 8765

    main(
        config_path=CONFIG_PATH,
        csv_path=CSV_PATH,
        clip_dir=CLIP_DIR,
        port=PORT,
    )
