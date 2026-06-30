# Copyright 2026 The PT4FM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Overlay the per-frame value V(o_t) on each camera video, colored by trend.

For every frame ``t`` the value ``value_current`` (from stage-3
``advantages_{tag}.parquet``) is drawn in the top-left corner. The color marks
"drawdown" -- a drop below an earlier peak that has not yet recovered:

  * RED   if V(o_t) < max(V(o_0..o_{t-1}))  -- below an earlier peak, not recovered
           (since some past peak the value dropped and has not climbed back above it)
  * GREEN if V(o_t) >= the running max so far -- a new running high, the value is growing

So after a peak, every following frame that stays below that peak is red, until the
value rises back above the peak (a new high), at which point it turns green again.

Outputs mirror the dataset's ``videos/`` tree under ``--out-subdir`` (default
``videos-w-value``), one file per camera, e.g.
``videos-w-value/chunk-000/observation.images.view1/episode_000000.mp4``.

Example:
    python -m pt4fm.process.annotate_value_video \
        --dataset /data/ram_rl_ready --tag ram_N10_q30 --success-ep 1 --failure-ep 186
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def _camera_keys(info: dict) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def _in_video_path(root: Path, info: dict, ep: int, key: str) -> Path:
    cs = int(info.get("chunks_size", info.get("total_episodes", 1)) or 1)
    return root / info["video_path"].format(episode_chunk=ep // cs, video_key=key, episode_index=ep)


def _out_video_path(root: Path, info: dict, ep: int, key: str, out_subdir: str) -> Path:
    cs = int(info.get("chunks_size", info.get("total_episodes", 1)) or 1)
    rel = info["video_path"].format(episode_chunk=ep // cs, video_key=key, episode_index=ep)
    # swap the leading "videos" component for the output subdir
    parts = Path(rel).parts
    parts = (out_subdir,) + parts[1:]
    return root / Path(*parts)


def _trend_colors(values: np.ndarray, tol: float = 0.0) -> list[tuple[int, int, int]]:
    """BGR color per frame using a drawdown rule: red if V_t is below the running
    max of all earlier frames by more than `tol` (below a prior peak, not yet
    recovered), else green (at/above the running high -> growing). `tol` lets small
    noise-level dips stay green."""
    n = len(values)
    red, green = (0, 0, 255), (0, 170, 0)
    colors = [green] * n
    prefix_max = -np.inf  # max of values strictly before the current index
    for t in range(n):
        colors[t] = red if values[t] < prefix_max - tol else green
        prefix_max = max(prefix_max, values[t])
    return colors


def annotate_video(in_path: Path, out_path: Path, values: np.ndarray, colors,
                   fps: float) -> int:
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {in_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, 0.5 * W / 320.0)
    thick = max(1, int(round(scale * 2)))
    n = len(values)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        j = min(i, n - 1)
        v, color = float(values[j]), colors[j]
        label = f"V={v:+.3f}"
        (tw, th), base = cv2.getTextSize(label, font, scale, thick)
        x, y = 6, 6 + th
        cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
        cv2.putText(frame, label, (x, y), font, scale, color, thick, cv2.LINE_AA)
        # small trend swatch next to the text
        cv2.rectangle(frame, (x + tw + 10, y - th), (x + tw + 10 + th, y), color, -1)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    return i


def main() -> None:
    p = argparse.ArgumentParser(description="Overlay per-frame value on camera videos, colored by trend")
    p.add_argument("--dataset", required=True)
    p.add_argument("--tag", default=None, help="advantages_{tag}.parquet (None = advantages.parquet)")
    p.add_argument("--success-ep", type=int, default=None, help="episode id to annotate")
    p.add_argument("--failure-ep", type=int, default=None, help="episode id to annotate")
    p.add_argument("--episodes", default=None, help="extra comma list (e.g. 108,150) or 'all'")
    p.add_argument("--cameras", default=None, help="comma list of camera keys; default = all video features")
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="drawdown deadband: dips smaller than this below the prior peak stay green")
    p.add_argument("--out-subdir", default="videos-w-value")
    args = p.parse_args()

    root = Path(args.dataset)
    info = _load_info(root)
    fps = float(info.get("fps", 10.0))
    cams = args.cameras.split(",") if args.cameras else _camera_keys(info)

    name = f"advantages_{args.tag}.parquet" if args.tag else "advantages.parquet"
    adv = pd.read_parquet(root / "meta" / name)
    if "value_current" not in adv.columns:
        raise ValueError("parquet lacks 'value_current'; rerun stage-3 compute_advantages")

    all_eps = sorted(adv["episode_index"].unique().tolist())
    eps: list[int] = []
    for v in (args.success_ep, args.failure_ep):
        if v is not None:
            eps.append(int(v))
    if args.episodes:
        eps += all_eps if args.episodes.strip().lower() == "all" else [int(x) for x in args.episodes.split(",")]
    eps = list(dict.fromkeys(eps))  # de-dup, keep order
    if not eps:
        p.error("specify at least one of --success-ep / --failure-ep / --episodes")

    for ep in eps:
        e = adv[adv["episode_index"] == ep].sort_values("frame_index")
        if len(e) == 0:
            print(f"[skip] episode {ep}: no rows")
            continue
        values = e["value_current"].to_numpy().astype(float)
        colors = _trend_colors(values, tol=args.tolerance)
        for cam in cams:
            ip = _in_video_path(root, info, ep, cam)
            op = _out_video_path(root, info, ep, cam, args.out_subdir)
            if not ip.exists():
                print(f"[skip] {ip} (missing)")
                continue
            nf = annotate_video(ip, op, values, colors, fps)
            red = sum(1 for c in colors if c == (0, 0, 255))
            print(f"ep{ep:06d} {cam.split('.')[-1]}: {nf} frames -> {op}  (red {red}/{len(values)})")


if __name__ == "__main__":
    main()
