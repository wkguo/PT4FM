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

"""Visualize per-frame advantages over an episode for a LeRobot dataset.

For a chosen episode it renders one figure with:
  * top rows  : one row per camera, K evenly-sampled frames across the episode,
  * bottom row: the FULL per-frame advantage curve (x = timestamp), with the
                sampled frames marked.

By default it picks one success + one failure episode (from episodes.jsonl
``rollout_success``, falling back to the per-episode mean advantage). Reads the
``meta/advantages_{tag}.parquet`` produced by stage-3.

Example:
    python -m pt4fm.process.visualize_advantage \
        --dataset /data/ram_rl_ready --tag ram_N10_q30 --num-frames 8 --out outputs/adv_viz
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def _episode_parquet(root: Path, info: dict, ep: int) -> Path:
    cs = int(info.get("chunks_size", info.get("total_episodes", 1)) or 1)
    return root / info["data_path"].format(episode_chunk=ep // cs, episode_index=ep)


def _video_path(root: Path, info: dict, ep: int, key: str) -> Path:
    cs = int(info.get("chunks_size", info.get("total_episodes", 1)) or 1)
    return root / info["video_path"].format(episode_chunk=ep // cs, video_key=key, episode_index=ep)


def _camera_keys(info: dict) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def _decode_frames(video_path: Path, target_indices: list[int]) -> dict[int, np.ndarray]:
    """Decode the requested 0-based frame indices from a video (RGB HWC uint8)."""
    want = set(int(i) for i in target_indices)
    out: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in want:
                out[i] = frame.to_ndarray(format="rgb24")
                if len(out) == len(want):
                    break
    return out


def _episode_success(episodes_meta: dict, adv_ep: pd.DataFrame, ep: int) -> bool | None:
    m = episodes_meta.get(ep)
    if m is not None:
        for k in ("rollout_success", "is_success"):
            if k in m:
                return bool(m[k])
    return None


def visualize_episode(root: Path, info: dict, adv: pd.DataFrame, ep: int,
                      num_frames: int, success: bool | None, out_dir: Path) -> Path:
    cams = _camera_keys(info)
    ep_adv = adv[adv["episode_index"] == ep].sort_values("frame_index")
    if len(ep_adv) == 0:
        raise ValueError(f"No advantages for episode {ep}")

    # timestamps from the data parquet (fallback to frame_index / fps)
    df = pq.read_table(_episode_parquet(root, info, ep), columns=["frame_index", "timestamp"]).to_pandas()
    ep_adv = ep_adv.merge(df, on="frame_index", how="left")
    if ep_adv["timestamp"].isna().any():
        ep_adv["timestamp"] = ep_adv["frame_index"] / float(info.get("fps", 1.0))

    frame_idx = ep_adv["frame_index"].to_numpy()
    ts = ep_adv["timestamp"].to_numpy()
    adv_c = ep_adv["advantage_continuous"].to_numpy()
    adv_b = ep_adv["advantage"].to_numpy().astype(bool) if "advantage" in ep_adv else (adv_c >= 0)

    n = len(frame_idx)
    K = min(num_frames, n)
    sample_pos = np.linspace(0, n - 1, K).astype(int)
    sample_fidx = [int(frame_idx[p]) for p in sample_pos]
    sample_ts = [float(ts[p]) for p in sample_pos]

    # decode the sampled frames for each camera
    frames_by_cam = {}
    for cam in cams:
        vp = _video_path(root, info, ep, cam)
        frames_by_cam[cam] = _decode_frames(vp, sample_fidx) if vp.exists() else {}

    # figure: len(cams) image rows (K cols) + 1 full-width advantage row
    nrows = len(cams) + 1
    fig = plt.figure(figsize=(2.0 * K, 2.0 * len(cams) + 3.0))
    gs = fig.add_gridspec(nrows, K, height_ratios=[2] * len(cams) + [3], hspace=0.25, wspace=0.05)

    for r, cam in enumerate(cams):
        for c, (fi, t) in enumerate(zip(sample_fidx, sample_ts)):
            ax = fig.add_subplot(gs[r, c])
            img = frames_by_cam[cam].get(fi)
            if img is not None:
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(cam.split(".")[-1], fontsize=11)
            if r == 0:
                ax.set_title(f"t={t:.2f}s", fontsize=9)

    axp = fig.add_subplot(gs[len(cams), :])
    axp.plot(ts, adv_c, "-", color="#1f77b4", lw=1.2, label="advantage")
    pos = adv_b
    axp.scatter(ts[pos], adv_c[pos], s=10, c="tab:green", label="positive", zorder=3)
    axp.scatter(ts[~pos], adv_c[~pos], s=10, c="tab:red", label="negative", zorder=3)
    for t in sample_ts:  # mark the frames shown above
        axp.axvline(t, color="gray", ls="--", lw=0.7, alpha=0.6)
    axp.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    axp.set_xlabel("timestamp (s)"); axp.set_ylabel("advantage")
    axp.legend(loc="best", fontsize=8); axp.grid(alpha=0.3)

    tag = "SUCCESS" if success else ("FAILURE" if success is False else "UNKNOWN")
    fig.suptitle(f"episode {ep}  [{tag}]   frames={n}  pos_rate={100*pos.mean():.0f}%", fontsize=13)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"adv_ep{ep:06d}_{tag.lower()}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize per-frame advantages (cameras + curve)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--tag", default=None, help="advantages_{tag}.parquet (None = advantages.parquet)")
    p.add_argument("--out", default="outputs/adv_viz")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--success-ep", type=int, default=None)
    p.add_argument("--failure-ep", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.dataset)
    info = _load_info(root)
    name = f"advantages_{args.tag}.parquet" if args.tag else "advantages.parquet"
    adv = pd.read_parquet(root / "meta" / name)

    episodes_meta = {}
    ep_jsonl = root / "meta" / "episodes.jsonl"
    if ep_jsonl.exists():
        for line in ep_jsonl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                episodes_meta[int(r["episode_index"])] = r

    eps = sorted(adv["episode_index"].unique().tolist())
    succ = {e: _episode_success(episodes_meta, adv, e) for e in eps}
    rng = random.Random(args.seed)
    succ_eps = [e for e in eps if succ[e] is True]
    fail_eps = [e for e in eps if succ[e] is False]
    chosen = []
    if args.success_ep is not None:
        chosen.append((args.success_ep, True))
    elif succ_eps:
        chosen.append((rng.choice(succ_eps), True))
    if args.failure_ep is not None:
        chosen.append((args.failure_ep, False))
    elif fail_eps:
        chosen.append((rng.choice(fail_eps), False))

    for ep, is_succ in chosen:
        out = visualize_episode(root, info, adv, ep, args.num_frames, is_succ, Path(args.out))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
