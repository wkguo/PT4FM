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

"""Visualize the value function V(o_t) across a whole episode (paper-style).

Unlike :mod:`visualize_advantage` (which shows the noisy per-frame advantage),
this plots the *continuous value* ``value_current`` = V(o_t) in [-1, 0] over the
whole demo, so the systematic progress is visible: a successful episode climbs
toward 0, a failed one plateaus below it. It mirrors the RECAP paper's value
figure: a frame strip on top + the smooth V curve below, with the largest value
*rise* (green) and *drop* (red) intervals auto-highlighted as shaded bands.

Reads ``meta/advantages_{tag}.parquet`` (needs the ``value_current`` column).

Example:
    python -m pt4fm.process.visualize_value \
        --dataset /data/ram_rl_ready --tag ram_N10_q30 --num-frames 8 --out outputs/value_viz
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pt4fm.process.visualize_advantage import (
    _camera_keys,
    _decode_frames,
    _episode_parquet,
    _episode_success,
    _load_info,
    _video_path,
)


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average (odd window k), edge-padded."""
    if k <= 1 or len(y) <= 2:
        return y
    k = min(k if k % 2 else k + 1, len(y) if len(y) % 2 else len(y) - 1)
    k = max(k, 1)
    pad = k // 2
    yp = np.pad(y, pad, mode="edge")
    kern = np.ones(k) / k
    return np.convolve(yp, kern, mode="valid")


def _extreme_bands(v: np.ndarray, win: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (rise_band, drop_band) index intervals with the largest +/- ΔV
    over a sliding window of `win` frames."""
    n = len(v)
    win = max(2, min(win, n - 1))
    deltas = v[win:] - v[:-win]  # ΔV over `win` frames, index i -> [i, i+win]
    if len(deltas) == 0:
        return (0, min(win, n - 1)), (0, min(win, n - 1))
    i_rise = int(np.argmax(deltas))
    i_drop = int(np.argmin(deltas))
    return (i_rise, i_rise + win), (i_drop, i_drop + win)


def visualize_episode(root: Path, info: dict, adv: pd.DataFrame, ep: int,
                      num_frames: int, cams: list[str], success: bool | None,
                      out_dir: Path, smooth: int = 0, bands: bool = True) -> Path:
    ep_v = adv[adv["episode_index"] == ep].sort_values("frame_index")
    if len(ep_v) == 0:
        raise ValueError(f"No rows for episode {ep}")
    if "value_current" not in ep_v:
        raise ValueError("parquet lacks 'value_current'; rerun stage-3 compute_advantages")

    df = pq.read_table(_episode_parquet(root, info, ep), columns=["frame_index", "timestamp"]).to_pandas()
    ep_v = ep_v.merge(df, on="frame_index", how="left")
    if ep_v["timestamp"].isna().any():
        ep_v["timestamp"] = ep_v["frame_index"] / float(info.get("fps", 1.0))

    frame_idx = ep_v["frame_index"].to_numpy()
    ts = ep_v["timestamp"].to_numpy()
    v = ep_v["value_current"].to_numpy().astype(float)
    n = len(v)

    # auto event bands (largest rise / drop over ~8% of episode), detected on raw V
    win = max(3, n // 12)
    (r0, r1), (d0, d1) = _extreme_bands(v, win)
    v_s = _smooth(v, smooth) if smooth and smooth > 1 else None

    K = min(num_frames, n)
    sample_pos = np.linspace(0, n - 1, K).astype(int)
    sample_fidx = [int(frame_idx[p]) for p in sample_pos]
    sample_ts = [float(ts[p]) for p in sample_pos]

    frames_by_cam = {}
    for cam in cams:
        vp = _video_path(root, info, ep, cam)
        frames_by_cam[cam] = _decode_frames(vp, sample_fidx) if vp.exists() else {}

    nrows = len(cams) + 1
    fig = plt.figure(figsize=(2.0 * K, 2.0 * len(cams) + 3.2))
    gs = fig.add_gridspec(nrows, K, height_ratios=[2] * len(cams) + [3], hspace=0.28, wspace=0.05)

    for r, cam in enumerate(cams):
        for c, fi in enumerate(sample_fidx):
            ax = fig.add_subplot(gs[r, c])
            img = frames_by_cam[cam].get(fi)
            if img is not None:
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(cam.split(".")[-1], fontsize=11)
            if r == 0:
                ax.set_title(f"t={sample_ts[c]:.1f}s", fontsize=9)

    axp = fig.add_subplot(gs[len(cams), :])
    axp.plot(ts, v, "-", color="#1f1f1f", lw=1.2, label="V(o_t)")  # raw value, no smoothing
    if v_s is not None:
        axp.plot(ts, v_s, "-", color="tab:blue", lw=1.8, alpha=0.75, label=f"smoothed (k={smooth})")
    if bands:
        axp.axvspan(ts[r0], ts[r1], color="tab:green", alpha=0.22, label="largest value rise")
        axp.axvspan(ts[d0], ts[d1], color="tab:red", alpha=0.22, label="largest value drop")
    for t in sample_ts:
        axp.axvline(t, color="gray", ls="--", lw=0.6, alpha=0.45)
    axp.set_xlabel("Time (s)", fontsize=12); axp.set_ylabel("Value", fontsize=12)
    axp.set_ylim(min(-1.0, v.min() - 0.03), min(0.05, v.max() + 0.05))
    axp.legend(loc="lower right", fontsize=8); axp.grid(alpha=0.25)

    tag = "SUCCESS" if success else ("FAILURE" if success is False else "UNKNOWN")
    fig.suptitle(f"episode {ep}  [{tag}]   V: {v[0]:.2f} -> {v[-1]:.2f}   frames={n}", fontsize=13)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"value_ep{ep:06d}_{tag.lower()}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize the value function V(o_t) over an episode")
    p.add_argument("--dataset", required=True)
    p.add_argument("--tag", default=None, help="advantages_{tag}.parquet (None = advantages.parquet)")
    p.add_argument("--out", default="outputs/value_viz")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--cameras", default=None,
                   help="comma-separated camera keys; default = all video features")
    p.add_argument("--success-ep", type=int, default=None)
    p.add_argument("--failure-ep", type=int, default=None)
    p.add_argument("--smooth", type=int, default=0,
                   help="optional moving-average window for an extra overlaid line (0 = raw only)")
    p.add_argument("--no-bands", action="store_true", help="hide the rise/drop highlight bands")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.dataset)
    info = _load_info(root)
    name = f"advantages_{args.tag}.parquet" if args.tag else "advantages.parquet"
    adv = pd.read_parquet(root / "meta" / name)

    cams = args.cameras.split(",") if args.cameras else _camera_keys(info)

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
        out = visualize_episode(root, info, adv, ep, args.num_frames, cams, is_succ,
                                Path(args.out), smooth=args.smooth, bands=not args.no_bands)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
