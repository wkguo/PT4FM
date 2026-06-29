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

"""sft2rl — make a (SFT) LeRobot dataset RL-ready for the PT4FM pipeline.

A LeRobot demonstration dataset (all-success, no reward) cannot drive offline RL
on its own: RECAP/PT4FM derive the reward from a per-episode **success/failure**
signal (`compute_returns.py` reads an ``is_success`` column from each parquet;
``sft`` data is all-success, ``rollout`` data needs real labels). This tool
**materializes that RL contract** onto any LeRobot v2.1 dataset so that the SAME
preprocessing handles today's demos and the hundreds of hours of self-collected
data to come.

What it writes (per episode, constant within an episode unless noted):
  * ``is_success`` (int64 0/1)            — REQUIRED by the RL pipeline.
  * ``done``       (int64, 1 at last step) — optional (`--add-done`).
  * ``reward``     (float32)               — optional (`--add-reward`); RECAP
        recomputes reward from is_success in stage-1, so this is only for other
        tooling / inspection. ``-1`` per step, terminal ``0`` (success) or
        ``failure_reward`` (failure).

Where the success label comes from (priority):
  1. ``--success-map FILE``         json/csv: episode_index (or source name) -> 0/1
  2. ``--success-from-episode-field`` read a field from meta/episodes.jsonl
  3. ``--default-success {true,false}`` (default true; correct for clean demos)

Design for scale (hundreds of hours):
  * Only the small per-episode parquets are rewritten; videos are **symlinked**
    by default (``--videos symlink|hardlink|copy|skip``) — near-zero extra disk.
  * Parallel across episodes (``--num-workers``); idempotent (``--force``).
  * Source dataset is never modified; output is a self-contained LeRobot dir.

Example:
    python -m pt4fm.data.sft2rl \
        --src  /data/lerobot_demos \
        --dst  /data/lerobot_demos_rlready \
        --default-success true --add-done --num-workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger("pt4fm.sft2rl")

# Feature schema (LeRobot info.json) for the columns we add.
_NEW_FEATURE_SPEC = {
    "is_success": {"dtype": "int64", "shape": [1], "names": None},
    "done": {"dtype": "int64", "shape": [1], "names": None},
    "reward": {"dtype": "float32", "shape": [1], "names": None},
}


# --------------------------------------------------------------------------- #
# meta / layout helpers
# --------------------------------------------------------------------------- #
def load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def load_episodes(root: Path) -> list[dict]:
    path = root / "meta" / "episodes.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def episode_parquet_relpath(info: dict, episode_index: int) -> str:
    """Render data_path template, e.g. data/chunk-000/episode_000001.parquet."""
    chunks_size = int(info.get("chunks_size", info.get("total_episodes", 1)) or 1)
    episode_chunk = episode_index // chunks_size
    return info["data_path"].format(
        episode_chunk=episode_chunk, episode_index=episode_index
    )


def video_relpaths(info: dict, episode_index: int) -> list[str]:
    """All video file relpaths for one episode (one per video feature)."""
    tmpl = info.get("video_path")
    if not tmpl:
        return []
    chunks_size = int(info.get("chunks_size", info.get("total_episodes", 1)) or 1)
    episode_chunk = episode_index // chunks_size
    out = []
    for key, feat in info.get("features", {}).items():
        if feat.get("dtype") == "video":
            out.append(
                tmpl.format(
                    episode_chunk=episode_chunk,
                    video_key=key,
                    episode_index=episode_index,
                )
            )
    return out


# --------------------------------------------------------------------------- #
# success resolution
# --------------------------------------------------------------------------- #
def load_success_map(path: Path) -> dict[str, int]:
    """Load {episode_index|name: 0/1} from a json or csv file."""
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
        return {str(k): int(bool(v)) for k, v in raw.items()}
    # csv: columns include episode_index (or name) and is_success/success
    df = pd.read_csv(path)
    key_col = next((c for c in ("episode_index", "name", "source_demo_id") if c in df.columns), df.columns[0])
    val_col = next((c for c in ("is_success", "success", "label") if c in df.columns), df.columns[-1])
    return {str(k): int(bool(v)) for k, v in zip(df[key_col], df[val_col])}


def resolve_success(
    episode_index: int,
    episode_meta: Optional[dict],
    success_map: Optional[dict[str, int]],
    success_field: Optional[str],
    default_success: bool,
) -> int:
    """Return 1/0 for an episode following the documented priority order."""
    if success_map is not None:
        for key in (str(episode_index), str(episode_index).zfill(6)):
            if key in success_map:
                return success_map[key]
        if episode_meta is not None:
            for cand in ("source_demo_id", "name", "demo_id"):
                v = episode_meta.get(cand)
                if v is not None and str(v) in success_map:
                    return success_map[str(v)]
        # fall through to default if not found
    if success_field and episode_meta is not None and success_field in episode_meta:
        return int(bool(episode_meta[success_field]))
    return int(bool(default_success))


# --------------------------------------------------------------------------- #
# per-episode conversion (runs in a worker process)
# --------------------------------------------------------------------------- #
def _scalar_stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {
        "min": [float(a.min())],
        "max": [float(a.max())],
        "mean": [float(a.mean())],
        "std": [float(a.std())],
        "count": [int(a.size)],
    }


def convert_one_episode(args: dict) -> dict:
    """Add RL columns to one episode parquet; link its videos. Returns stats."""
    src_pq = Path(args["src_pq"])
    dst_pq = Path(args["dst_pq"])
    is_success = int(args["is_success"])
    add_done = args["add_done"]
    add_reward = args["add_reward"]
    failure_reward = float(args["failure_reward"])

    table = pq.read_table(src_pq)
    df = table.to_pandas()
    n = len(df)

    df["is_success"] = np.full(n, is_success, dtype=np.int64)
    stats = {"is_success": _scalar_stats(df["is_success"].to_numpy())}

    if add_done:
        done = np.zeros(n, dtype=np.int64)
        if n > 0:
            done[-1] = 1
        df["done"] = done
        stats["done"] = _scalar_stats(done)

    if add_reward:
        reward = np.full(n, -1.0, dtype=np.float32)
        if n > 0:
            reward[-1] = 0.0 if is_success else failure_reward
        df["reward"] = reward
        stats["reward"] = _scalar_stats(reward)

    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_pq, index=False)

    # link videos
    for rel in args["video_links"]:
        s = Path(args["src_root"]) / rel
        d = Path(args["dst_root"]) / rel
        if not s.exists():
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists() or d.is_symlink():
            d.unlink()
        mode = args["videos"]
        if mode == "symlink":
            d.symlink_to(s.resolve())
        elif mode == "hardlink":
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
        elif mode == "copy":
            shutil.copy2(s, d)
        # "skip" -> nothing

    return {"episode_index": int(args["episode_index"]), "length": n, "stats": stats}


# --------------------------------------------------------------------------- #
# dataset-level driver
# --------------------------------------------------------------------------- #
def convert_dataset(
    src: Path,
    dst: Path,
    *,
    success_map_path: Optional[Path] = None,
    success_field: Optional[str] = None,
    default_success: bool = True,
    add_done: bool = False,
    add_reward: bool = False,
    failure_reward: float = -300.0,
    videos: str = "symlink",
    num_workers: int = 8,
    force: bool = False,
) -> None:
    src, dst = Path(src), Path(dst)
    if dst.exists():
        if not force:
            raise FileExistsError(f"{dst} exists; pass --force to overwrite.")
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    info = load_info(src)
    if info.get("codebase_version", "v2.1") not in ("v2.0", "v2.1"):
        logger.warning("Untested codebase_version=%s; proceeding best-effort.",
                       info.get("codebase_version"))
    episodes = load_episodes(src)
    ep_meta_by_idx = {int(e["episode_index"]): e for e in episodes}
    total = int(info["total_episodes"])
    success_map = load_success_map(success_map_path) if success_map_path else None

    # Build per-episode jobs.
    jobs = []
    n_success = 0
    for ep in range(total):
        is_succ = resolve_success(
            ep, ep_meta_by_idx.get(ep), success_map, success_field, default_success
        )
        n_success += is_succ
        rel = episode_parquet_relpath(info, ep)
        jobs.append({
            "episode_index": ep,
            "src_pq": str(src / rel),
            "dst_pq": str(dst / rel),
            "src_root": str(src),
            "dst_root": str(dst),
            "is_success": is_succ,
            "add_done": add_done,
            "add_reward": add_reward,
            "failure_reward": failure_reward,
            "videos": videos,
            "video_links": video_relpaths(info, ep),
        })

    logger.info(
        "Converting %d episodes (%d success / %d failure) src=%s -> dst=%s "
        "[videos=%s, workers=%d]",
        total, n_success, total - n_success, src.name, dst.name, videos, num_workers,
    )

    results: list[dict] = []
    if num_workers and num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futs = [ex.submit(convert_one_episode, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs)):
                results.append(fut.result())
                if (i + 1) % 100 == 0:
                    logger.info("  %d/%d episodes done", i + 1, total)
    else:
        for i, j in enumerate(jobs):
            results.append(convert_one_episode(j))
            if (i + 1) % 100 == 0:
                logger.info("  %d/%d episodes done", i + 1, total)

    results.sort(key=lambda r: r["episode_index"])
    _write_meta(src, dst, info, episodes, results, add_done, add_reward)
    logger.info("Done. RL-ready dataset at %s", dst)
    logger.info(
        "Next: declare this dataset as type 'rollout' (so failures count) or "
        "'sft' in the PT4FM data config, then run stage 1 (compute_returns)."
    )


def _write_meta(src, dst, info, episodes, results, add_done, add_reward):
    src, dst = Path(src), Path(dst)
    # 1) info.json: add new features.
    new_info = json.loads(json.dumps(info))  # deep copy
    feats = new_info.setdefault("features", {})
    for col, present in (("is_success", True), ("done", add_done), ("reward", add_reward)):
        if present:
            feats[col] = json.loads(json.dumps(_NEW_FEATURE_SPEC[col]))
    (dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=2))

    # 2) copy episodes.jsonl, tasks.jsonl, and any extra meta (processed_demos, etc.).
    for name in ("episodes.jsonl", "tasks.jsonl", "processed_demos.json"):
        sp = src / "meta" / name
        if sp.exists():
            shutil.copy2(sp, dst / "meta" / name)

    # 3) episodes_stats.jsonl: merge in stats for the new columns.
    stats_by_ep = {r["episode_index"]: r["stats"] for r in results}
    src_es = src / "meta" / "episodes_stats.jsonl"
    dst_es = dst / "meta" / "episodes_stats.jsonl"
    if src_es.exists():
        lines = []
        for line in src_es.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ep = int(rec["episode_index"])
            rec.setdefault("stats", {}).update(stats_by_ep.get(ep, {}))
            lines.append(json.dumps(rec))
        dst_es.write_text("\n".join(lines) + "\n")
    else:
        # Synthesize a minimal episodes_stats.jsonl from what we computed.
        lines = [json.dumps({"episode_index": ep, "stats": st})
                 for ep, st in sorted(stats_by_ep.items())]
        dst_es.write_text("\n".join(lines) + "\n")

    # 4) stats.json (global), only if the source had one.
    src_stats = src / "meta" / "stats.json"
    if src_stats.exists():
        gstats = json.loads(src_stats.read_text())
        for col in ("is_success", "done", "reward"):
            arrs = [r["stats"][col] for r in results if col in r["stats"]]
            if arrs:
                mins = float(min(a["min"][0] for a in arrs))
                maxs = float(max(a["max"][0] for a in arrs))
                counts = sum(a["count"][0] for a in arrs)
                mean = float(sum(a["mean"][0] * a["count"][0] for a in arrs) / max(counts, 1))
                gstats[col] = {"min": [mins], "max": [maxs], "mean": [mean],
                               "std": [0.0], "count": [counts]}
        (dst / "meta" / "stats.json").write_text(json.dumps(gstats, indent=2))


# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    p = argparse.ArgumentParser(description="Make a LeRobot SFT dataset RL-ready for PT4FM")
    p.add_argument("--src", required=True, help="source LeRobot dataset root")
    p.add_argument("--dst", required=True, help="output (RL-ready) dataset root")
    g = p.add_argument_group("success label source")
    g.add_argument("--success-map", default=None,
                   help="json/csv mapping episode_index (or source name) -> 0/1")
    g.add_argument("--success-from-episode-field", default=None,
                   help="field name in meta/episodes.jsonl holding the success flag")
    g.add_argument("--default-success", choices=["true", "false"], default="true",
                   help="fallback when no map/field matches (default: true)")
    p.add_argument("--add-done", action="store_true", help="add done column (1 at last step)")
    p.add_argument("--add-reward", action="store_true",
                   help="add reward column (-1/step, terminal 0|failure_reward)")
    p.add_argument("--failure-reward", type=float, default=-300.0)
    p.add_argument("--videos", choices=["symlink", "hardlink", "copy", "skip"], default="symlink")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    convert_dataset(
        src=Path(args.src),
        dst=Path(args.dst),
        success_map_path=Path(args.success_map) if args.success_map else None,
        success_field=args.success_from_episode_field,
        default_success=(args.default_success == "true"),
        add_done=args.add_done,
        add_reward=args.add_reward,
        failure_reward=args.failure_reward,
        videos=args.videos,
        num_workers=args.num_workers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
