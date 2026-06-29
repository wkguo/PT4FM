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

"""merge_datasets — concatenate several LeRobot v2.1 datasets into one.

Episodes from all inputs are concatenated (in the given order) and **renumbered
contiguously** (``episode_index`` 0..N-1, global ``index`` recomputed). Task lists
are unioned into a single global ``tasks.jsonl`` and every frame's ``task_index``
is remapped to the global id (so ``prompt_from_task`` stays correct across the
merged multi-task corpus). Videos are re-linked under the new episode numbers.

Assumes all inputs share the same feature schema (same robot/cameras/dims).

Example:
    python -m pt4fm.data.merge_datasets --out /data/merged \
        --src /data/d0618 --src /data/d0619 --src /data/d0622 --src /data/d0623 --src /data/d0624
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pt4fm.data.sft2rl import (
    episode_parquet_relpath,
    load_episodes,
    load_info,
    video_relpaths,
)

logger = logging.getLogger("pt4fm.merge_datasets")


def _load_tasks(root: Path) -> dict[int, str]:
    out = {}
    for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[int(r["task_index"])] = r["task"]
    return out


def merge_datasets(srcs: list[Path], out: Path, videos: str = "symlink", force: bool = False) -> dict:
    srcs = [Path(s) for s in srcs]
    out = Path(out)
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} exists; pass --force.")
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    base_info = load_info(srcs[0])
    n_video_feats = sum(1 for f in base_info.get("features", {}).values() if f.get("dtype") == "video")

    # 1) global task table (ordered by first appearance).
    global_tasks: list[str] = []
    task_str_to_gid: dict[str, int] = {}
    for s in srcs:
        for _, t in sorted(_load_tasks(s).items()):
            if t not in task_str_to_gid:
                task_str_to_gid[t] = len(global_tasks)
                global_tasks.append(t)

    # merged info with a single chunk holding all episodes.
    merged_info = json.loads(json.dumps(base_info))

    new_episodes, new_stats = [], []
    running_index = 0
    new_idx = 0
    total_frames = 0
    # need chunks_size; defer template rendering until we know total.
    # First pass: count episodes to set chunks_size = total (1 chunk).
    per_src = []
    total_eps = 0
    for s in srcs:
        eps = load_episodes(s)
        per_src.append((s, load_info(s), eps))
        total_eps += len(eps)
    merged_info["chunks_size"] = total_eps
    merged_info["total_chunks"] = 1

    for s, info, eps in per_src:
        local_tasks = _load_tasks(s)
        local_to_global = {li: task_str_to_gid[t] for li, t in local_tasks.items()}
        stats_by_ep = {}
        es_path = s / "meta" / "episodes_stats.jsonl"
        if es_path.exists():
            for line in es_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    stats_by_ep[int(r["episode_index"])] = r.get("stats", {})
        ep_by_idx = {int(e["episode_index"]): e for e in eps}

        for old_ep in sorted(ep_by_idx):
            old_rel = episode_parquet_relpath(info, old_ep)
            new_rel = episode_parquet_relpath(merged_info, new_idx)
            df = pq.read_table(s / old_rel).to_pandas()
            n = len(df)
            df["episode_index"] = np.full(n, new_idx, dtype=df["episode_index"].dtype)
            if "index" in df.columns:
                df["index"] = np.arange(running_index, running_index + n, dtype=df["index"].dtype)
            if "task_index" in df.columns:
                df["task_index"] = df["task_index"].map(lambda x: local_to_global.get(int(x), int(x))).astype(
                    df["task_index"].dtype)
            (out / new_rel).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / new_rel, index=False)
            running_index += n
            total_frames += n

            for old_vrel, new_vrel in zip(video_relpaths(info, old_ep), video_relpaths(merged_info, new_idx)):
                sp = Path(os.path.realpath(s / old_vrel))
                d = out / new_vrel
                if not sp.exists():
                    continue
                d.parent.mkdir(parents=True, exist_ok=True)
                if videos == "symlink":
                    d.symlink_to(sp)
                elif videos == "hardlink":
                    try:
                        os.link(sp, d)
                    except OSError:
                        shutil.copy2(sp, d)
                elif videos == "copy":
                    shutil.copy2(sp, d)

            em = dict(ep_by_idx[old_ep])
            em["episode_index"] = new_idx
            # remap episode-level task fields if present
            if "tasks" in em:
                em["tasks"] = em["tasks"]  # keep human label list as-is
            new_episodes.append(em)
            new_stats.append({"episode_index": new_idx, "stats": stats_by_ep.get(old_ep, {})})
            new_idx += 1

    merged_info["total_episodes"] = new_idx
    merged_info["total_frames"] = total_frames
    merged_info["total_tasks"] = len(global_tasks)
    merged_info["total_videos"] = new_idx * n_video_feats
    merged_info["splits"] = {"train": f"0:{new_idx}"}
    (out / "meta" / "info.json").write_text(json.dumps(merged_info, indent=2))
    (out / "meta" / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in new_episodes) + "\n")
    (out / "meta" / "episodes_stats.jsonl").write_text("\n".join(json.dumps(r) for r in new_stats) + "\n")
    (out / "meta" / "tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": i, "task": t}) for i, t in enumerate(global_tasks)) + "\n")

    logger.info("merged %d datasets -> %d episodes / %d frames / %d tasks at %s",
                len(srcs), new_idx, total_frames, len(global_tasks), out)
    logger.info("global tasks: %s", {i: t for i, t in enumerate(global_tasks)})
    return {"episodes": new_idx, "frames": total_frames, "tasks": global_tasks, "out": str(out)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    p = argparse.ArgumentParser(description="Merge LeRobot datasets into one")
    p.add_argument("--src", action="append", required=True, dest="srcs", help="input dataset root (repeatable)")
    p.add_argument("--out", required=True)
    p.add_argument("--videos", choices=["symlink", "hardlink", "copy"], default="symlink")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    merge_datasets([Path(s) for s in args.srcs], Path(args.out), videos=args.videos, force=args.force)


if __name__ == "__main__":
    main()
