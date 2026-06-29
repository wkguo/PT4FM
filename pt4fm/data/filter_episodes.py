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

"""filter_episodes — drop episodes from a LeRobot v2.1 dataset and renumber.

Use cases: remove over-long outlier episodes (default: length > ``--max-len-mult``
x the dataset median), explicit indices (``--drop-indices``), or an absolute cap
(``--max-len``). The result is a valid, **contiguously re-indexed** LeRobot dataset
(``episode_index`` 0..M-1, global ``index`` recomputed), with videos re-linked.

Per-dataset median is the right reference for multi-task corpora (e.g. a RAM-install
task is legitimately ~2x longer than a disk-insert task), so the multiplier is
applied to each dataset's own median, not a global one.

In-place (``--in-place``) writes to a temp dir then atomically swaps. Otherwise it
writes a fresh dataset at ``--dst``. The source's videos may be real files or
symlinks; both are followed to their real target when re-linking.

Example (drop > 3x median, in place):
    python -m pt4fm.data.filter_episodes --src /data/ds --in-place --max-len-mult 3.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

from pt4fm.data.sft2rl import (
    episode_parquet_relpath,
    load_episodes,
    load_info,
    video_relpaths,
)

logger = logging.getLogger("pt4fm.filter_episodes")


def select_kept_episodes(
    episodes: list[dict],
    max_len_mult: Optional[float],
    max_len: Optional[int],
    drop_indices: Optional[set[int]],
) -> tuple[list[int], list[int]]:
    """Return (kept_old_indices_sorted, dropped_old_indices_sorted)."""
    lengths = np.array([int(e["length"]) for e in episodes], dtype=np.float64)
    idxs = [int(e["episode_index"]) for e in episodes]
    threshold = None
    if max_len is not None:
        threshold = float(max_len)
    elif max_len_mult is not None:
        threshold = float(max_len_mult) * float(np.median(lengths))
    drop = set(drop_indices or set())
    kept, dropped = [], []
    for ep, L in zip(idxs, lengths):
        if ep in drop or (threshold is not None and L > threshold):
            dropped.append(ep)
        else:
            kept.append(ep)
    return sorted(kept), sorted(dropped)


def _resolve_link_target(p: Path) -> Path:
    return Path(os.path.realpath(p))


def filter_dataset(
    src: Path,
    dst: Optional[Path] = None,
    *,
    max_len_mult: Optional[float] = 3.0,
    max_len: Optional[int] = None,
    drop_indices: Optional[set[int]] = None,
    videos: str = "symlink",
    in_place: bool = False,
    force: bool = False,
) -> dict:
    src = Path(src)
    info = load_info(src)
    episodes = load_episodes(src)
    ep_by_idx = {int(e["episode_index"]): e for e in episodes}

    kept, dropped = select_kept_episodes(episodes, max_len_mult, max_len, drop_indices)
    logger.info("%s: keeping %d, dropping %d episodes %s",
                src.name, len(kept), len(dropped), dropped if len(dropped) <= 20 else f"({len(dropped)})")
    if not dropped:
        logger.info("%s: nothing to drop.", src.name)

    if in_place:
        work = src.parent / (src.name + ".tmp_filter")
    else:
        if dst is None:
            raise ValueError("Provide --dst or --in-place")
        work = Path(dst)
    if work.exists():
        if not force and not in_place:
            raise FileExistsError(f"{work} exists; pass --force.")
        shutil.rmtree(work)
    (work / "meta").mkdir(parents=True, exist_ok=True)

    # Rewrite kept parquets with contiguous episode_index + global index, re-link videos.
    new_episodes, new_stats = [], []
    src_stats = {}
    es_path = src / "meta" / "episodes_stats.jsonl"
    if es_path.exists():
        for line in es_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                src_stats[int(r["episode_index"])] = r.get("stats", {})

    running_index = 0
    total_frames = 0
    for new_idx, old_ep in enumerate(kept):
        old_rel = episode_parquet_relpath(info, old_ep)
        new_rel = episode_parquet_relpath(info, new_idx)
        df = pq.read_table(src / old_rel).to_pandas()
        n = len(df)
        df["episode_index"] = np.full(n, new_idx, dtype=df["episode_index"].dtype)
        if "index" in df.columns:
            df["index"] = np.arange(running_index, running_index + n, dtype=df["index"].dtype)
        (work / new_rel).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(work / new_rel, index=False)
        running_index += n
        total_frames += n

        # videos: map old episode video -> new episode name, following links.
        for old_vrel, new_vrel in zip(video_relpaths(info, old_ep), video_relpaths(info, new_idx)):
            s = _resolve_link_target(src / old_vrel)
            d = work / new_vrel
            if not s.exists():
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            if videos == "symlink":
                d.symlink_to(s)
            elif videos == "hardlink":
                try:
                    os.link(s, d)
                except OSError:
                    shutil.copy2(s, d)
            elif videos == "copy":
                shutil.copy2(s, d)

        # meta records, renumbered
        em = dict(ep_by_idx[old_ep]); em["episode_index"] = new_idx
        new_episodes.append(em)
        new_stats.append({"episode_index": new_idx, "stats": src_stats.get(old_ep, {})})

    # meta files
    new_info = json.loads(json.dumps(info))
    new_info["total_episodes"] = len(kept)
    new_info["total_frames"] = total_frames
    new_info["splits"] = {"train": f"0:{len(kept)}"}
    n_video_feats = sum(1 for f in info.get("features", {}).values() if f.get("dtype") == "video")
    new_info["total_videos"] = len(kept) * n_video_feats
    (work / "meta" / "info.json").write_text(json.dumps(new_info, indent=2))
    (work / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e) for e in new_episodes) + "\n")
    (work / "meta" / "episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(r) for r in new_stats) + "\n")
    for name in ("tasks.jsonl", "stats.json", "processed_demos.json"):
        sp = src / "meta" / name
        if sp.exists():
            shutil.copy2(sp, work / "meta" / name)

    if in_place:
        backup = src.parent / (src.name + ".bak_filter")
        if backup.exists():
            shutil.rmtree(backup)
        src.rename(backup)
        work.rename(src)
        shutil.rmtree(backup)
        out = src
    else:
        out = work
    logger.info("%s: wrote %d episodes / %d frames -> %s", src.name, len(kept), total_frames, out)
    return {"kept": len(kept), "dropped": dropped, "frames": total_frames, "out": str(out)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    p = argparse.ArgumentParser(description="Filter (drop) episodes from a LeRobot dataset and renumber")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", default=None, help="output dir (omit with --in-place)")
    p.add_argument("--in-place", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--max-len-mult", type=float, default=3.0, help="drop length > mult x dataset median")
    g.add_argument("--max-len", type=int, default=None, help="drop length > this absolute value")
    p.add_argument("--drop-indices", type=int, nargs="*", default=None, help="explicit episode indices to drop")
    p.add_argument("--videos", choices=["symlink", "hardlink", "copy"], default="symlink")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    filter_dataset(
        src=Path(args.src),
        dst=Path(args.dst) if args.dst else None,
        max_len_mult=(None if args.max_len is not None else args.max_len_mult),
        max_len=args.max_len,
        drop_indices=set(args.drop_indices) if args.drop_indices else None,
        videos=args.videos,
        in_place=args.in_place,
        force=args.force,
    )


if __name__ == "__main__":
    main()
