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

"""Unit tests for the sft2rl converter (numpy+pandas+pyarrow only)."""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pt4fm.data.sft2rl import convert_dataset


def _build_dataset(root, n_ep=3, ep_len=5):
    (root / "meta").mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "Test_Arm",
        "total_episodes": n_ep,
        "total_frames": n_ep * ep_len,
        "total_chunks": 1,
        "chunks_size": n_ep,  # all episodes in chunk-000
        "fps": 15.0,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [7], "names": None},
            "observation.state": {"dtype": "float32", "shape": [13], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    es_lines, ep_lines = [], []
    for ep in range(n_ep):
        d = root / "data" / "chunk-000"
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "frame_index": np.arange(ep_len),
            "episode_index": np.full(ep_len, ep),
            "action": list(np.random.randn(ep_len, 7).astype(np.float32)),
            "observation.state": list(np.random.randn(ep_len, 13).astype(np.float32)),
        })
        df.to_parquet(d / f"episode_{ep:06d}.parquet", index=False)
        es_lines.append(json.dumps({"episode_index": ep, "stats": {}}))
        ep_lines.append(json.dumps({"episode_index": ep, "length": ep_len,
                                    "source_demo_id": f"demo_{ep}"}))
    (root / "meta" / "episodes_stats.jsonl").write_text("\n".join(es_lines) + "\n")
    (root / "meta" / "episodes.jsonl").write_text("\n".join(ep_lines) + "\n")


def test_sft2rl_default_success(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _build_dataset(src, n_ep=3, ep_len=5)
    convert_dataset(src, dst, default_success=True, add_done=True, add_reward=True,
                    videos="skip", num_workers=1)
    for ep in range(3):
        df = pq.read_table(dst / "data" / "chunk-000" / f"episode_{ep:06d}.parquet").to_pandas()
        assert {"is_success", "done", "reward"} <= set(df.columns)
        assert (df["is_success"] == 1).all()
        assert df["done"].iloc[-1] == 1 and df["done"].iloc[:-1].sum() == 0
        assert df["reward"].iloc[-1] == 0.0 and (df["reward"].iloc[:-1] == -1).all()
    info = json.loads((dst / "meta" / "info.json").read_text())
    assert {"is_success", "done", "reward"} <= set(info["features"].keys())
    es = [json.loads(line) for line in (dst / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    assert all("is_success" in r["stats"] for r in es)


def test_sft2rl_success_map_and_failure_reward(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _build_dataset(src, n_ep=3, ep_len=4)
    smap = tmp_path / "succ.json"
    smap.write_text(json.dumps({"0": 1, "1": 0, "2": 1}))
    convert_dataset(src, dst, success_map_path=smap, add_reward=True,
                    failure_reward=-300.0, videos="skip", num_workers=1)
    # episode 1 is a failure -> terminal reward = -300, is_success = 0
    df1 = pq.read_table(dst / "data" / "chunk-000" / "episode_000001.parquet").to_pandas()
    assert (df1["is_success"] == 0).all()
    assert df1["reward"].iloc[-1] == -300.0
    df0 = pq.read_table(dst / "data" / "chunk-000" / "episode_000000.parquet").to_pandas()
    assert (df0["is_success"] == 1).all() and df0["reward"].iloc[-1] == 0.0


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
