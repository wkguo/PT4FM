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

"""Environment-compatibility shims for PT4FM.

RLinf's RECAP data layer (4 files: ``recap/utils.py``, ``recap/value_model.py``,
``workers/sft/fsdp_cfg_worker.py``, ``data/lerobot_writer.py``) imports the
*old* LeRobot API ``lerobot.common.datasets.*``. The openpi stack that PT4FM
runs on (VLA-R3L/openpi + ``lerobot>=0.3``) renamed that to ``lerobot.datasets.*``
and dropped the ``lerobot.common`` package entirely.

:func:`install_lerobot_common_shim` installs a ``sys.meta_path`` finder that
transparently redirects any ``lerobot.common[.X...]`` import to ``lerobot[.X...]``,
so RLinf's recap code imports cleanly under the newer lerobot **without editing
RLinf or downgrading lerobot**. It is a no-op when ``lerobot.common`` already
exists (older lerobot) or when lerobot is absent.

Caveat: this fixes *import* compatibility. If a future lerobot also changes
runtime signatures (e.g. ``LeRobotDataset.__init__``), those call sites may still
need attention — but as of lerobot 0.3.x the relevant symbols
(``hf_transform_to_torch``, ``LeRobotDataset``, ``LeRobotDatasetMetadata``) are
API-compatible, only relocated.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path

_OLD_ROOT = "lerobot.common"
_NEW_ROOT = "lerobot"
_installed = False


class _LerobotCommonRedirector(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Redirect ``lerobot.common[.sub]`` -> ``lerobot[.sub]`` at import time."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _OLD_ROOT and not fullname.startswith(_OLD_ROOT + "."):
            return None
        new_name = _NEW_ROOT + fullname[len(_OLD_ROOT):]
        try:
            module = importlib.import_module(new_name)
        except Exception:
            return None
        sys.modules[fullname] = module
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return sys.modules[spec.name]

    def exec_module(self, module):  # already executed via real import
        pass


def install_lerobot_common_shim() -> bool:
    """Install the redirect finder. Idempotent. Returns True if a shim is active.

    No-op (returns True) when ``lerobot.common`` is already importable, and
    no-op (returns False) when ``lerobot`` itself is not installed.
    """
    global _installed
    if _installed:
        return True
    # Old lerobot already provides lerobot.common -> nothing to do.
    if importlib.util.find_spec("lerobot") is None:
        return False
    try:
        importlib.import_module("lerobot.common")
        _installed = True
        return True
    except Exception:
        pass
    # Only install if the new layout exists.
    if importlib.util.find_spec("lerobot.datasets") is None:
        return False
    sys.meta_path.insert(0, _LerobotCommonRedirector())
    _installed = True
    return True



def install_lerobot_pyav_video_fallback() -> bool:
    """Patch LeRobot's pyav decoder when torchvision lacks VideoReader.

    LeRobot 0.3.x routes its ``pyav`` backend through
    ``torchvision.io.VideoReader``. Newer torchvision builds can omit that API,
    which breaks video datasets even though PyAV itself is installed. This keeps
    the backend on CPU and returns the same CHW float tensor format expected by
    LeRobot.
    """
    try:
        import av
        import torch
        import torchvision
        from lerobot.datasets import video_utils
    except Exception:
        return False

    if hasattr(getattr(torchvision, "io", None), "VideoReader"):
        return False

    def _frame_time_seconds(frame, stream) -> float:
        if frame.time is not None:
            return float(frame.time)
        if frame.pts is None:
            return 0.0
        return float(frame.pts * stream.time_base)

    def _decode_video_frames_pyav(
        video_path: Path | str,
        timestamps: list[float],
        tolerance_s: float,
        backend: str = "pyav",
        log_loaded_timestamps: bool = False,
    ) -> torch.Tensor:
        del backend, log_loaded_timestamps
        video_path = str(video_path)
        if not timestamps:
            return torch.empty((0, 3, 0, 0), dtype=torch.float32)

        first_ts = float(min(timestamps))
        last_ts = float(max(timestamps))
        frames = []
        frame_ts = []

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            if stream.time_base is not None:
                seek_ts = max(first_ts - 0.5, 0.0)
                offset = int(seek_ts / float(stream.time_base))
                try:
                    container.seek(offset, backward=True, any_frame=False, stream=stream)
                except Exception:
                    container.seek(0)

            for frame in container.decode(stream):
                ts = _frame_time_seconds(frame, stream)
                if ts + tolerance_s < first_ts:
                    continue
                arr = frame.to_ndarray(format="rgb24")
                frames.append(torch.from_numpy(arr).permute(2, 0, 1))
                frame_ts.append(ts)
                if ts >= last_ts:
                    break

        if not frames:
            raise RuntimeError(f"No video frames decoded from {video_path}")

        query_ts = torch.tensor(timestamps, dtype=torch.float32)
        loaded_ts = torch.tensor(frame_ts, dtype=torch.float32)
        dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
        min_dist, argmin = dist.min(1)

        is_within_tol = min_dist < tolerance_s
        assert is_within_tol.all(), (
            f"One or several query timestamps unexpectedly violate the tolerance "
            f"({min_dist[~is_within_tol]} > {tolerance_s=})."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
            "\nbackend: pyav-direct"
        )

        closest = torch.stack([frames[int(i)] for i in argmin])
        return closest.to(torch.float32) / 255.0

    video_utils.decode_video_frames_torchvision = _decode_video_frames_pyav
    return True
