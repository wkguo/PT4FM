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

"""PT4FM CFG policy-training worker.

A minimal extension of RLinf's ``FSDPCfgWorker`` with three small overrides:

1. ``model_provider_func`` builds the PT4FM policy (AWR + SFT-aux forward).
2. ``_load_advantages_lookup`` reads the richer parquet
   (``advantage`` + ``advantage_weight`` + ``is_demo``) into the lookup.
3. ``build_dataloader`` rebinds the two RECAP data classes to the PT4FM ones
   for the duration of the (inherited) build, so the rest of the heavy data
   wiring is reused verbatim.

Nothing in ``run_training`` is touched: the extra per-sample fields ride along
inside the opaque ``[B, 3]`` ``advantage`` tensor emitted by the dataloader.
"""

from __future__ import annotations

from pathlib import Path

from pt4fm.compat import install_lerobot_common_shim

install_lerobot_common_shim()  # must precede any rlinf.recap import

from rlinf.workers.sft.fsdp_cfg_worker import FSDPCfgWorker


class PT4FMCfgWorker(FSDPCfgWorker):
    """RECAP CFG worker + PT4FM model/data plumbing."""

    def model_provider_func(self):
        from pt4fm.models import get_cfg_model

        model = get_cfg_model(self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    @staticmethod
    def _load_advantages_lookup(data_path: str, advantage_tag: str | None = None) -> dict:
        """Return ``(episode_index, frame_index) -> dict`` with PT4FM fields.

        Backwards compatible with RECAP parquets: missing ``advantage_weight``
        defaults to 1.0 and missing ``is_demo`` defaults to True.
        """
        import pandas as pd

        if advantage_tag:
            meta_path = Path(data_path) / "meta" / f"advantages_{advantage_tag}.parquet"
        else:
            meta_path = Path(data_path) / "meta" / "advantages.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Advantage file not found: {meta_path}. Run PT4FM stage-3 "
                f"(compute_advantages) first."
            )

        df = pd.read_parquet(meta_path)
        ep = df["episode_index"].values.astype(int)
        fr = df["frame_index"].values.astype(int)
        adv = df["advantage"].values.astype(bool)
        if "advantage_weight" in df.columns:
            wt = df["advantage_weight"].values.astype(float)
        else:
            wt = [1.0] * len(df)
        if "is_demo" in df.columns:
            demo = df["is_demo"].values.astype(bool)
        else:
            demo = [True] * len(df)

        return {
            (int(e), int(f)): {
                "advantage": bool(a),
                "advantage_weight": float(w),
                "is_demo": bool(d),
            }
            for e, f, a, w, d in zip(ep, fr, adv, wt, demo)
        }

    def build_dataloader(self):
        """Reuse RECAP's build_dataloader with PT4FM dataset/dataloader classes."""
        import rlinf.workers.sft.fsdp_cfg_worker as wmod

        # Ensure PT4FM embodiment configs (e.g. pi05_industrial_arm) are registered
        # before RECAP's build_dataloader calls get_openpi_config().
        from pt4fm.integrations import register_all

        register_all()

        from pt4fm.data.cfg_dataset import (
            PT4FMAdvantagePreservingDataset,
            PT4FMCFGDataLoaderImpl,
        )

        saved_adv = wmod.AdvantagePreservingDataset
        saved_impl = wmod.CFGDataLoaderImpl
        wmod.AdvantagePreservingDataset = PT4FMAdvantagePreservingDataset
        wmod.CFGDataLoaderImpl = PT4FMCFGDataLoaderImpl
        try:
            return super().build_dataloader()
        finally:
            wmod.AdvantagePreservingDataset = saved_adv
            wmod.CFGDataLoaderImpl = saved_impl
