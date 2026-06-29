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

"""PT4FM stage-3 post-processor: Cal-QL calibration + AWR weight + is_demo.

This runs **after** RLinf's RECAP ``compute_advantages.py`` (which performs the
expensive distributed value inference and writes
``meta/advantages_{tag}.parquet``). It reuses that output verbatim and augments
it with the PT4FM learning signal, writing back into the same parquet (or a new
``out_tag``):

* ``is_demo``          — True for ``sft`` datasets, False for ``rollout``.
* ``advantage_continuous`` — optionally **Cal-QL-calibrated** advantage
  ``A' = A + gamma^N * (max(V_next, G_next) - V_next)`` (lower-bounds the
  bootstrap by the realized Monte-Carlo return-to-go ``G_{t+N}``).
* ``advantage``        — boolean label, re-thresholded across all rollout
  datasets at the unified ``positive_quantile`` (sft stays all-True, as RECAP).
* ``advantage_weight`` — continuous AWR weight ``exp((A'-Amax)/beta)`` (globally
  mean-1 normalized; re-normalized per micro-batch at train time).

The DataFrame-level helpers are import-light (numpy/pandas only) and unit-tested.

Example:
    python -m pt4fm.process.postprocess_advantages \
        --dataset /data/libero_sft:sft --dataset /data/libero_rollout:rollout \
        --tag fail300_N10_q30 --returns-tag fail300 \
        --awr-beta 0.7 --awr-wmax 20 --calql --lookahead 10 --gamma 1.0 \
        --positive-quantile 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pt4fm.losses import advantage_to_weight_np, calql_calibrate

logger = logging.getLogger("pt4fm.postprocess_advantages")


# --------------------------------------------------------------------------- #
# DataFrame-level core (unit-tested)
# --------------------------------------------------------------------------- #
def normalize_return(g: np.ndarray, ret_min: float, ret_max: float) -> np.ndarray:
    """Map raw return-to-go ``[ret_min, ret_max] -> [-1, 0]`` (RECAP convention)."""
    rng = ret_max - ret_min
    if rng <= 0:
        return np.full_like(np.asarray(g, dtype=np.float64), -0.5)
    return (np.asarray(g, dtype=np.float64) - ret_min) / rng - 1.0


def compute_lookahead_return(
    df: pd.DataFrame, lookahead: int
) -> np.ndarray:
    """``G_{t+N}`` (raw) by shifting the per-frame ``return`` within each episode.

    Where the look-ahead falls past the episode end (no row), returns ``NaN``.
    """
    lut = {
        (int(e), int(f)): float(r)
        for e, f, r in zip(df["episode_index"], df["frame_index"], df["return"])
    }
    out = np.full(len(df), np.nan, dtype=np.float64)
    eps = df["episode_index"].to_numpy()
    frs = df["frame_index"].to_numpy()
    for i in range(len(df)):
        out[i] = lut.get((int(eps[i]), int(frs[i]) + lookahead), np.nan)
    return out


def calql_calibrate_df(
    df: pd.DataFrame,
    lookahead: int,
    gamma: float,
    ret_min: float,
    ret_max: float,
) -> np.ndarray:
    """Return a Cal-QL-calibrated ``advantage_continuous`` array for ``df``.

    Requires columns ``advantage_continuous``, ``value_next``, ``return``.
    Rows without a valid ``G_{t+N}`` keep their original advantage.
    """
    a = df["advantage_continuous"].to_numpy(dtype=np.float64)
    v_next = df["value_next"].to_numpy(dtype=np.float64)
    g_next_raw = compute_lookahead_return(df, lookahead)
    g_next = normalize_return(g_next_raw, ret_min, ret_max)

    valid = ~np.isnan(g_next_raw)
    v_next_cal = v_next.copy()
    v_next_cal[valid] = calql_calibrate(v_next[valid], g_next[valid], enabled=True)

    gamma_n = float(gamma) ** int(lookahead)
    return a + gamma_n * (v_next_cal - v_next)


def assign_pt4fm_columns(
    df: pd.DataFrame,
    *,
    dataset_type: str,
    advantage_used: np.ndarray,
    bool_threshold: Optional[float],
    awr_beta: float,
    awr_wmax: float,
    weight_norm_mean: Optional[float] = None,
) -> pd.DataFrame:
    """Attach ``is_demo``, ``advantage``(bool), ``advantage_continuous``,
    ``advantage_weight`` to a copy of ``df``.

    ``advantage_used`` is the (possibly calibrated) continuous advantage.
    ``bool_threshold`` re-labels rollout data; ``None`` or sft keeps all-True.
    ``weight_norm_mean`` (if given) is the global mean used to normalize weights
    so magnitudes are comparable across datasets.
    """
    out = df.copy()
    is_demo = (dataset_type or "").lower() == "sft"
    out["is_demo"] = is_demo
    out["advantage_continuous"] = advantage_used

    if is_demo or bool_threshold is None:
        out["advantage"] = True
    else:
        out["advantage"] = advantage_used >= bool_threshold

    w = advantage_to_weight_np(advantage_used, beta=awr_beta, w_max=awr_wmax, normalize=False)
    if weight_norm_mean is not None and weight_norm_mean > 1e-8:
        w = w / weight_norm_mean
    out["advantage_weight"] = w
    return out


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _adv_path(dataset: Path, tag: Optional[str]) -> Path:
    name = f"advantages_{tag}.parquet" if tag else "advantages.parquet"
    return dataset / "meta" / name


def _read_return_range(dataset: Path) -> tuple[float, float]:
    stats_path = dataset / "meta" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found; needed for Cal-QL return normalization. "
            f"Run stage-1 (compute_returns) first."
        )
    stats = json.loads(stats_path.read_text())
    rs = stats["return"]

    def _scalar(x):
        return float(x[0] if isinstance(x, (list, tuple)) else x)

    return _scalar(rs["min"]), _scalar(rs["max"])


def run(
    datasets: list[tuple[str, str]],
    tag: Optional[str],
    out_tag: Optional[str],
    awr_beta: float,
    awr_wmax: float,
    calql: bool,
    lookahead: int,
    gamma: float,
    positive_quantile: float,
) -> None:
    """End-to-end stage-3 post-process across one or more datasets."""
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    loaded = []  # (path, type, df, advantage_used)
    all_adv = []
    for path_str, ds_type in datasets:
        dataset = Path(path_str)
        adv_path = _adv_path(dataset, tag)
        if not adv_path.exists():
            raise FileNotFoundError(
                f"{adv_path} not found. Run RECAP compute_advantages.py "
                f"(stage 3a) before this post-processor."
            )
        df = pd.read_parquet(adv_path)

        adv_used = df["advantage_continuous"].to_numpy(dtype=np.float64)
        if calql:
            ret_min, ret_max = _read_return_range(dataset)
            adv_used = calql_calibrate_df(df, lookahead, gamma, ret_min, ret_max)
            logger.info(
                "%s: Cal-QL calibration applied (N=%d, gamma=%.3f, "
                "return_range=[%.3f, %.3f])",
                dataset.name, lookahead, gamma, ret_min, ret_max,
            )
        loaded.append((dataset, ds_type, df, adv_used))
        # Only rollout datasets contribute to the unified bool threshold.
        if (ds_type or "").lower() != "sft":
            all_adv.append(adv_used)

    # Unified positive threshold across rollout datasets (RECAP convention).
    threshold = None
    if all_adv:
        combined = np.concatenate(all_adv)
        threshold = float(np.percentile(combined, (1.0 - positive_quantile) * 100.0))
        logger.info(
            "Unified threshold @ q=%.2f -> %.4f (rollout samples=%d, positives=%d)",
            positive_quantile, threshold, len(combined),
            int((combined >= threshold).sum()),
        )

    # Global weight normalization (mean over the raw exp-weights of all data).
    raw_w = np.concatenate(
        [advantage_to_weight_np(a, awr_beta, awr_wmax, normalize=False) for _, _, _, a in loaded]
    )
    weight_norm_mean = float(raw_w.mean()) if raw_w.size else 1.0

    for dataset, ds_type, df, adv_used in loaded:
        out = assign_pt4fm_columns(
            df,
            dataset_type=ds_type,
            advantage_used=adv_used,
            bool_threshold=threshold,
            awr_beta=awr_beta,
            awr_wmax=awr_wmax,
            weight_norm_mean=weight_norm_mean,
        )
        write_tag = out_tag if out_tag is not None else tag
        out_path = _adv_path(dataset, write_tag)
        out.to_parquet(out_path, index=False)
        logger.info(
            "%s: wrote %s  (n=%d, positives=%.1f%%, weight[min/mean/max]=%.2f/%.2f/%.2f)",
            dataset.name, out_path.name, len(out),
            100.0 * out["advantage"].mean(),
            out["advantage_weight"].min(), out["advantage_weight"].mean(),
            out["advantage_weight"].max(),
        )


def _parse_dataset(spec: str) -> tuple[str, str]:
    """Parse ``path[:type]`` (default type ``sft``)."""
    if ":" in spec and not spec.startswith(("http://", "https://")):
        path, ds_type = spec.rsplit(":", 1)
        return path, ds_type
    return spec, "sft"


def main() -> None:
    p = argparse.ArgumentParser(description="PT4FM stage-3 advantage post-process")
    p.add_argument("--dataset", action="append", required=True, dest="datasets",
                   help="path[:type] (type = sft|rollout). Repeatable.")
    p.add_argument("--tag", default=None, help="input advantages tag")
    p.add_argument("--out-tag", default=None, help="output tag (default: overwrite --tag)")
    p.add_argument("--awr-beta", type=float, default=0.7)
    p.add_argument("--awr-wmax", type=float, default=20.0)
    p.add_argument("--calql", action="store_true", help="enable Cal-QL calibration")
    p.add_argument("--lookahead", type=int, default=10, help="N for G_{t+N} (= advantage_lookahead_step)")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--positive-quantile", type=float, default=0.3)
    args = p.parse_args()

    datasets = [_parse_dataset(s) for s in args.datasets]
    run(
        datasets=datasets,
        tag=args.tag,
        out_tag=args.out_tag,
        awr_beta=args.awr_beta,
        awr_wmax=args.awr_wmax,
        calql=args.calql,
        lookahead=args.lookahead,
        gamma=args.gamma,
        positive_quantile=args.positive_quantile,
    )


if __name__ == "__main__":
    main()
