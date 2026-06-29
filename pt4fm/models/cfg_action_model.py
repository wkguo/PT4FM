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

"""PT4FM policy model: AWR-weighted, CFG-conditioned flow loss + SFT anchor.

``PT4FMCfgActionModel`` is a thin subclass of RLinf's
``OpenPi0ForCFGActionPrediction`` that overrides **only** ``forward`` (plus a
grad-carrying per-sample flow-loss helper). Everything else — model
construction, weight loading, transforms, inference-time classifier-free
guidance — is inherited unchanged. The unified training objective is::

    L = L_RL + lambda_sft * L_SFT

    L_RL  = E_i[ w_awr(A_i) * || v_theta(x_t, t | o_i, l_i^cfg) - u_t ||^2 ]
    L_SFT = E_{j in demo}[ || v_theta(x_t, t | o_j, l_j^raw) - u_t ||^2 ]

With AWR disabled (``w_awr == 1``) and ``lambda_sft == 1`` in
``reuse_unconditional`` mode, the objective collapses **exactly** to RECAP's
plain mean flow loss, so PT4FM is a strict superset of RECAP.

The per-sample advantage payload is packed into ``data["advantage"]`` as a
``[B, 3]`` float tensor ``[advantage_bool, advantage_weight, is_demo]`` by
:class:`pt4fm.data.cfg_dataset.PT4FMCFGDataLoaderImpl`, so the RLinf CFG worker's
``run_training`` (which only forwards an opaque ``advantage`` tensor) needs no
changes. A plain ``[B]`` advantage (legacy/SFT path) is still accepted.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
    OpenPi0ForCFGActionPrediction,
    Observation,
    compute_cfg_routing_masks,
)

from pt4fm.losses import awr_weight, masked_mean

_VALID_SFT_MODES = ("reuse_unconditional", "separate_forward")


def _cfg_get(cfg_obj: Any, key: str, default: Any) -> Any:
    """Read ``key`` from a (possibly OmegaConf/dict/None) config blob."""
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        val = cfg_obj.get(key, default)
    else:
        val = getattr(cfg_obj, key, default)
    return default if val is None else val


class PT4FMCfgActionModel(OpenPi0ForCFGActionPrediction):
    """RECAP CFG policy + AWR weighting + SFT auxiliary anchor."""

    # -- config accessors (knobs are injected into self.config.__dict__ by the
    # -- RLinf get_model override loop; all default to RECAP behaviour) --------
    @property
    def _awr_cfg(self):
        return getattr(self.config, "awr", None)

    @property
    def _sft_cfg(self):
        return getattr(self.config, "sft_aux", None)

    # ------------------------------------------------------------------ #
    # grad-carrying per-sample flow loss (parent's `_compute_flow_losses`
    # detaches the per-sample term, which we cannot weight). This mirrors it
    # but returns the per-sample MSE *with* gradient.
    # ------------------------------------------------------------------ #
    def _per_sample_flow_loss(
        self,
        images,
        img_masks,
        state,
        actions,
        lang_tokens,
        lang_masks,
        device,
        time,
        noise,
    ) -> torch.Tensor:
        images = [img.to(device) for img in images]
        img_masks = [m.to(device) for m in img_masks]
        state = state.to(device)
        actions = actions.to(device, dtype=torch.float32)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time
        )
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)

        per_element_loss = F.mse_loss(u_t, v_t, reduction="none")
        # mean over (action_horizon, action_dim); keeps gradient.
        return per_element_loss.mean(dim=(-1, -2))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _unpack_advantage(adv: torch.Tensor):
        """Split the packed advantage payload into (bool, weight, is_demo)."""
        adv = adv.to(torch.float32)
        if adv.dim() == 2 and adv.shape[-1] == 3:
            advantage_bool = adv[:, 0] > 0.5
            advantage_weight = adv[:, 1]
            is_demo = adv[:, 2] > 0.5
        else:
            advantage_bool = adv.view(-1) > 0.5
            advantage_weight = torch.ones_like(advantage_bool, dtype=torch.float32)
            # Legacy/plain path: treat everything as a demo for the SFT anchor.
            is_demo = torch.ones_like(advantage_bool)
        return advantage_bool, advantage_weight, is_demo

    def forward(self, data: dict[str, torch.Tensor], **kwargs):
        """Unified AWR-CFG + SFT-aux training forward. Returns (loss, metrics)."""
        is_sft_mode = "observation" in data and "actions" in data
        if is_sft_mode:
            observation = data["observation"]
            actions = data["actions"]
            device = actions.device
        else:
            device = data["dones"].device
            observation = self.input_transform(data, transpose=False)
            observation = Observation.from_dict(observation)
            actions = data["raw_actions"]

        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            pos_lang_tokens,
            pos_lang_masks,
            neg_lang_tokens,
            neg_lang_masks,
            state,
        ) = self._preprocess_observation(observation, train=True)

        adv_raw = data.get("advantage", data.get("advantages"))
        if adv_raw is None:
            raise ValueError(
                "Missing 'advantage' in data. Run compute_advantages.py "
                "(PT4FM stage 3) to generate advantages_{tag}.parquet."
            )
        advantage_bool, advantage_weight, is_demo = self._unpack_advantage(
            adv_raw.to(device)
        )

        routing = compute_cfg_routing_masks(
            advantage_bool,
            positive_only_conditional=self.config.positive_only_conditional,
            unconditional_prob=self.config.unconditional_prob,
        )
        conditional_mask = routing["conditional_mask"]
        unconditional_mask = ~conditional_mask

        # --- build the CFG-routed language tokens (identical to RECAP) --------
        if self.config.positive_only_conditional:
            sel = routing["positive_conditional_mask"].unsqueeze(-1)
            cfg_lang_tokens = torch.where(sel, pos_lang_tokens, lang_tokens)
            cfg_lang_masks = torch.where(sel, pos_lang_masks, lang_masks)
        else:
            pos = routing["positive_mask"].unsqueeze(-1)
            guidance_lang_tokens = torch.where(pos, pos_lang_tokens, neg_lang_tokens)
            guidance_lang_masks = torch.where(pos, pos_lang_masks, neg_lang_masks)
            cond = conditional_mask.unsqueeze(-1)
            cfg_lang_tokens = torch.where(cond, guidance_lang_tokens, lang_tokens)
            cfg_lang_masks = torch.where(cond, guidance_lang_masks, lang_masks)

        actions = actions.to(device, dtype=torch.float32)
        time = kwargs.get("time")
        if time is None:
            time = self.sample_time(actions.shape[0], device)
        noise = kwargs.get("noise")
        if noise is None:
            noise = self.sample_noise(actions.shape, device)

        # --- read PT4FM knobs (default => RECAP) ------------------------------
        awr_enabled = bool(_cfg_get(self._awr_cfg, "enabled", False))
        awr_beta = float(_cfg_get(self._awr_cfg, "beta", 1.0))
        awr_wmax = float(_cfg_get(self._awr_cfg, "w_max", 20.0))
        sft_weight = float(_cfg_get(self._sft_cfg, "weight", 1.0))
        sft_mode = str(_cfg_get(self._sft_cfg, "mode", "reuse_unconditional"))
        sft_demo_only = bool(_cfg_get(self._sft_cfg, "demo_only", True))
        if sft_mode not in _VALID_SFT_MODES:
            raise ValueError(f"sft_aux.mode must be one of {_VALID_SFT_MODES}, got {sft_mode}")

        n_total = float(advantage_bool.numel())

        if sft_mode == "reuse_unconditional":
            # Decompose RECAP's single flow forward into RL (conditional, AWR-
            # weighted) + SFT anchor (unconditional, raw prompt). Zero extra cost.
            per_sample = self._per_sample_flow_loss(
                images, img_masks, state, actions,
                cfg_lang_tokens, cfg_lang_masks, device, time, noise,
            )
            rl_mask = conditional_mask
            w = (
                awr_weight(advantage_weight, awr_beta, awr_wmax, mask=rl_mask, normalize=True)
                if awr_enabled
                else rl_mask.float()
            )
            rl_term = (w * per_sample).sum() / max(n_total, 1.0)
            sft_term = (unconditional_mask.float() * per_sample).sum() / max(n_total, 1.0)
            total = rl_term + sft_weight * sft_term
            ps = per_sample.detach()
            extra = {
                "pt4fm/awr_weight_mean": float(w[rl_mask].mean().item()) if rl_mask.any() else 0.0,
                "pt4fm/awr_weight_max": float(w.max().item()),
                "conditional_loss_sum": float((conditional_mask.float() * ps).sum().item()),
                "unconditional_loss_sum": float((unconditional_mask.float() * ps).sum().item()),
            }
        else:  # separate_forward
            # RL term: AWR-weighted CFG flow over ALL samples (RECAP routing).
            per_sample_cfg = self._per_sample_flow_loss(
                images, img_masks, state, actions,
                cfg_lang_tokens, cfg_lang_masks, device, time, noise,
            )
            w = (
                awr_weight(advantage_weight, awr_beta, awr_wmax, normalize=True)
                if awr_enabled
                else torch.ones_like(advantage_weight)
            )
            rl_term = (w * per_sample_cfg).sum() / max(n_total, 1.0)
            # SFT term: independent raw-prompt BC over demo samples (2nd forward).
            sft_term = torch.zeros((), device=device)
            if sft_weight > 0.0:
                per_sample_raw = self._per_sample_flow_loss(
                    images, img_masks, state, actions,
                    lang_tokens, lang_masks, device, time, noise,
                )
                anchor_mask = is_demo if sft_demo_only else torch.ones_like(is_demo)
                sft_term = masked_mean(per_sample_raw, anchor_mask)
            total = rl_term + sft_weight * sft_term
            ps = per_sample_cfg.detach()
            extra = {
                "pt4fm/awr_weight_mean": float(w.mean().item()),
                "pt4fm/awr_weight_max": float(w.max().item()),
                "conditional_loss_sum": float((conditional_mask.float() * ps).sum().item()),
                "unconditional_loss_sum": float((unconditional_mask.float() * ps).sum().item()),
            }

        metrics = self._build_metrics(routing, total, rl_term, sft_term, sft_weight, extra)
        return total, metrics

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_metrics(routing, total, rl_term, sft_term, sft_weight, extra) -> dict:
        cond = routing["conditional_mask"]
        m = {
            # Keep RECAP's count keys so the worker's reduce logic still fires.
            "conditional_count": int(cond.sum().item()),
            "unconditional_count": int((~cond).sum().item()),
            "positive_label_count": int(routing["positive_mask"].sum().item()),
            "negative_label_count": int(routing["negative_mask"].sum().item()),
            "positive_conditional_count": int(routing["positive_conditional_mask"].sum().item()),
            "positive_unconditional_count": int(routing["positive_unconditional_mask"].sum().item()),
            "negative_conditional_count": int(routing["negative_conditional_mask"].sum().item()),
            "negative_unconditional_count": int(routing["negative_unconditional_mask"].sum().item()),
            # PT4FM diagnostics.
            "pt4fm/rl_loss": float(rl_term.detach().item()),
            "pt4fm/sft_loss": float(sft_term.detach().item()),
            "pt4fm/sft_weight": float(sft_weight),
            "pt4fm/total_loss": float(total.detach().item()),
        }
        m.update(extra)
        return m


def get_model(cfg, torch_dtype=None):
    """Build a :class:`PT4FMCfgActionModel`, reusing RLinf's loader unchanged.

    RLinf's ``openpi_cfg.get_model`` imports ``OpenPi0ForCFGActionPrediction``
    from its module *inside* the function body, so temporarily rebinding that
    name to our subclass makes the loader construct a PT4FM model — while we
    inherit all of its (substantial) download / weight-loading / transform
    wiring without copying it.
    """
    import rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model as cfg_mod
    from rlinf.models.embodiment.openpi_cfg import get_model as rlinf_get_model

    # Make PT4FM-provided embodiment configs (e.g. pi05_industrial_arm) resolvable.
    from pt4fm.integrations import register_all

    register_all()

    original = cfg_mod.OpenPi0ForCFGActionPrediction
    cfg_mod.OpenPi0ForCFGActionPrediction = PT4FMCfgActionModel
    try:
        return rlinf_get_model(cfg, torch_dtype)
    finally:
        cfg_mod.OpenPi0ForCFGActionPrediction = original
