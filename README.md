# PT4FM — Post-Training for Foundation-model VLAs (Offline RL + SFT)

PT4FM is a post-training framework for **pi0.5-class flow-matching VLAs** that combines
**offline RL with SFT**, driven entirely by **LeRobot-format datasets** (no online
rollouts required at train time). It is built around the **RECAP** recipe
(advantage-conditioned policy extraction) and adds optional, config-gated enhancements.

The unified policy objective (see [docs/method.md](docs/method.md)):

```
L(theta) = L_RL(theta) + lambda_sft * L_SFT(theta)

L_RL  = E_i[ w_awr(A_i) * || v_theta(x_t, t | o_i, l_i^cfg) - u_t ||^2 ]   # CFG-conditioned + AWR-weighted
L_SFT = E_{j in demo}[ || v_theta(x_t, t | o_j, l_j^raw) - u_t ||^2 ]       # raw-prompt BC anchor
```

> Design rule: every enhancement is **config-gated and defaults to RECAP behavior**.
> With `cfg.enabled=true`, `awr.enabled=false`, `sft_aux.mode=reuse_unconditional`,
> `sft_aux.weight=1.0` (and Cal-QL / expectile off) the method is a faithful RECAP
> reproduction. AWR / Cal-QL / IQL-expectile are additive options.

Inference is unchanged RECAP classifier-free guidance:
`v = v_uncond + s * (v_cond - v_uncond)` with `s = cfgrl_guidance_scale`.

## Repository layout (three repos)

PT4FM is the umbrella repo; the actual trainers live alongside it:

| Repo | Role |
|---|---|
| **PT4FM** (this) | data tools (`sft2rl` / `filter_episodes` / `merge_datasets`), the `lerobot.common -> lerobot` compat shim, PyTorch RECAP extensions, the Industrial_Arm 3-cam transform, the RLinf patch, docs, `requirements.txt`, `SETUP.md` |
| **openpi** (fork, branch `pt4fm-offline-rl`) | JAX policy trainer (stage 4): `scripts/train_offline_rl.py` + `pine_foundry*` configs + fast norm-stats |
| **RLinf** (fork) | value model + advantages (stages 1-3, PyTorch); the `industrial_arm` robot_type is included on `main` |

```
PT4FM/
  pt4fm/
    losses.py                 # AWR / Cal-QL / expectile primitives (pure, unit-tested)
    data/                     # sft2rl, filter_episodes, merge_datasets, cfg_dataset
    models/ workers/          # PyTorch RECAP extensions (config-gated)
    integrations/             # Industrial_Arm 3-cam openpi transform + config registration
    process/                  # stage-3 advantage post-processing (Cal-QL + AWR weight)
    compat/                   # lerobot compat shim + PyAV video fallback
    shim_path/sitecustomize.py
    tests/
  configs/                    # libero_value / libero_policy presets
  docs/                       # method, framework, pipeline, custom_data, jax_migration
  patches/                    # rlinf_industrial_arm.patch
  scripts/                    # 0_sft2rl .. 4_train_policy
  requirements.txt  SETUP.md
```

## Method

Four-stage offline pipeline over LeRobot data (`sft` = successful demos, `rollout` =
success + failure):

1. **Compute returns** — discounted Monte-Carlo returns from a per-episode
   success/failure signal (`r = -1`/step, terminal `0`/`failure_reward`).
2. **Value model** — a small distributional VLM critic (SigLIP2 + Gemma3 + value head,
   201 categorical bins over `[-1, 0]`) regressed onto the normalized returns.
3. **Advantages** — `A_t = normalize(sum_k gamma^k r) + gamma^N V(o_{t+N}) - V(o_t)`,
   binarized by a quantile threshold into a positive/negative indicator.
4. **Policy training** — advantage-conditioned flow matching: the prompt is conditioned
   on the advantage indicator (`"Advantage: positive/negative"`), trained with
   classifier-free guidance; an always-on SFT flow-BC term anchors the policy to the
   demonstration manifold.

Enhancements over vanilla RECAP (all optional):
- **AWR continuous weighting** — uses the advantage magnitude, not just its sign.
- **Cal-QL calibration** — lower-bounds the bootstrap value by the realized MC
  return-to-go, stabilizing mixed success/failure data.
- **IQL expectile value** — an optimistic-within-support value for sharper advantages.

## Hybrid (PyTorch + JAX)

Stages 1-3 (returns / value / advantages) run in PyTorch (RLinf RECAP); the stage-4
policy update runs on **JAX openpi**, so it resumes a finetuned JAX pi0.5 checkpoint
directly (no JAX->PyTorch conversion) and keeps JAX numerical stability. The two halves
hand off through the `meta/advantages_{tag}.parquet` sidecar. See
[docs/jax_migration.md](docs/jax_migration.md).

## Setup & run

See **[SETUP.md](SETUP.md)** for the full migration runbook (clone the three repos,
build the environment, transfer the model/data, and run all stages). Validated on
RTX PRO 6000 (Blackwell, CUDA 12.8), Python 3.11, with `torch 2.11+cu128` /
`jax 0.7.2` / `lerobot 0.3.3`; pinned in [requirements.txt](requirements.txt).

Pure-function unit tests (no GPU):
```bash
PYTHONPATH=$PWD python -m pytest pt4fm/tests -q
```

## Knob reference (paper-faithful defaults)

| Field | Default | Meaning |
|---|---|---|
| `cfg.enabled` | `true` (RECAP) | advantage-conditioned prompt + classifier-free guidance |
| `cfg.positive_only_conditional` | `true` | only positive samples are prompt-conditioned |
| `awr.enabled` | `false` | continuous AWR weighting (extension; off = RECAP) |
| `sft_aux.mode` | `reuse_unconditional` | free SFT anchor; or `separate_forward` (2x policy forward) |
| `sft_aux.weight` | `1.0` | lambda_sft |
| stage-3 `--calql` | off | Cal-QL bootstrap calibration (extension) |
| value `expectile.enabled` | `false` | IQL expectile value (extension) |
| `cfgrl_guidance_scale` | `1.0` | inference-time CFG strength |

## Roadmap

- **QAM (Q-learning with Adjoint Matching) — expected.** A planned actor-critic
  flow-Q route for `L_RL` (adjoint matching through the flow ODE), as an alternative to
  advantage-conditioned extraction. Tracked as an extension point in
  [docs/method.md](docs/method.md); not yet implemented.

## Docs

- [SETUP.md](SETUP.md) — install, migration, and the full per-stage runbook.
- [docs/method.md](docs/method.md) — math, the RECAP/AWR/Cal-QL/IQL/FQL synthesis, and why.
- [docs/framework.md](docs/framework.md) — architecture, integration points, data flow.
- [docs/pipeline.md](docs/pipeline.md) — step-by-step pipeline, tags, troubleshooting.
- [docs/custom_data.md](docs/custom_data.md) — bringing your own data / embodiment (sft2rl, norm stats, transforms).
- [docs/jax_migration.md](docs/jax_migration.md) — the JAX stage-4 trainer and the hybrid split.

## Notes & limitations

- Meaningful offline RL needs failure-containing rollouts or rewards; on pure-success
  demonstrations every term degenerates to SFT.
- Stages 2-3 (RLinf RECAP) import the legacy `lerobot.common` API; the bundled shim
  (`pt4fm/compat`, auto-loaded via `pt4fm/shim_path`) redirects it to `lerobot.datasets`
  and patches the PyAV video path under lerobot 0.3.x.
- Data and checkpoints are not stored in git; transfer them separately (use `rsync -aL`
  to dereference symlinked videos).
