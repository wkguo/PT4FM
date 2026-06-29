# PT4FM Method — AWR-CFG with Cal-QL-calibrated value and SFT anchor

This document explains *what* PT4FM optimizes, *why* the pieces are combined this
way, and *how* each reference method (RECAP, AWR/AWAC, Cal-QL, IQL, FQL/QC, QAM)
maps into the design.

## 1. Problem setting

We post-train a pretrained flow-matching VLA `π_θ` (pi0.5: PaliGemma/SigLIP prefix
+ action expert producing a velocity field `v_θ(x_t, t | o, ℓ)`), using only an
**offline** LeRobot dataset — no environment interaction. The data are of two kinds:

- **`sft`** — successful expert demonstrations (all "good"),
- **`rollout`** — trajectories from a policy, containing successes *and* failures.

The pretraining objective is conditional flow matching. For a clean action chunk
`a` (horizon `H`), noise `ε`, time `t∼U(0,1)`, `x_t = t·ε + (1−t)·a`, target
velocity `u_t = ε − a`, the BC loss is `‖v_θ(x_t,t|o,ℓ) − u_t‖²`. PT4FM keeps this
exact loss and only **re-weights / re-conditions** it — never backprops a Q-gradient
through the flow ODE — which is what makes it stable and scalable at VLA size.

## 2. Why advantage-weighted / -conditioned BC (not actor-critic flow-Q)

We studied three offline-RL families and chose against the heavy ones for v1:

- **FQL / QC (`qc/agents/acfql.py`)**: `actor_loss = bc_flow_loss + α·distill_loss
  + (−Q)`. Requires (a) a Q-critic over `(image, language, state, action-chunk)`,
  (b) distilling the multi-step flow into a *one-step* policy to backprop `Q`. On a
  billion-parameter VLA this is expensive and the one-step distillation discards the
  expressive pretrained flow. **But** its always-on `bc_flow_loss` is exactly the
  "SFT-as-auxiliary" idea — we keep that.
- **QAM (`qam`)**: replaces FQL's biased distillation with adjoint matching through
  the flow ODE — more correct, even more costly. Noted as a future extension.
- **Cal-QL (`Cal-QL/JaxCQL/conservative_sac.py`)**: CQL + calibration
  `maximum(Q_ood, mc_returns)`. We borrow the **calibration idea** (reference =
  Monte-Carlo return) without the full CQL critic.

The scalable alternative — and what Physical Intelligence's RECAP itself uses for
pi0.5 — is **advantage-weighted / advantage-conditioned behavior cloning**: it needs
only a state-value `V` (no action `Q`), no policy gradient through sampling, and
stays in the pretraining loss family. PT4FM takes RECAP as the backbone and
sharpens its signal.

## 3. The value model `V` (stage 2)

A small, decoupled critic (RECAP's `ValueCriticModel`: SigLIP2-so400m + Gemma3-270M
+ a learnable value expert → a **categorical** value distribution over `num_bins`
atoms in `[v_min, v_max] = [-1, 0]`). It regresses **normalized Monte-Carlo returns**
computed in stage 1 (`G_t = r_t + γ·G_{t+1}`, with per-step `r=-1`, terminal `0` for
success and `r_fail` for failure). Because the target is the realized MC return, the
value is an estimate of the *behavior-policy* value `V^μ` — no bootstrapping, no
over-estimation.

**PT4FM enhancement — IQL expectile (`expectile.tau>0.5`).** We optionally re-weight
the per-sample categorical loss by the IQL expectile factor `|τ − 1{target<pred}|`
(`pt4fm/losses.py:expectile_weight`). This biases `V` toward the *better-than-average*
returns passing through similar observations — an "optimistic-within-support" value,
closer to `V^*` than `V^μ` — giving a more discriminative advantage. `τ=0.5`
recovers plain regression (RECAP).

## 4. The advantage and its calibration (stage 3)

RECAP's `N`-step look-ahead advantage:

```
A_t = normalize(Σ_{k=0}^{N-1} γ^k r_{t+k}) + γ^N · V(o_{t+N}) − V(o_t)
```

then labels the top `positive_quantile` fraction (per rollout data) as `advantage=True`
(`sft` data is forced all-True, as in RECAP).

**PT4FM enhancement 1 — Cal-QL calibration (`--calql`).** The bootstrap `V(o_{t+N})`
can be over/under-optimistic on states the value net saw rarely. We lower-bound it by
the *realized* MC return-to-go from that state — a valid, on-policy reference
(`pt4fm/losses.py:calql_calibrate`, mirroring Cal-QL's `maximum(Q, mc_returns)`):

```
A_t = normalize(Σ γ^k r_{t+k}) + γ^N · max(V(o_{t+N}), G_{t+N}) − V(o_t)
```

This sharpens advantages on success-leading transitions and keeps pessimistic
(failure) bootstraps untouched — most useful with mixed `rollout` data.

**PT4FM enhancement 2 — AWR continuous weight.** RECAP throws away advantage
*magnitude* (binary positive/negative). We additionally bake an AWR/AWAC weight into
the parquet (`pt4fm/losses.py:advantage_to_weight_np`):

```
w_awr(A) = clip( exp((A − max A)/β), 0, w_max )
```

`β→∞` (or `awr.enabled=false`) ⇒ uniform weights ⇒ RECAP. `β` finite ⇒ high-advantage
transitions contribute proportionally more.

## 5. The unified policy objective (stage 4)

For each sampled chunk we compute the per-sample flow loss `ℓ_i = ‖v_θ − u_t‖²`
(grad-carrying; `pt4fm/models/cfg_action_model.py:_per_sample_flow_loss`). CFG routing
(`compute_cfg_routing_masks`) decides, per sample, whether the prompt is the
advantage-conditioned guidance prompt (`ℓ^cfg`, "…\nAdvantage: positive") or the raw
prompt (`ℓ^raw`), with `unconditional_prob` dropout — identical to RECAP. Then:

```
L = L_RL + λ_sft · L_SFT
```

**`reuse_unconditional` (default, zero extra forward).** Decompose RECAP's single
flow forward: conditional samples are the RL term (AWR-weighted), unconditional
samples are the SFT anchor.

```
L_RL  = ( Σ_i  cond_i · w_i · ℓ_i ) / N        # w_i: AWR, mean-1 over conditional set
L_SFT = ( Σ_i  uncond_i · ℓ_i )    / N
```

With `w_i=1` and `λ_sft=1` this is `(Σ_i ℓ_i)/N` = RECAP's mean flow loss **exactly**.

**`separate_forward` (faithful SFT anchor, 2× policy forward).** The RL term is the
AWR-weighted CFG loss over *all* samples (RECAP routing); the SFT term is an
*independent* raw-prompt BC over the demo samples:

```
L_RL  = ( Σ_i w_i · ℓ_i^cfg ) / N
L_SFT = mean_{j: is_demo} ℓ_j^raw            # second forward, raw prompt
```

Set `λ_sft=0` ⇒ exactly RECAP. Use this when you want a true SFT loss over **all**
demonstrations (not just the CFG-unconditional subset) as the regularizer.

### Why the SFT auxiliary matters
Advantage-conditioned BC can drift toward narrow, high-advantage modes or exploit
value-model errors (offline reward-hacking). The always-on demo BC term keeps `π_θ`
on the demonstration manifold — the same role `bc_flow_loss` plays in FQL/ACFQL and a
behavior-regularizer plays in AWAC/QAM. `λ_sft` trades off RL improvement vs. anchor.

## 6. Inference

Unchanged RECAP classifier-free guidance: `v = v_uncond + s·(v_cond − v_uncond)` with
`s = cfgrl_guidance_scale`. Larger `s` pushes harder toward high-advantage actions.

## 7. RECAP as a strict special case

| Knob | RECAP-reproducing value |
|---|---|
| `actor.model.expectile.enabled` | `false` (or `tau=0.5`) |
| stage-3 `--calql` | off |
| `actor.model.openpi.awr.enabled` | `false` |
| `actor.model.openpi.sft_aux.mode` | `reuse_unconditional` |
| `actor.model.openpi.sft_aux.weight` | `1.0` |

Under these, every PT4FM term degenerates to the corresponding RECAP computation —
the basis of the parity test in [pipeline.md](pipeline.md).

## 8. Future extensions (not in v1)
- **Chunk-level Q-critic + FQL/QAM `L_RL`**: swap the AWR-weighted BC term for a
  distill-DDPG / adjoint-matching term once a `Q(o, a_{t:t+H})` critic is trained.
  The SFT anchor and data pipeline carry over unchanged.
- **Per-timestep (within-chunk) advantage** instead of chunk-level.
- **Online fine-tuning**: Cal-QL was designed for offline→online; the calibrated
  value is a natural warm-start.
