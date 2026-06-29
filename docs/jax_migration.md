# PT4FM — JAX migration of the offline-RL policy update (hybrid)

The stage-4 policy update was migrated to **JAX openpi** (the user's `wkguo/openpi`
fork) so it trains the stable JAX pi0.5 and resumes directly from a finetuned JAX
checkpoint — no JAX→PyTorch conversion. Stages 1–3 (returns / value / advantages)
stay in the PyTorch PT4FM/RECAP pipeline and hand off via
`meta/advantages_{tag}.parquet`.

```
[data tools] sft2rl / filter_episodes / merge_datasets   (PT4FM, PyTorch — done)
[norm]       compute_norm_stats_fast (openpi)            (done; see note below)
[stage 1-3]  returns -> value -> advantages parquet      (PT4FM/RLinf, PyTorch)
[stage 4]    scripts/train_offline_rl.py pine_foundry_rl (openpi, JAX)  <-- new
```

## What was added (in the openpi fork `/home/wenkai/hdd_projects/openpi`)

| File | Purpose |
|---|---|
| `src/openpi/training/offline_rl.py` | method core: `OfflineRLConfig`, jnp `awr_weight`/`cfg_routing_masks`, `compute_offline_rl_loss`, and `create_offline_rl_data_loader` (attaches advantage + guidance prompts; builds the LeRobot dataset with the **0.3.x** API) |
| `scripts/train_offline_rl.py` | entrypoint forked from `scripts/train.py`: 3-element batch `(obs, actions, rl_info)`, offline-RL `train_step`, reuses `init_train_state`/checkpoint/sharding/wandb |
| `src/openpi/training/config.py` | `AWRConfig`/`SFTAuxConfig`/`CFGConfig`/`OfflineRLConfig`, an optional `TrainConfig.offline_rl` field, and the `pine_foundry_rl` config (resumes `foundry_policy/50000`) |
| `src/openpi/policies/pine_foundry_policy.py` | 3-camera Industrial_Arm transforms (added earlier for `pine_foundry`) |
| `src/openpi/training/offline_rl_test.py` | CPU unit tests (helpers + RECAP parity) |
| `scripts/compute_norm_stats_fast.py` | **fix**: pad raw stats to `action_dim` (see note) + `--repo-id` override |

## The objective (config-gated; defaults reduce to SFT/RECAP)

```
L = L_RL + lambda_sft * L_SFT
L_RL  = mean_i[ w_awr(A_i) * ||v_theta(.|o_i, l_cfg) - u_t||^2 ]   (conditional set)
L_SFT = mean_anchor[ ||v_theta(.|o, l_raw) - u_t||^2 ]
```

- `awr.enabled=false, sft_aux.mode=reuse_unconditional, sft_aux.weight=1, cfg.enabled=false`
  ⇒ exactly `jnp.mean(model.compute_loss)` (plain SFT/RECAP) — verified by parity test.
- `cfg.enabled=true` adds RECAP classifier-free guidance (per-sample prompt
  `"...\nAdvantage: positive|negative"`); needs the stage-3 boolean `advantage`.
- `awr.enabled=true` weights conditional samples by `exp((A-maxA)/beta)` (uses the
  stage-3 `advantage_weight` column).

## ⚠️ norm-stats fix (important)

openpi's `Normalize` runs **after** the policy Inputs pad state/action to
`action_dim` (32) and slices stats to the data dim, so norm-stats must be
**`action_dim`-length**. The fast script previously wrote raw-dim stats (state 13,
action 7), which **break training** (broadcast error) — and is why the foundry
checkpoint's bundled `norm_stats.json` (also 13/7) is inconsistent with the current
code. `compute_norm_stats_fast.py` now pads to `action_dim` with zero stats for the
padded dims (matching the regular `compute_norm_stats.py`). **The 6 norms were
recomputed to 32-dim.** For continued training from `foundry_policy/50000`, use these
recomputed 32-dim norms (real dims are identical; padded dims constant).

## Run order

```bash
cd /home/wenkai/hdd_projects/openpi   # JAX env (r3l / vla_pt)

# (norms already computed to 32-dim for the 5 per-dataset + merged dirs)

# stage 1-3 (PyTorch, PT4FM) to produce meta/advantages_<tag>.parquet on the merged set
#   bash PT4FM/scripts/1_compute_returns.sh ... ; 2_train_value.sh ... ; 3_compute_advantages.sh ...
#   (REQUIRES failure/reward data for a real RL signal — pure-success demos => degenerate.)

# stage 4 (JAX): offline-RL policy training, resuming foundry_policy/50000
uv run scripts/train_offline_rl.py pine_foundry_rl \
    --exp-name pine_rl_v1 \
    --offline-rl.advantage-tag <your_stage3_tag> \
    --offline-rl.awr.enabled True --offline-rl.awr.beta 0.7 \
    --offline-rl.sft-aux.mode reuse_unconditional --offline-rl.sft-aux.weight 1.0 \
    --fsdp-devices 3 --batch-size 24
```

## Verification (all passed)

- `offline_rl_test.py` (CPU): AWR β→∞⇒uniform/mean-1/monotonic; CFG routing; **RECAP
  parity** (`L == jnp.mean(compute_loss)`); AWR + separate_forward finite.
- Data plumbing on the **real merged set** (cfg off + on): batches yield
  `state[B,32]`, `actions[B,10,32]`, `rl_info{advantage,advantage_weight,is_demo}`
  (+ guidance tokens when CFG on).
- **Real pi0.5 forward + grads** through `compute_offline_rl_loss` (GPU): finite loss,
  grads flow.
- **Full jitted `train_step` + FSDP across 3 GPUs**: 2 steps, loss decreased
  (2.83→2.56), RL/SFT decomposition active, grad-norm finite.

Not yet exercised: the `scripts/train_offline_rl.py` CLI with the full 12 GB foundry
**resume** + checkpointing (pure glue reusing `train.py`'s `init_train_state` /
checkpoint manager, same mechanism as `pi05_fr3`). On first real run confirm the
foundry param tree matches the `pine_foundry` model (both pi0.5, action_horizon=10).

## Caveats
- A throwaway all-positive `meta/advantages_plumbing.parquet` was written on the
  merged set for plumbing tests; replace with a real stage-3 tag (or delete it).
- Meaningful offline RL still needs failure-containing rollouts or rewards; on
  pure-success demos every term degenerates to SFT (see [custom_data.md](custom_data.md)).
- `separate_forward` re-samples noise/time (a second `compute_loss`); default
  `reuse_unconditional` is single-forward and is the RECAP-parity path.
