# PT4FM Pipeline — step-by-step run guide

Four sequential stages over LeRobot datasets. Stages 1 and 3a reuse RLinf's RECAP
scripts unchanged; stages 2, 3b, 4 are PT4FM. All outputs are sidecars in each
dataset's `meta/` — original data is never modified.

## 0. Prerequisites

- RLinf embodied env for full FSDP runs (`source switch_env openpi`, or
  `bash requirements/install.sh embodied --model openpi`).
- Checkpoints: pi0.5 base (PyTorch) for the policy; SigLIP2-so400m + Gemma3-270M for
  the value model. See RLinf RECAP docs for download/convert.
- `source scripts/env.sh` (sets `PYTHONPATH` for `pt4fm` + `rlinf` + `openpi`).

## 1. Tag system

Tags thread artifacts between stages — keep them consistent:

| Tag | Set in | Consumed by |
|---|---|---|
| `returns_tag` (e.g. `fail300`) | stage 1 `data.tag` | stage 2 `data.tag`, stage 3a `advantage.returns_tag`, stage 3b `--returns-tag` |
| `advantage_tag` (e.g. `fail300_N10_q30`) | stage 3a `advantage.tag` / 3b `--tag` | stage 4 `data.advantage_tag` |

`train_data_paths` (the dataset list) must match across all stages.

## 2. Stage 1 — compute returns

```bash
bash scripts/1_compute_returns.sh compute_returns \
    data.train_data_paths='[{dataset_path:/data/libero_sft,type:sft},
                            {dataset_path:/data/libero_rollout,type:rollout}]' \
    data.gamma=1.0 data.failure_reward=-300 data.tag=fail300
```
Writes `meta/returns_fail300.parquet` and updates `meta/stats.json` with `return`
statistics (needed by stage-3b Cal-QL normalization).

## 3. Stage 2 — value-model SFT

Edit `configs/libero_value.yaml` (model paths, `data.train_data_paths`,
`eval_data_paths`) or override on the CLI:

```bash
bash scripts/2_train_value.sh libero_value \
    data.tag=fail300 \
    actor.model.siglip_path=/ckpt/siglip2-so400m-patch14-224 \
    actor.model.gemma3_path=/ckpt/gemma-3-270m \
    actor.model.tokenizer_path=/ckpt/gemma-3-270m \
    actor.model.expectile.enabled=true actor.model.expectile.tau=0.7
```
Watch `eval/spearman_correlation` (rank agreement between predicted and true returns)
rise. Note the checkpoint path (`…/checkpoints/global_step_{N}/actor/model_state_dict`)
for stage 3a. Set `expectile.enabled=false` for vanilla RECAP value training.

## 4. Stage 3 — compute + augment advantages

`scripts/3_compute_advantages.sh` runs 3a (RECAP distributed value inference) then 3b
(PT4FM Cal-QL + AWR weight + is_demo):

```bash
bash scripts/3_compute_advantages.sh \
    --value-ckpt /path/to/value_ckpt \
    --tag fail300_N10_q30 --returns-tag fail300 \
    --dataset /data/libero_sft:sft --dataset /data/libero_rollout:rollout \
    --lookahead 10 --gamma 1.0 --positive-quantile 0.3 \
    --awr-beta 0.7 --awr-wmax 20 --calql --nproc 4
```
Notes:
- `--nproc N` uses `torchrun` for the (3a) value inference.
- Add `--skip-recap` to re-run only 3b (e.g. retune `--awr-beta` without re-inferring).
- 3a's RECAP config still needs its model paths; pass them via the RECAP config or
  extend the script's 3a invocation. The value checkpoint loads into RLinf's
  `ValueCriticModel` for inference even if trained with expectile (same parameters).

Verify:
```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet('/data/libero_sft/meta/advantages_fail300_N10_q30.parquet')
print(list(df.columns))
print(df[['advantage','advantage_weight','is_demo']].describe(include='all'))
PY
```
Expect columns including `advantage`, `advantage_continuous`, `advantage_weight`,
`is_demo`.

## 5. Stage 4 — policy post-training

Edit `configs/libero_policy.yaml` (`actor.model.model_path`, `data.train_data_paths`)
then:

```bash
# Full method: AWR + (Cal-QL value from stage 2/3) + SFT anchor
bash scripts/4_train_policy.sh libero_policy \
    data.advantage_tag=fail300_N10_q30 \
    actor.model.model_path=/ckpt/pi05_base_pytorch \
    actor.model.openpi.config_name=pi05_libero \
    actor.model.openpi.awr.enabled=true actor.model.openpi.awr.beta=0.7 \
    actor.model.openpi.sft_aux.mode=separate_forward \
    actor.model.openpi.sft_aux.weight=0.5
```
Monitor `train/actor/loss`, `pt4fm/rl_loss`, `pt4fm/sft_loss`, `pt4fm/awr_weight_mean`,
and the RECAP CFG ratios (`conditional_ratio`, `positive_label_ratio`, …).

At inference, set `actor.model.openpi.cfgrl_guidance_scale` (e.g. 1.0–3.0) to control
guidance strength when serving the policy.

## 6. Parity test (RECAP reproduction)

To confirm PT4FM is a strict superset, run stage 4 with all enhancements off and
compare against RLinf's `train_cfg.py` on the same tiny shard / seed:

```bash
bash scripts/4_train_policy.sh libero_policy \
    data.advantage_tag=fail300_N10_q30 \
    actor.model.openpi.awr.enabled=false \
    actor.model.openpi.sft_aux.mode=reuse_unconditional \
    actor.model.openpi.sft_aux.weight=1.0 \
    actor.micro_batch_size=2 actor.global_batch_size=2 runner.max_steps=20
```
`pt4fm/total_loss` should track RECAP's `train/actor/loss` within numerical noise
(both reduce to the mean flow loss). `pt4fm/rl_loss + pt4fm/sft_loss ≈ total`.

## 7. Smoke test (no cluster, pure cores)

The math primitives and stage-3b core run without RLinf/Ray:
```bash
PYTHONPATH=$PWD python -m pytest pt4fm/tests -q
```
For a minimal single-process policy step, use a tiny `micro_batch_size`/`global_batch_size`
and `cluster.num_nodes=1` with a 1-GPU placement; full multi-GPU FSDP needs the RLinf
embodied env.

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: rlinf/openpi` | `source scripts/env.sh`; check `RLINF_PATH`/`OPENPI_PATH`. |
| `Advantage file not found` (stage 4) | run stage 3 first; check `data.advantage_tag` == stage-3 `--tag`. |
| `… samples missing from advantages lookup` | dataset/tag mismatch — re-run stage 3 on the *same* `train_data_paths`. |
| stage-3b `stats.json … 'return'` missing | run stage 1 first (it writes return stats); needed only for `--calql`. |
| Hydra "could not find 'model/value'" | ensure `REPO_PATH` points to RLinf root (searchpath = `$REPO_PATH/examples/sft/config`). |
| OOM in stage 4 | lower `micro_batch_size`; keep `gradient_checkpointing: True`; prefer `sft_aux.mode=reuse_unconditional` (1× forward). |
| SFT anchor too weak on pure-`sft` data | `reuse_unconditional` only anchors the CFG-unconditional subset; switch to `sft_aux.mode=separate_forward`. |
