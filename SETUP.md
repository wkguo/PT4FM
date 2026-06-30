# PT4FM — setup & migration runbook (clone on a new server and train)

PT4FM post-training spans **three repos** (kept separate; PT4FM is the umbrella with
data tools, the lerobot compat shim, docs, and the RLinf patch):

| repo | role | get it |
|---|---|---|
| **PT4FM** (this) | data tools (sft2rl/filter/merge), lerobot shim, docs, RLinf patch, requirements | `git clone <PT4FM url>` |
| **openpi** (fork, branch `pt4fm-offline-rl`) | JAX policy trainer (stage 4) + `pine_foundry*` configs + norm script | `git clone -b pt4fm-offline-rl https://github.com/wkguo/openpi.git` |
| **RLinf** (upstream + patch) | value model + advantages (stages 1-3, PyTorch) | `git clone https://github.com/RLinf/RLinf.git` then apply patch |

Target: RTX PRO 6000 (Blackwell, CUDA 12.8), Python 3.11 — same as the dev box.

## 1. Clone + patch
```bash
mkdir -p ~/foundry && cd ~/foundry
git clone <PT4FM url> PT4FM
git clone -b pt4fm-offline-rl https://github.com/wkguo/openpi.git openpi
git clone https://github.com/RLinf/RLinf.git RLinf
# add the industrial_arm robot_type to RLinf's RECAP value/advantage stages:
cd RLinf && git apply ../PT4FM/patches/rlinf_industrial_arm.patch && cd ..
```

## 2. Environment
```bash
conda create -n vla_pt python=3.11 -y && conda activate vla_pt
# 1. CUDA wheels (must come first):
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install "jax[cuda13]==0.7.2"
# 2. torchcodec — MUST use the cu128 index (PyPI default is cu13 and fails with libnvrtc.so.13):
pip install --no-deps torchcodec --index-url https://download.pytorch.org/whl/cu128
# 3. the rest:
pip install -r PT4FM/requirements.txt
# 4. editable installs so `import openpi` = the fork, and rlinf is importable:
pip install -e openpi
pip install -e RLinf            # or add RLinf to PYTHONPATH
```
Sanity (CPU): `JAX_PLATFORMS=cpu PYTHONPATH=$PWD/PT4FM python -m pytest openpi/src/openpi/training/offline_rl_test.py -q --noconftest`

## 3. Bring the model + data (NOT in git)
- **Base policy** `foundry_policy/50000` (JAX/Orbax, ~12 GB) — for stage-4 resume.
- **Dataset** (e.g. `ram_rl_ready`) — RL-ready LeRobot dir. Videos are symlinks, so copy
  with `rsync -aL` (dereference) or regenerate via PT4FM data tools (§5).
- **Value backbones** for stage 2: `siglip2-so400m-patch14-224`, `gemma-3-270m`
  (`huggingface-cli download ...`; gemma is gated → accept license + login).

## 4. Shared env vars
```bash
PT4FM=~/foundry/PT4FM; RLINF=~/foundry/RLinf; OPENPI=~/foundry/openpi
SHIM=$PT4FM/pt4fm/shim_path           # lerobot.common->lerobot + pyav video fallback (auto-loaded)
PY=$(which python)
```

## 5. Run (paper-faithful RECAP: CFG on, AWR/CalQL/expectile off)

`<DS>` = your RL-ready dataset dir; `<M>` = model dir (siglip/gemma/foundry).

```bash
# 0  RL-ready (materialize is_success from a per-episode field; absolute-pose/discrete gripper OK)
PYTHONPATH=$PT4FM $PY -m pt4fm.data.sft2rl --src <raw_rollout> --dst <DS> \
  --success-from-episode-field rollout_success --add-done --add-reward --failure-reward -300 --num-workers 16
# 0b reuse the base model's norm (keep pretrained normalization for fine-tuning), padded to 32:
#    (copy <foundry>/assets/norm_stats.json -> <DS>/norm_stats.json, padding state/action arrays to 32 with 0)

# 1  returns (no GPU)
cd $RLINF/examples/recap/process
PYTHONPATH=$SHIM:$PT4FM:$RLINF $PY compute_returns.py --config-name compute_returns \
  data.train_data_paths="[{dataset_path:<DS>,type:rollout}]" data.gamma=1.0 data.failure_reward=-300 data.tag=ram

# 2  value model (GPU)
cd $RLINF/examples/recap/value
PYTHONPATH=$SHIM:$PT4FM:$RLINF REPO_PATH=$RLINF $PY train_value.py --config-name libero_sft_value \
  data.tag=ram data.train_data_paths="[{dataset_path:<DS>,type:rollout,weight:1.0,robot_type:industrial_arm,model_type:pi05}]" \
  data.eval_data_paths="[]" data.action_dim=32 data.action_horizon=10 \
  actor.model.siglip_path=<M>/siglip2-so400m-patch14-224 actor.model.gemma3_path=<M>/gemma-3-270m \
  actor.model.tokenizer_path=<M>/gemma-3-270m actor.model.action_dim=32 actor.model.action_horizon=10

# 3  advantages (GPU) -> <DS>/meta/advantages_ram_N10_q30.parquet
#    FSDP_USE_ORIG_PARAMS=1 is REQUIRED: it rebuilds the value model in uniform bf16 to
#    match the FSDP-trained checkpoint; without it inference uses a mixed-precision build
#    (SigLIP embeddings fp32) and crashes with a layernorm/matmul dtype mismatch.
cd $RLINF/examples/recap/process
FSDP_USE_ORIG_PARAMS=1 PYTHONPATH=$SHIM:$PT4FM:$RLINF $PY compute_advantages.py --config-name compute_advantages \
  advantage.value_checkpoint=<VALUE_CKPT> advantage.tag=ram_N10_q30 advantage.returns_tag=ram advantage.positive_quantile=0.3 \
  advantage.model.siglip_path=<M>/siglip2-so400m-patch14-224 advantage.model.gemma3_path=<M>/gemma-3-270m \
  advantage.model.tokenizer_path=<M>/gemma-3-270m \
  data.train_data_paths="[{dataset_path:<DS>,type:rollout,robot_type:industrial_arm}]" \
  data.model_type=pi05 data.robot_type=industrial_arm data.advantage_lookahead_step=10 data.gamma=1.0
#    On a shared/busy GPU add: advantage.batch_size=96 advantage.num_dataloader_workers_per_gpu=4
#    and CUDA_VISIBLE_DEVICES=<free_gpu> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4  JAX policy training (CFG; resume foundry/50000)
#    Bool flags use tyro switch syntax: --flag (True) / --no-flag (False), no value argument.
#    <DS> must be an absolute path (repo_id is used directly as a local filesystem root).
#    Use CUDA_VISIBLE_DEVICES to select GPUs; XLA_PYTHON_CLIENT_MEM_FRACTION controls JAX mem.
cd $OPENPI
RUN_NAME=ram_$(date +%Y%m%d_%H%M%S)    # wandb run name; exp_name also sets the checkpoint subdir
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
PYTHONPATH=$OPENPI/src $PY scripts/train_offline_rl.py pine_foundry_rl \
  --project-name pi05_RECAP --exp-name "$RUN_NAME" \
  --data.repo-id <DS> --weight-loader.params-path <M>/foundry_policy/50000/params \
  --offline-rl.advantage-tag ram_N10_q30 \
  --offline-rl.cfg.enabled --offline-rl.cfg.positive-only-conditional \
  --offline-rl.cfg.uncond-prob 0.1 \
  --offline-rl.awr.no-enabled \
  --fsdp-devices 2 --batch-size 64
```

See [docs/jax_migration.md](docs/jax_migration.md), [docs/method.md](docs/method.md),
[docs/pipeline.md](docs/pipeline.md), [docs/custom_data.md](docs/custom_data.md) for details.

## Notes
- **Stages 2-3** must keep `PYTHONPATH=$SHIM:...` (RLinf RECAP imports `lerobot.common`; the shim
  redirects to `lerobot.datasets` and patches the PyAV video path under lerobot 0.3.x).
- **Stage 3** needs `FSDP_USE_ORIG_PARAMS=1` (forces uniform bf16, matching the FSDP-trained value
  checkpoint; without it the value model rebuilds in mixed precision and crashes with a layernorm /
  matmul dtype mismatch at the first inference forward).
- **Stage 4** (`--data.repo-id`) requires an **absolute path** (used directly as a filesystem root,
  not a HuggingFace repo id). Bool flags are tyro switch syntax: `--flag` / `--no-flag`.
  wandb is enabled by default; set `WANDB_API_KEY` or run `wandb login` beforehand.
- **Video decoding**: install `torchcodec` from the cu128 index (see §2); lerobot then auto-selects
  it as the fastest backend. The PT4FM compat shim ships a PyAV fallback for environments where
  torchvision lacks VideoReader and torchcodec is unavailable (stages 2-3 via `$SHIM`).
- `robot_type=industrial_arm` is added by `patches/rlinf_industrial_arm.patch` (5 files: view/state
  key mapping, checkpoint_utils build_input_transforms, value model inference dtype cast, eval-loader
  guard).
- Keep `action_dim=32` for the value model (13-D state pads up; never set 7).
- `failure_reward / positive_quantile / advantage_lookahead_step` are tunable; values above are the baseline.

## Visualization
After stage 3, visualize value + advantage to sanity-check the value model:
```bash
# V(o_t) curve + frame strip per episode (paper-style)
PYTHONPATH=$PT4FM $PY -m pt4fm.process.visualize_value \
  --dataset <DS> --tag ram_N10_q30 --num-frames 8 --success-ep <id> --failure-ep <id>

# Per-frame advantage curve + frame strip
PYTHONPATH=$PT4FM $PY -m pt4fm.process.visualize_advantage \
  --dataset <DS> --tag ram_N10_q30 --num-frames 8 --success-ep <id> --failure-ep <id>

# Annotate raw video with V(o_t) overlay (drawdown coloring: red=value drop not yet recovered)
PYTHONPATH=$PT4FM $PY -m pt4fm.process.annotate_value_video \
  --dataset <DS> --tag ram_N10_q30 --success-ep <id> --failure-ep <id>
# optional: --tolerance 0.02 (ignore sub-0.02 dips), --cameras observation.images.view1
```
