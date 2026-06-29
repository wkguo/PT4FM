# PT4FM — bringing your own data & embodiment

Two things are needed to post-train pi0.5 with PT4FM on a non-LIBERO robot:

1. **An RL-ready dataset** — a per-episode success/failure signal so returns/advantages
   are meaningful (stage 0: `sft2rl`).
2. **A matching data transform / `config_name`** — to align your cameras, state, and
   action dims with the model (e.g. `pi05_industrial_arm`).

This doc covers both, using the **Industrial_Arm 3-camera** dataset as the worked
example.

---

## 0. Why pure demos aren't enough (recap)

A demonstration LeRobot dataset has no reward/success column. RECAP/PT4FM derive the
reward from a per-episode **`is_success`** flag (stage-1 `compute_returns.py`:
`reward=-1/step`, terminal `0` for success / `failure_reward` for failure). With **only
success** episodes there is no failure contrast → the "RL" degenerates to ≈ SFT. Real
offline-RL signal needs either failure-containing rollouts or graded rewards. So the
unified entry point for *all* incoming data is **stage 0: `sft2rl`**, which materializes
the RL contract; pure demos pass through as all-success (use them as `sft`), while
self-collected rollouts carry real labels (use them as `rollout`).

---

## 1. Stage 0 — `sft2rl` (make data RL-ready)

Adds `is_success` (int64 0/1, required), and optionally `done`/`reward`, to every
episode parquet; updates `meta/info.json` + `meta/episodes_stats.jsonl`; **symlinks
videos** (near-zero extra disk) — never mutating the source.

```bash
# clean demonstrations (all success)
bash scripts/0_sft2rl.sh \
    --src /data/lerobot_demos \
    --dst /data/lerobot_demos_rlready \
    --default-success true --add-done --num-workers 16

# self-collected rollouts with a per-episode success map
#   success.json: {"0": 1, "1": 0, "2": 1, ...}   (episode_index -> 0/1)
bash scripts/0_sft2rl.sh \
    --src /data/rollouts --dst /data/rollouts_rlready \
    --success-map /data/rollouts/success.json --add-done --add-reward --num-workers 16
```

Success-label resolution priority (per episode):
1. `--success-map FILE` (json `{episode_index|source_name: 0/1}` or csv),
2. `--success-from-episode-field FIELD` (read a field from `meta/episodes.jsonl`),
3. `--default-success {true,false}` (default `true`).

Output columns (constant per episode unless noted):

| column | dtype | meaning |
|---|---|---|
| `is_success` | int64 0/1 | **required** by stage 1; sft data → 1 |
| `done` | int64 | 1 on last frame (`--add-done`) |
| `reward` | float32 | `-1`/step, terminal `0`/`failure_reward` (`--add-reward`; RECAP recomputes anyway) |

Then declare the output in the data config as `type: sft` (all-success demos) or
`type: rollout` (has failures), and run stage 1.

> For the hundreds of hours of self-collected data: collect with the SFT-finetuned
> pi0.5 in the target task, log per-episode success (env eval / human / VLM judge) into
> a `success.json`, run stage 0 with `--success-map`, mix with the demo `sft` set.

---

## 2. Industrial_Arm data transform & `config_name`

Schema (`meta/info.json`): `action` float32 **[7]** = `(x,y,z,rx,ry,rz,gripper)`
absolute EEF pose; `observation.state` float32 **[13]** = pose(7)+force/torque(6);
three 480×640 cameras `observation.images.{hand,view1,view2}`; 15 Hz; prompt from task.

Provided in [`pt4fm/integrations/industrial_arm.py`](../pt4fm/integrations/industrial_arm.py):

- `IndustrialArmInputs` / `IndustrialArmOutputs` — openpi data⇄model transforms.
- `LeRobotIndustrialArmDataConfig` — repack + transforms + model transforms.
- `register_industrial_arm_configs()` — injects the **`pi05_industrial_arm`** config
  into RLinf's `get_openpi_config` registry (PT4FM's CFG worker calls this
  automatically; the policy `config_name` just needs to be `pi05_industrial_arm`).

Camera → pi0.5 image-slot mapping (all three real, masks all True):

| pi0.5 slot | dataset camera |
|---|---|
| `base_0_rgb` | `observation.images.view1` (primary external) |
| `left_wrist_0_rgb` | `observation.images.hand` (wrist) |
| `right_wrist_0_rgb` | `observation.images.view2` (secondary external) |

State **[13]** and the action chunk **[H,7]** are padded to the model action dim (32);
outputs slice back to the first 7 dims. Actions are **absolute** Cartesian, so no delta
transform by default — set `extra_delta_transform=True` (delta on dims 0–5, gripper
absolute) only if your checkpoint expects deltas.

Use it by setting in `configs/libero_policy.yaml` (or via CLI):

```yaml
actor:
  model:
    openpi:
      config_name: "pi05_industrial_arm"
```

If your camera keys differ, either edit the defaults on
`LeRobotIndustrialArmDataConfig` or copy the module for a new embodiment.

---

## 3. Norm stats (one-time, per dataset)

openpi normalizes state/actions with `norm_stats` it loads from
`<pi05_checkpoint>/assets/<asset_id>/norm_stats.json` (here `asset_id =
"industrial_arm"`). Compute them once from your RL-ready dataset with openpi's tool,
then place the result under the checkpoint assets dir:

```bash
# from the openpi repo, against your RL-ready LeRobot dataset
python scripts/compute_norm_stats.py --config-name pi05_industrial_arm \
    data.repo_id=/data/lerobot_demos_rlready
# -> copy assets/industrial_arm/norm_stats.json into <pi05_checkpoint>/assets/industrial_arm/
```

(Stage 2's value model uses the dataset `return` stats from stage 1, not these.)

---

## 4. End-to-end for the Industrial_Arm example

```bash
# 0) RL-ready (demos => all success)
bash scripts/0_sft2rl.sh --src /data/industrial_arm --dst /data/industrial_arm_rlready \
    --default-success true --add-done --num-workers 16
# (compute norm_stats as in §3, place under the pi05 checkpoint assets)

# 1) returns   2) value   3) advantages   4) policy
bash scripts/1_compute_returns.sh compute_returns \
    data.train_data_paths='[{dataset_path:/data/industrial_arm_rlready,type:sft}]' data.tag=v1
bash scripts/2_train_value.sh libero_value data.tag=v1 \
    data.train_data_paths='[{dataset_path:/data/industrial_arm_rlready,type:sft,weight:1.0,robot_type:industrial_arm,model_type:pi05}]'
bash scripts/3_compute_advantages.sh --value-ckpt <ckpt> --tag v1_N10_q30 --dataset /data/industrial_arm_rlready:sft \
    --lookahead 10 --awr-beta 0.7 --calql
bash scripts/4_train_policy.sh libero_policy \
    data.advantage_tag=v1_N10_q30 actor.model.model_path=<pi05_ckpt> \
    actor.model.openpi.config_name=pi05_industrial_arm \
    data.train_data_paths='[{dataset_path:/data/industrial_arm_rlready,type:sft,weight:1.0}]'
```

> Reminder: until you add failure-containing rollouts, this all-success `sft` run is
> RL-degenerate (≈ SFT). The value of stage 0 is that the *same* commands then accept
> your labeled rollouts unchanged — just add them as `type: rollout`.
