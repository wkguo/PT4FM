# PT4FM Framework — architecture, integration points, data flow

PT4FM is a **thin extension layer** over RLinf's RECAP pipeline. It never edits
`rlinf` or `openpi`; it subclasses their classes and, where construction is buried
inside a factory, temporarily rebinds the target class so the factory builds the
PT4FM subtype while we inherit all of its (heavy) loading logic.

## 1. Package layout

```
PT4FM/
├── pt4fm/
│   ├── losses.py                      # pure math: awr_weight, calql_calibrate,
│   │                                  #            expectile_weight, advantage_to_weight_np
│   ├── models/
│   │   ├── cfg_action_model.py        # PT4FMCfgActionModel(OpenPi0ForCFGActionPrediction)
│   │   │                              #   overrides forward(): AWR weight + SFT-aux term
│   │   │                              #   get_model(): class-rebind reuse of RLinf loader
│   │   └── value_critic.py            # PT4FMValueCriticModel(ValueCriticModel)
│   │                                  #   overrides _compute_categorical_loss(): IQL expectile
│   ├── data/
│   │   └── cfg_dataset.py             # PT4FMAdvantagePreservingDataset / PT4FMCFGDataLoaderImpl
│   │                                  #   carry advantage_weight + is_demo; pack [B,3]
│   ├── workers/
│   │   ├── cfg_worker.py              # PT4FMCfgWorker(FSDPCfgWorker)
│   │   └── value_worker.py            # PT4FMValueWorker(FSDPValueSftWorker)
│   ├── process/
│   │   └── postprocess_advantages.py  # stage-3b: Cal-QL + AWR weight + is_demo
│   ├── train_value.py                 # stage-2 entrypoint (hydra)
│   ├── train_policy.py                # stage-4 entrypoint (hydra)
│   └── tests/                         # unit tests for the pure cores
├── configs/  {libero_value.yaml, libero_policy.yaml}
├── scripts/  {env.sh, 1_…, 2_…, 3_…, 4_…}
└── docs/     {method.md, framework.md, pipeline.md}
```

## 2. Two reuse mechanisms

**(a) Subclass + override.** The actual learning changes live in three overrides:
- `PT4FMCfgActionModel.forward` — the AWR/CFG/SFT objective.
- `PT4FMValueCriticModel._compute_categorical_loss` — expectile re-weighting.
- `PT4FMAdvantagePreservingDataset.__getitem__` / `PT4FMCFGDataLoaderImpl.__iter__` —
  carry the extra per-sample fields.

**(b) Class-rebind on construct.** RLinf's `openpi_cfg.get_model` and
`value_model.get_model` look up the model class as a module global *at call time*.
PT4FM's `get_model` (in each model module) temporarily rebinds that global to the
PT4FM subclass, calls the RLinf loader (download, weight-loading, transforms,
`setup_wrappers`), then restores it. Result: the RLinf loader constructs a PT4FM model
with **zero copied loading code**. The same trick rebinds
`fsdp_cfg_worker.AdvantagePreservingDataset / CFGDataLoaderImpl` for the duration of
`build_dataloader`.

This means: when RLinf updates its loaders/transforms, PT4FM inherits the changes.

## 3. The opaque `advantage[B,3]` contract

RLinf's `FSDPCfgWorker.run_training` forwards the dataloader's third element verbatim
as `data["advantage"]`. To avoid overriding that hot loop, PT4FM packs three
per-sample fields into one tensor:

```
advantage[:, 0] = advantage (bool→float)   # CFG routing (RECAP)
advantage[:, 1] = advantage_weight (float) # AWR weight (stage-3b parquet)
advantage[:, 2] = is_demo (bool→float)     # SFT-anchor scoping
```

`PT4FMCFGDataLoaderImpl.__iter__` builds it; `PT4FMCfgActionModel._unpack_advantage`
decodes it (and still accepts a plain `[B]` advantage for the legacy/RECAP path).

## 4. End-to-end data flow

```
                          stage 1                stage 2                 stage 3a/3b                 stage 4
 LeRobot dataset ──▶ compute_returns ──▶ value-model SFT ──▶ compute_advantages + ──▶ AWR-CFG+SFT policy
 (sft / rollout)     meta/returns_*       (V; expectile)      postprocess (Cal-QL,      training (FSDP)
        │            + stats.json[return]   ▲                  AWR weight, is_demo)         │
        │                                   │                       │                       │
        └── tags: returns_tag ──────────────┴── value_ckpt ─────────┴── advantage_tag ──────┘
```

Per-stage I/O (all sidecars in `meta/`, originals untouched):

| Stage | Reads | Writes |
|---|---|---|
| 1 returns | dataset | `meta/returns_{tag}.parquet`, `meta/stats.json[return]` |
| 2 value | returns_{tag} | value checkpoint |
| 3a advantages | value_ckpt, returns_{tag} | `meta/advantages_{tag}.parquet` (RECAP cols) |
| 3b postprocess | advantages_{tag}, stats.json | `meta/advantages_{tag}.parquet` (+`advantage_weight`,`is_demo`; Cal-QL-updated `advantage`/`advantage_continuous`) |
| 4 policy | advantages_{tag}, pi0.5 ckpt | policy checkpoints |

`advantages_{tag}.parquet` columns after stage 3b: `episode_index, frame_index,
advantage(bool), advantage_continuous(float), advantage_weight(float), is_demo(bool),
value_current, value_next, return, …`.

## 5. Config surfaces

PT4FM knobs live inside existing RLinf config sub-trees so they survive
`validate_cfg`'s `OmegaConf.set_struct(True)` (they are present in the YAML before
struct-locking) and are injected onto the model config by RLinf's override loops:

- **Policy**: `actor.model.openpi.{awr, sft_aux}` — RLinf's `openpi_cfg.get_model`
  copies every `actor.model.openpi.*` key onto the model config `__dict__`, so
  `self.config.awr` / `self.config.sft_aux` are visible to `forward`.
- **Value**: `actor.model.expectile` — read by `pt4fm.models.value_critic.get_model`
  and pushed onto the model via `set_expectile()`.

## 6. Metrics

`PT4FMCfgActionModel.forward` returns RECAP's count/loss-sum keys (so the worker's
existing reduction still fires) **plus** PT4FM diagnostics: `pt4fm/rl_loss`,
`pt4fm/sft_loss`, `pt4fm/total_loss`, `pt4fm/sft_weight`, `pt4fm/awr_weight_mean`,
`pt4fm/awr_weight_max`. Value training adds `expectile_weight_mean` when enabled.

## 7. What is intentionally NOT changed
- `FSDPCfgWorker.run_training`, `FSDPValueSftWorker.run_training` — untouched
  (PT4FM rides inside model/data).
- RECAP `compute_returns.py`, `compute_advantages.py` (3a) — reused as-is.
- openpi `PI0Pytorch` flow loss, transforms, normalization — reused as-is.
