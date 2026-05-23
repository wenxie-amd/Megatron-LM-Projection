# M4 — MoE / MLA support

## What landed

- Configs (`configs.py`):
  - `AttentionConfig` gained MLA fields: `use_mla`, `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`.
  - New `MoEConfig`: `enabled`, `moe_ffn_hidden_size`, `num_routed_experts`, `num_shared_experts`, `moe_router_topk`, `moe_layer_freq`, `first_k_dense_replace`, `add_router_bias`. `ModelConfig.moe` defaults to an empty (disabled) MoE.
- `core/modules.py`:
  - `AttentionModule.param_count()` branches on `use_mla` to call `_mla_param_count` vs `_mha_param_count`. MLA implements: optional Q LoRA path, KV low-rank latent + rope-only stub, output projection sized by `num_heads * v_head_dim`.
  - New `MoEModule`: gate + routed experts + shared experts. `param_count(ep_size=…)` divides routed experts across the EP group; shared experts and the gate stay replicated. Errors if `num_routed_experts % ep_size != 0`.
- `core/layer.py`: `TransformerLayer` now takes a `kind: "dense" | "moe"` and composes either `MLPModule` or `MoEModule`. Breakdown entries are `"mlp"` for dense layers, `"moe"` for MoE layers.
- `core/block.py`: tracks `first_layer_idx` and computes `num_dense_on_rank` / `num_moe_on_rank` by intersecting the rank's slice with `[0, first_k_dense_replace)`. Holds one representative of each layer kind.
- `parallel/ranks.py`: `ModelPartition` gained `first_layer_idx`. `partition_for_rank` computes it from the cumsum of preceding stages' layer counts. `validate_parallel_config` adds the EP-divides-num_routed_experts check.
- `core/model.py`: `param_breakdown` emits `"layer.*"` entries for dense layers and `"moe_layer.*"` entries for MoE layers, keeping Llama's existing namespace intact.
- New YAML: `model_configs/deepseek_v2_lite.yaml`.
- UI:
  - `ModelStructure.tsx` shows the dense layer block, the MoE layer block, and the MLA-specific attention spec when applicable.
  - `Step1Model.tsx` adds an "MoE" config group when `moe.enabled`.
  - `state/store.ts` typed `attention` and `moe` views to match.
- Tests: `tests/unit/test_moe.py` (6 tests): total param count pinned to 15,706,484,224; per-module breakdown; routed-expert size; EP sharding behaviour; PP-split with dense-first ordering.

## Verified numbers

- DeepSeek-V2-Lite total params: **15,706,484,224** (matches DeepSeek's published 15.7B figure).
- Per routed expert: 8,650,752 params.
- Per MoE block (gate + 64 routed + 2 shared): 571,080,704 params.

## Known limitations

- Activation memory for MoE doesn't yet model token-dispatch / pad-to-capacity / load-imbalance overheads. Same formula as dense for now.
- `moe_layer_freq > 1` (alternating dense/MoE pattern) is parsed into config but not honoured in the layer-kind decision — DeepSeek-V2-Lite has freq=1 so this doesn't bite v1.
- TP+EP combined sharding details (e.g. 2D expert sharding) are not modelled.
