import type { ModelConfigView } from "../state/store";

interface Props {
  model: ModelConfigView;
}

export function ModelStructure({ model }: Props) {
  const layers = model.architecture.num_layers;
  const heads = model.attention.num_attention_heads;
  const h = model.architecture.hidden_size;
  const ffn = model.architecture.ffn_hidden_size;
  const norm = model.norm.normalization;
  const mla = model.attention.use_mla;
  const moe = model.moe.enabled;
  const numDense = moe ? model.moe.first_k_dense_replace : layers;
  const numMoE = moe ? layers - model.moe.first_k_dense_replace : 0;

  return (
    <div className="structure">
      <h3>Overall</h3>
      <ol className="structure-stack">
        <li>Embedding (vocab={model.architecture.vocab_size}, hidden={h})</li>
        {numDense > 0 && (
          <li>
            [Dense TransformerLayer] × <b>{numDense}</b>
          </li>
        )}
        {numMoE > 0 && (
          <li>
            [MoE TransformerLayer] × <b>{numMoE}</b>
          </li>
        )}
        <li>Final {norm}</li>
        {model.architecture.untie_embeddings_and_output_weights && <li>Output projection (untied)</li>}
      </ol>

      <h3>{numDense > 0 ? "Dense layer" : "Layer"}</h3>
      <ol className="structure-stack indent">
        <li>{norm}</li>
        <li>
          {mla
            ? `MLA (heads=${heads}, kv_lora_rank=${model.attention.kv_lora_rank}, qk_nope=${model.attention.qk_nope_head_dim}, qk_rope=${model.attention.qk_rope_head_dim}, v_dim=${model.attention.v_head_dim})`
            : `Attention (heads=${heads}, kv_heads=${model.attention.num_query_groups ?? heads}, head_dim=${model.attention.kv_channels ?? h / heads})`}
        </li>
        <li>{norm}</li>
        <li>MLP ({model.mlp.swiglu ? "SwiGLU" : "vanilla"}, ffn={ffn})</li>
      </ol>

      {numMoE > 0 && (
        <>
          <h3>MoE layer</h3>
          <ol className="structure-stack indent">
            <li>{norm}</li>
            <li>
              {mla
                ? `MLA (heads=${heads}, kv_lora_rank=${model.attention.kv_lora_rank}, qk_nope=${model.attention.qk_nope_head_dim}, qk_rope=${model.attention.qk_rope_head_dim}, v_dim=${model.attention.v_head_dim})`
                : `Attention (heads=${heads}, head_dim=${model.attention.kv_channels ?? h / heads})`}
            </li>
            <li>{norm}</li>
            <li>
              MoE (top-{model.moe.moe_router_topk} of {model.moe.num_routed_experts} routed +{" "}
              {model.moe.num_shared_experts} shared, moe_ffn={model.moe.moe_ffn_hidden_size})
            </li>
          </ol>
        </>
      )}
    </div>
  );
}
