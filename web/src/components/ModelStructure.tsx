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
  const v4 = !!model.attention.use_deepseek_v4;
  const moe = model.moe.enabled;
  const numDense = moe ? model.moe.first_k_dense_replace : layers;
  const numMoE = moe ? layers - model.moe.first_k_dense_replace : 0;
  const hcMult = model.hyper_connection?.hc_mult ?? 1;
  const mtpLayers = model.mtp?.num_layers ?? 0;

  const headDim = model.attention.kv_channels ?? h / heads;
  const attentionLine = v4
    ? `V4 hybrid attention (heads=${heads}, single-latent KV head_dim=${headDim}, q_lora_rank=${model.attention.q_lora_rank}, o_lora_rank=${model.attention.o_lora_rank}, o_groups=${model.attention.o_groups}, per-layer cr ∈ {0, 4, 128})`
    : mla
    ? `MLA (heads=${heads}, kv_lora_rank=${model.attention.kv_lora_rank}, qk_nope=${model.attention.qk_nope_head_dim}, qk_rope=${model.attention.qk_rope_head_dim}, v_dim=${model.attention.v_head_dim})`
    : `Attention (heads=${heads}, kv_heads=${model.attention.num_query_groups ?? heads}, head_dim=${headDim})`;

  return (
    <div className="structure">
      <h3>Overall</h3>
      <ol className="structure-stack">
        <li>Embedding (vocab={model.architecture.vocab_size}, hidden={h})</li>
        {v4 && hcMult > 1 && <li>Lift to {hcMult} mHC streams ([B, S, {hcMult}, {h}])</li>}
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
        {v4 && hcMult > 1 && <li>HyperHead — collapse {hcMult} streams → 1</li>}
        {v4 && mtpLayers > 0 && (
          <li>
            [MTP block (eh_proj + V4 layer{hcMult > 1 ? " + per-depth HyperHead" : ""})] × <b>{mtpLayers}</b>
          </li>
        )}
        <li>Final {norm}</li>
        {model.architecture.untie_embeddings_and_output_weights && <li>Output projection (untied)</li>}
      </ol>

      <h3>{numDense > 0 ? "Dense layer" : "Layer"}</h3>
      <ol className="structure-stack indent">
        {v4 && hcMult > 1 && <li>HyperMixer (attn) — compute pre/post/comb for {hcMult} streams</li>}
        <li>{norm}</li>
        <li>{attentionLine}</li>
        {v4 && hcMult > 1 && <li>HyperMixer (FFN) — compute pre/post/comb for {hcMult} streams</li>}
        <li>{norm}</li>
        <li>
          MLP ({model.mlp.swiglu ? "SwiGLU" : "vanilla"}
          {model.mlp.swiglu_limit && model.mlp.swiglu_limit > 0
            ? `, clamp=${model.mlp.swiglu_limit}`
            : ""}
          , ffn={ffn})
        </li>
      </ol>

      {numMoE > 0 && (
        <>
          <h3>MoE layer</h3>
          <ol className="structure-stack indent">
            {v4 && hcMult > 1 && <li>HyperMixer (attn)</li>}
            <li>{norm}</li>
            <li>{attentionLine}</li>
            {v4 && hcMult > 1 && <li>HyperMixer (FFN)</li>}
            <li>{norm}</li>
            <li>
              MoE (top-{model.moe.moe_router_topk} of {model.moe.num_routed_experts} routed +{" "}
              {model.moe.num_shared_experts} shared, moe_ffn={model.moe.moe_ffn_hidden_size}
              {model.moe.router_score_function && model.moe.router_score_function !== "softmax"
                ? `, score=${model.moe.router_score_function}`
                : ""}
              {model.moe.num_hash_layers && model.moe.num_hash_layers > 0
                ? `; first ${model.moe.num_hash_layers} layers use tid2eid hash routing`
                : ""}
              )
            </li>
          </ol>
        </>
      )}
    </div>
  );
}
