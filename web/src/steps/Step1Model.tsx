import { useEffect, useState } from "react";

import { ModelStructure } from "../components/ModelStructure";
import { ParamPie } from "../components/ParamPie";
import { getBridge } from "../pyodide/bridge";
import { useProjection } from "../state/context";
import type { ModelConfigView } from "../state/store";
import { effectiveModelName, isProxyModel } from "../state/store";

interface Breakdown {
  param_count: number;
  param_breakdown: { name: string; count: number }[];
  ffn_breakdown: {
    kind: "mlp" | "moe";
    entries: { name: string; count: number }[];
  };
}

export function Step1Model() {
  const { state, dispatch, loadModel } = useProjection();
  const [breakdown, setBreakdown] = useState<Breakdown | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!state.modelConfig) return;
    setLoading(true);
    setErr(null);
    setBreakdown(null);
    (async () => {
      try {
        const bridge = await getBridge();
        const payload =
          state.numLayersOverride !== null && state.numLayersOverride !== state.modelConfig!.architecture.num_layers
            ? {
                ...state.modelConfig!,
                architecture: { ...state.modelConfig!.architecture, num_layers: state.numLayersOverride },
              }
            : (state.selectedModel ?? state.modelConfig!.name);
        const b = bridge.getModelBreakdown(payload);
        setBreakdown(b);
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, [state.modelConfig, state.numLayersOverride, state.selectedModel]);

  const config = state.modelConfig;

  return (
    <section className="step">
      <h2>Step 1 · Model Selection</h2>

      <label className="field">
        <span className="field-label">Model</span>
        <select
          value={state.selectedModel ?? ""}
          onChange={(e) => loadModel(e.target.value)}
        >
          <option value="" disabled>
            — choose a model —
          </option>
          {state.availableModels.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>

      {config && (
        <>
          <div className="model-name-row">
            <h3>{effectiveModelName(state)}</h3>
            {isProxyModel(state) && <span className="proxy-badge">proxy</span>}
          </div>
          {state.selectedModel && state.selectedModel.includes("/") && (
            <p className="hf-link">
              <a href={`https://huggingface.co/${state.selectedModel}`} target="_blank" rel="noreferrer">
                View on HuggingFace ↗
              </a>
            </p>
          )}
          {config.description && <p className="description">{config.description}</p>}

          <div className="config-grid">
            <ConfigGroup
              title="Architecture"
              entries={[
                ["num_layers", config.architecture.num_layers, { editable: true }],
                ["hidden_size", config.architecture.hidden_size],
                ["ffn_hidden_size", config.architecture.ffn_hidden_size],
                ["vocab_size", config.architecture.vocab_size],
                ["max_position_embeddings", config.architecture.max_position_embeddings],
                ["untie_embeddings_and_output_weights", config.architecture.untie_embeddings_and_output_weights],
                ["make_vocab_size_divisible_by", config.architecture.make_vocab_size_divisible_by],
              ]}
              editValue={state.numLayersOverride ?? config.architecture.num_layers}
              onEdit={(v) => dispatch({ type: "OVERRIDE_NUM_LAYERS", value: v === config.architecture.num_layers ? null : v })}
            />
            <ConfigGroup
              title={
                config.attention.use_deepseek_v4
                  ? "Attention (DeepSeek-V4 hybrid)"
                  : config.attention.use_mla
                  ? "Attention (MLA)"
                  : "Attention"
              }
              entries={attentionEntries(config)}
            />
            {config.attention.use_deepseek_v4 && config.hybrid_attention && (
              <V4HybridAttentionGroup
                hybrid={config.hybrid_attention}
                numLayers={config.architecture.num_layers}
                mtpLayers={config.mtp?.num_layers ?? 0}
              />
            )}
            {config.hyper_connection && config.hyper_connection.hc_mult > 1 && (
              <ConfigGroup
                title="Hyper-Connections (mHC)"
                entries={[
                  ["hc_mult", config.hyper_connection.hc_mult],
                  ["sinkhorn_iters", config.hyper_connection.sinkhorn_iters],
                ]}
              />
            )}
            {config.mtp && config.mtp.num_layers > 0 && (
              <ConfigGroup
                title="Multi-Token Prediction (MTP)"
                entries={[
                  ["num_layers", config.mtp.num_layers],
                  ["use_separate_hc_head", config.mtp.use_separate_hc_head],
                ]}
              />
            )}
            <ConfigGroup
              title="MLP"
              entries={[
                ["swiglu", config.mlp.swiglu],
                ["add_bias_linear", config.mlp.add_bias_linear],
                ...(config.mlp.swiglu_limit && config.mlp.swiglu_limit > 0
                  ? ([["swiglu_limit", config.mlp.swiglu_limit]] as Entry[])
                  : []),
              ]}
            />
            {config.moe.enabled && (
              <ConfigGroup
                title="MoE"
                entries={[
                  ["moe_ffn_hidden_size", config.moe.moe_ffn_hidden_size],
                  ["num_routed_experts", config.moe.num_routed_experts],
                  ["num_shared_experts", config.moe.num_shared_experts],
                  ["moe_router_topk", config.moe.moe_router_topk],
                  ["moe_layer_freq", config.moe.moe_layer_freq],
                  ["first_k_dense_replace", config.moe.first_k_dense_replace],
                  ...(config.moe.router_score_function && config.moe.router_score_function !== "softmax"
                    ? ([["router_score_function", config.moe.router_score_function]] as Entry[])
                    : []),
                  ...(config.moe.num_hash_layers && config.moe.num_hash_layers > 0
                    ? ([["num_hash_layers", config.moe.num_hash_layers]] as Entry[])
                    : []),
                ]}
              />
            )}
            <ConfigGroup
              title="Normalization"
              entries={[
                ["normalization", config.norm.normalization],
                ["layernorm_epsilon", config.norm.layernorm_epsilon],
              ]}
            />
            <ConfigGroup
              title="Position embedding"
              entries={[
                ["position_embedding_type", config.position_embedding.position_embedding_type],
                ["rotary_base", config.position_embedding.rotary_base],
                ["rotary_percent", config.position_embedding.rotary_percent],
              ]}
            />
          </div>

          <ModelStructure
            model={
              state.numLayersOverride !== null
                ? { ...config, architecture: { ...config.architecture, num_layers: state.numLayersOverride } }
                : config
            }
          />

          {loading && <p>Computing parameter breakdown…</p>}
          {err && <pre className="error">{err}</pre>}
          {breakdown && (
            <div>
              <h3>Parameter breakdown ({breakdown.param_count.toLocaleString()} total)</h3>
              <ParamPie data={breakdown.param_breakdown} />
              <h3>
                {breakdown.ffn_breakdown.kind === "moe"
                  ? "Inside one MoE block"
                  : `Inside one MLP block${config.mlp.swiglu ? " (SwiGLU)" : ""}`}
              </h3>
              <ParamPie data={breakdown.ffn_breakdown.entries} />
            </div>
          )}
        </>
      )}
    </section>
  );
}

type EntryValue = string | number | boolean;
type EntryMeta = { editable?: boolean };
type Entry = [string, EntryValue] | [string, EntryValue, EntryMeta];

interface ConfigGroupProps {
  title: string;
  entries: Entry[];
  editValue?: number;
  onEdit?: (v: number) => void;
}

function attentionEntries(config: ModelConfigView): Entry[] {
  const att = config.attention;
  if (att.use_deepseek_v4) {
    // V4 hybrid: single-latent KV (K=V=kv_channels broadcast to all query
    // heads, so num_query_groups=1), Q LoRA path, grouped low-rank O.
    return [
      ["num_attention_heads", att.num_attention_heads],
      ["num_query_groups (single-latent KV)", att.num_query_groups ?? 1],
      ["kv_channels (head_dim)", att.kv_channels ?? "—"],
      ["q_lora_rank", att.q_lora_rank ?? "—"],
      ["o_lora_rank", att.o_lora_rank ?? "—"],
      ["o_groups", att.o_groups ?? 1],
      ["attention_dropout", att.attention_dropout],
    ];
  }
  if (att.use_mla) {
    return [
      ["num_attention_heads", att.num_attention_heads],
      ["use_mla", true],
      ["q_lora_rank", att.q_lora_rank ?? "—"],
      ["kv_lora_rank", att.kv_lora_rank ?? "—"],
      ["qk_nope_head_dim", att.qk_nope_head_dim ?? "—"],
      ["qk_rope_head_dim", att.qk_rope_head_dim ?? "—"],
      ["v_head_dim", att.v_head_dim ?? "—"],
      ["attention_dropout", att.attention_dropout],
    ];
  }
  return [
    ["num_attention_heads", att.num_attention_heads],
    ["num_query_groups", att.num_query_groups ?? "—"],
    ["kv_channels", att.kv_channels ?? "—"],
    ["attention_dropout", att.attention_dropout],
    ["add_qkv_bias", att.add_qkv_bias],
  ];
}

interface V4HybridProps {
  hybrid: NonNullable<ModelConfigView["hybrid_attention"]>;
  numLayers: number;
  mtpLayers: number;
}

/** Group the per-layer compress_ratios by branch and render one card per
 *  branch listing the matching layer ids. Helps users see at a glance which
 *  layers are dense / CSA / HCA. */
function V4HybridAttentionGroup({ hybrid, numLayers, mtpLayers }: V4HybridProps) {
  const decoderRatios = hybrid.compress_ratios.slice(0, numLayers);
  const mtpRatios = hybrid.compress_ratios.slice(numLayers, numLayers + mtpLayers);

  const branches = [
    {
      cr: 0,
      label: "dense + SWA (cr=0)",
      desc: "Full attention with sliding-window mask. No long-range compression branch.",
    },
    {
      cr: 4,
      label: "CSA (cr=4)",
      desc: `Compressed-Sparse Attention. Compressor (overlap, ratio=4) → Indexer picks top-${hybrid.index_topk} compressed slots per query.`,
    },
    {
      cr: 128,
      label: "HCA (cr=128)",
      desc: "Heavily-Compressed Attention. Compressor (non-overlap, ratio=128) with full causal pool cross-attention.",
    },
  ] as const;

  return (
    <div className="config-group">
      <h4>V4 hybrid attention — per-layer schedule</h4>
      <dl>
        <div className="config-row">
          <dt>index_topk</dt>
          <dd>{hybrid.index_topk}</dd>
        </div>
        <div className="config-row">
          <dt>index_head_dim</dt>
          <dd>{hybrid.index_head_dim}</dd>
        </div>
        <div className="config-row">
          <dt>index_n_heads</dt>
          <dd>{hybrid.index_n_heads}</dd>
        </div>
        <div className="config-row">
          <dt>attn_sliding_window</dt>
          <dd>{hybrid.attn_sliding_window}</dd>
        </div>
        <div className="config-row">
          <dt>attn_sink</dt>
          <dd>{String(hybrid.attn_sink)}</dd>
        </div>
        <div className="config-row">
          <dt>compress_rope_theta</dt>
          <dd>{hybrid.compress_rope_theta}</dd>
        </div>
      </dl>
      <div className="v4-branches">
        {branches.map(({ cr, label, desc }) => {
          const decoderIds = decoderRatios
            .map((r, i) => (r === cr ? i : -1))
            .filter((i) => i >= 0);
          const mtpIds = mtpRatios
            .map((r, i) => (r === cr ? numLayers + i : -1))
            .filter((i) => i >= 0);
          const total = decoderIds.length + mtpIds.length;
          if (total === 0) return null;
          return (
            <div key={cr} className="v4-branch">
              <h5>
                {label} <span className="v4-branch-count">— {total} layer(s)</span>
              </h5>
              <p className="v4-branch-desc">{desc}</p>
              <p className="v4-branch-ids">
                <strong>layer ids:</strong> {formatLayerIds(decoderIds, mtpIds, numLayers)}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Compress a sorted ascending list of layer ids into ranges (e.g. "0-1, 3, 5-9").
 *  MTP layer ids (>= numLayers) get a "(MTP)" suffix for clarity. */
function formatLayerIds(decoderIds: number[], mtpIds: number[], numLayers: number): string {
  const formatRanges = (ids: number[]): string => {
    if (ids.length === 0) return "";
    const ranges: string[] = [];
    let start = ids[0];
    let prev = ids[0];
    for (let i = 1; i < ids.length; i++) {
      if (ids[i] === prev + 1) {
        prev = ids[i];
      } else {
        ranges.push(start === prev ? `${start}` : `${start}–${prev}`);
        start = ids[i];
        prev = ids[i];
      }
    }
    ranges.push(start === prev ? `${start}` : `${start}–${prev}`);
    return ranges.join(", ");
  };

  const parts: string[] = [];
  if (decoderIds.length > 0) parts.push(formatRanges(decoderIds));
  if (mtpIds.length > 0) {
    const mtpRel = mtpIds.map((i) => i - numLayers);
    parts.push(`MTP layer ${formatRanges(mtpRel)}`);
  }
  return parts.join("; ");
}

function ConfigGroup({ title, entries, editValue, onEdit }: ConfigGroupProps) {
  return (
    <div className="config-group">
      <h4>{title}</h4>
      <dl>
        {entries.map((entry) => {
          const [key, value, meta] = entry;
          return (
            <div key={key} className="config-row">
              <dt>{key}</dt>
              <dd>
                {meta?.editable && onEdit ? (
                  <input
                    type="number"
                    value={editValue}
                    min={1}
                    onChange={(e) => {
                      const n = Number(e.target.value);
                      if (Number.isInteger(n) && n > 0) onEdit(n);
                    }}
                  />
                ) : (
                  String(value)
                )}
              </dd>
            </div>
          );
        })}
      </dl>
    </div>
  );
}
