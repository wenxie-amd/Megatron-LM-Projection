import { useEffect, useState } from "react";

import { ModelStructure } from "../components/ModelStructure";
import { ParamPie } from "../components/ParamPie";
import { getBridge } from "../pyodide/bridge";
import { useProjection } from "../state/context";
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
              title="Attention"
              entries={
                config.attention.use_mla
                  ? [
                      ["num_attention_heads", config.attention.num_attention_heads],
                      ["use_mla", true],
                      ["q_lora_rank", config.attention.q_lora_rank ?? "—"],
                      ["kv_lora_rank", config.attention.kv_lora_rank ?? "—"],
                      ["qk_nope_head_dim", config.attention.qk_nope_head_dim ?? "—"],
                      ["qk_rope_head_dim", config.attention.qk_rope_head_dim ?? "—"],
                      ["v_head_dim", config.attention.v_head_dim ?? "—"],
                      ["attention_dropout", config.attention.attention_dropout],
                    ]
                  : [
                      ["num_attention_heads", config.attention.num_attention_heads],
                      ["num_query_groups", config.attention.num_query_groups ?? "—"],
                      ["kv_channels", config.attention.kv_channels ?? "—"],
                      ["attention_dropout", config.attention.attention_dropout],
                      ["add_qkv_bias", config.attention.add_qkv_bias],
                    ]
              }
            />
            <ConfigGroup
              title="MLP"
              entries={[
                ["swiglu", config.mlp.swiglu],
                ["add_bias_linear", config.mlp.add_bias_linear],
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
                  : "Inside one MLP block"}
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
