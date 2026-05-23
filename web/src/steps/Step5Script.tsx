import { useState } from "react";

import { getBridge } from "../pyodide/bridge";
import { useProjection } from "../state/context";

export function Step5Script() {
  const { state } = useProjection();
  const [script, setScript] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [copied, setCopied] = useState(false);

  const generate = async () => {
    if (!state.modelConfig) {
      setError("Pick a model in Step 1 first.");
      return;
    }
    setRunning(true);
    setError(null);
    setCopied(false);
    try {
      const bridge = await getBridge();
      const modelPayload =
        state.numLayersOverride !== null && state.numLayersOverride !== state.modelConfig.architecture.num_layers
          ? {
              ...state.modelConfig,
              architecture: { ...state.modelConfig.architecture, num_layers: state.numLayersOverride },
            }
          : (state.selectedModel ?? state.modelConfig.name);
      const result = bridge.generateScript({
        model: modelPayload,
        parallel: state.parallel,
        workload: state.workload,
        ranks: state.ranks,
        num_gpus: state.numGpus,
      });
      setScript(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const copy = async () => {
    if (!script) return;
    await navigator.clipboard.writeText(script);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  if (state.primaryGpu && state.primaryGpu.vendor === "amd") {
    return (
      <section className="step">
        <h2>Step 5 · Generate Training Script</h2>
        <p className="placeholder">
          AMD launch-script generation (Primus) is out of v1. For an AMD GPU, run the same model config under
          Primus's <code>dev/wenx/deepseek-v4</code> branch using the parameters from Step 3.
        </p>
      </section>
    );
  }

  return (
    <section className="step">
      <h2>Step 5 · Generate Training Script</h2>
      <p className="hint">
        Megatron-LM <code>pretrain_gpt.py</code> launch script. Review and adjust paths
        (<code>--data-path</code>, <code>--save</code>, <code>--load</code>, <code>--tokenizer-model</code>)
        before running.
      </p>
      <div className="actions">
        <button type="button" onClick={generate} disabled={running}>
          {running ? "Generating…" : "Generate script"}
        </button>
        {script && (
          <button type="button" onClick={copy} className="secondary">
            {copied ? "Copied!" : "Copy to clipboard"}
          </button>
        )}
      </div>
      {error && <pre className="error">{error}</pre>}
      {script && (
        <pre className="script">{script}</pre>
      )}
    </section>
  );
}
