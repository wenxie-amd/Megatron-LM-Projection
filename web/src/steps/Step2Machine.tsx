import { useMemo, useState } from "react";

import { NumberField } from "../components/Field";
import { useProjection } from "../state/context";
import type { GpuSpecView } from "../state/store";

type Vendor = "all" | "nvidia" | "amd";

export function Step2Machine() {
  const { state, dispatch, loadPrimaryGpu, loadSecondaryGpu } = useProjection();
  const [vendor, setVendor] = useState<Vendor>("all");

  const filteredGpus = useMemo(() => {
    if (vendor === "all") return state.availableGpus;
    return state.availableGpus.filter((name) => {
      if (state.primaryGpu && state.primaryGpu.name === name) return state.primaryGpu.vendor === vendor;
      // Cheap heuristic: vendor is encoded in the prefix (h/b/gb → nvidia, mi → amd)
      const lower = name.toLowerCase();
      const isAmd = lower.startsWith("mi");
      return vendor === "amd" ? isAmd : !isAmd;
    });
  }, [state.availableGpus, state.primaryGpu, vendor]);

  return (
    <section className="step">
      <h2>Step 2 · Machine Selection</h2>

      <label className="field">
        <span className="field-label">Vendor</span>
        <select value={vendor} onChange={(e) => setVendor(e.target.value as Vendor)}>
          <option value="all">All</option>
          <option value="nvidia">NVIDIA</option>
          <option value="amd">AMD</option>
        </select>
      </label>

      <label className="field">
        <span className="field-label">Primary GPU</span>
        <select
          value={state.primaryGpuName ?? ""}
          onChange={(e) => loadPrimaryGpu(e.target.value)}
        >
          <option value="" disabled>
            — choose a GPU —
          </option>
          {filteredGpus.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>

      <NumberField
        label="Number of GPUs (world size)"
        value={state.numGpus}
        min={1}
        onChange={(v) => dispatch({ type: "SET_NUM_GPUS", value: Math.max(1, v) })}
        hint="Total number of GPUs across all nodes. DP is derived from this in Step 3."
      />

      <label className="field">
        <span className="field-label">Compare with (optional)</span>
        <select
          value={state.secondaryGpuName ?? ""}
          onChange={(e) => loadSecondaryGpu(e.target.value || null)}
        >
          <option value="">— none —</option>
          {state.availableGpus
            .filter((n) => n !== state.primaryGpuName)
            .map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
        </select>
      </label>

      {state.primaryGpu && (
        <GpuSpecTable primary={state.primaryGpu} secondary={state.secondaryGpu} />
      )}
    </section>
  );
}

interface SpecRowKV {
  key: keyof GpuSpecView;
  label: string;
  unit?: string;
}

const SPEC_ROWS: SpecRowKV[] = [
  { key: "vendor", label: "Vendor" },
  { key: "memory_gb", label: "Memory", unit: "GB" },
  { key: "bf16_tflops", label: "BF16", unit: "TFLOPS" },
  { key: "fp8_tflops", label: "FP8", unit: "TFLOPS" },
  { key: "bandwidth_gbps", label: "HBM bandwidth", unit: "GB/s" },
];

function GpuSpecTable({ primary, secondary }: { primary: GpuSpecView; secondary: GpuSpecView | null }) {
  return (
    <table className="spec-table">
      <thead>
        <tr>
          <th></th>
          <th>{primary.name}</th>
          {secondary && <th>{secondary.name}</th>}
        </tr>
      </thead>
      <tbody>
        {SPEC_ROWS.map((row) => (
          <tr key={row.key}>
            <th>{row.label}</th>
            <td>
              {String(primary[row.key])}
              {row.unit ? ` ${row.unit}` : ""}
            </td>
            {secondary && (
              <td>
                {String(secondary[row.key])}
                {row.unit ? ` ${row.unit}` : ""}
              </td>
            )}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
