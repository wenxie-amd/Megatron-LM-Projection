/**
 * Pyodide bridge. Loads Pyodide + the `projection` wheel lazily on first call.
 *
 * Why lazy: Pyodide is ~10MB compressed and slows first paint to ~1s. We don't
 * pay that cost until the user actually needs a projection.
 */

import type { DerivedConfig, ProjectionInput, ProjectionOutput, ValidationResult } from "./types";

const PYODIDE_VERSION = "0.27.7";
const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const WHEEL_PATH = "wheels/projection-0.1.0-py3-none-any.whl";

type PyProxy = { toJs(opts?: { dict_converter: unknown }): unknown; destroy(): void };
type PyodideRuntime = {
  loadPackage(names: string[]): Promise<void>;
  pyimport(modulePath: string): PyProxy & Record<string, unknown>;
  toPy(value: unknown): PyProxy;
  runPythonAsync(code: string): Promise<unknown>;
};
type LoadPyodide = (opts: { indexURL: string }) => Promise<PyodideRuntime>;

declare global {
  interface Window {
    loadPyodide?: LoadPyodide;
  }
}

let pyodidePromise: Promise<PyodideRuntime> | null = null;

export interface BridgeOptions {
  /** Base URL where the projection wheel is served. Defaults to Vite's import.meta.env.BASE_URL. */
  baseUrl?: string;
}

export async function getBridge(options: BridgeOptions = {}): Promise<Bridge> {
  if (!pyodidePromise) {
    pyodidePromise = bootstrapPyodide(options);
  }
  return new Bridge(await pyodidePromise);
}

async function bootstrapPyodide(options: BridgeOptions): Promise<PyodideRuntime> {
  const baseUrl = options.baseUrl ?? (import.meta.env.BASE_URL || "/");
  await loadPyodideScript();
  if (!window.loadPyodide) {
    throw new Error("Pyodide failed to expose loadPyodide on window");
  }
  const pyodide = await window.loadPyodide({ indexURL: PYODIDE_CDN });
  await pyodide.loadPackage(["micropip", "pyyaml", "pydantic"]);
  let wheelUrl = new URL(WHEEL_PATH, new URL(baseUrl, window.location.origin)).toString();
  if (import.meta.env.DEV) {
    // In dev, the wheel is rebuilt every time we run `npm run dev`. Bust the
    // URL cache so Pyodide / micropip don't reuse stale bytes.
    wheelUrl += `?t=${Date.now()}`;
  }
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("${wheelUrl}")
import projection.api
  `);
  return pyodide;
}

async function loadPyodideScript(): Promise<void> {
  if (typeof window === "undefined") {
    throw new Error("Pyodide bridge requires a browser-like environment");
  }
  if (window.loadPyodide) return;
  await new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `${PYODIDE_CDN}pyodide.js`;
    script.onload = () => resolve();
    script.onerror = () =>
      reject(new Error(`Failed to load Pyodide from ${script.src}`));
    document.head.appendChild(script);
  });
}

export class Bridge {
  private readonly pyodide: PyodideRuntime;

  constructor(pyodide: PyodideRuntime) {
    this.pyodide = pyodide;
  }

  listModels(): string[] {
    return this.callPyFunction<string[]>("list_models");
  }

  listGpus(): string[] {
    return this.callPyFunction<string[]>("list_gpus");
  }

  getModelConfig(name: string): Record<string, unknown> {
    return this.callPyFunction<Record<string, unknown>>("get_model_config", [name]);
  }

  getGpuSpec(name: string): Record<string, unknown> {
    return this.callPyFunction<Record<string, unknown>>("get_gpu_spec", [name]);
  }

  getModelBreakdown(model: string | Record<string, unknown>): {
    param_count: number;
    param_breakdown: { name: string; count: number }[];
    ffn_breakdown: {
      kind: "mlp" | "moe";
      entries: { name: string; count: number }[];
    };
  } {
    return this.callPyFunction("get_model_breakdown", [model]);
  }

  runProjection(input: ProjectionInput): ProjectionOutput {
    return this.callPyFunction<ProjectionOutput>("run_projection", [input]);
  }

  generateScript(input: ProjectionInput & { num_gpus?: number; nproc_per_node?: number }): string {
    return this.callPyFunction<string>("generate_script", [input]);
  }

  computeDerived(input: { parallel: Record<string, unknown>; workload: Record<string, unknown> }): DerivedConfig {
    return this.callPyFunction<DerivedConfig>("compute_derived", [input]);
  }

  computePerRankLayers(input: {
    model: string | Record<string, unknown>;
    parallel: Record<string, unknown>;
    workload: Record<string, unknown>;
    pp_rank: number;
  }): {
    pp_rank: number;
    total_num_layers: number;
    num_chunks_per_rank: number;
    total_recompute_num_layers: number;
  } {
    return this.callPyFunction("compute_per_rank_layers", [input]);
  }

  validateConfig(input: ProjectionInput): ValidationResult {
    return this.callPyFunction<ValidationResult>("validate_config", [input]);
  }

  private callPyFunction<T>(name: string, args: unknown[] = []): T {
    const module = this.pyodide.pyimport("projection.api");
    const fn = module[name] as unknown as ((...args: unknown[]) => PyProxy) | undefined;
    if (typeof fn !== "function") {
      module.destroy();
      throw new Error(`projection.api.${name} is not callable`);
    }
    const pyArgs = args.map((a) => (isPlainJsValue(a) ? this.pyodide.toPy(a) : a));
    let pyResult: PyProxy | unknown;
    try {
      pyResult = fn.apply(module, pyArgs);
    } finally {
      for (const pa of pyArgs) {
        if (pa && typeof (pa as PyProxy).destroy === "function") {
          (pa as PyProxy).destroy();
        }
      }
      module.destroy();
    }
    if (pyResult && typeof (pyResult as PyProxy).toJs === "function") {
      const proxy = pyResult as PyProxy;
      const jsValue = proxy.toJs({ dict_converter: Object.fromEntries });
      proxy.destroy();
      return jsValue as T;
    }
    return pyResult as T;
  }
}

function isPlainJsValue(v: unknown): boolean {
  return (
    v === null ||
    typeof v === "string" ||
    typeof v === "number" ||
    typeof v === "boolean" ||
    Array.isArray(v) ||
    (typeof v === "object" && v.constructor === Object)
  );
}
