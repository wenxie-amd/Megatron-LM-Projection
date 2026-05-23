import { createContext, useContext, useEffect, useReducer, useRef } from "react";
import type { Dispatch, ReactNode } from "react";

import { getBridge } from "../pyodide/bridge";
import type { Bridge } from "../pyodide/bridge";

import {
  INITIAL_STATE,
  type Action,
  type GpuSpecView,
  type ModelConfigView,
  reducer,
  type State,
} from "./store";

interface Ctx {
  state: State;
  dispatch: Dispatch<Action>;
  loadModel: (name: string) => Promise<void>;
  loadPrimaryGpu: (name: string) => Promise<void>;
  loadSecondaryGpu: (name: string | null) => Promise<void>;
  runProjection: () => Promise<void>;
}

const ProjectionCtx = createContext<Ctx | null>(null);

export function ProjectionProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  const bridgeRef = useRef<Bridge | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const bridge = await getBridge();
        if (cancelled) return;
        bridgeRef.current = bridge;
        dispatch({
          type: "BRIDGE_READY",
          models: bridge.listModels(),
          gpus: bridge.listGpus(),
        });
      } catch (err) {
        if (!cancelled) {
          dispatch({
            type: "BRIDGE_ERROR",
            error: err instanceof Error ? err.message : String(err),
          });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const loadModel = async (name: string) => {
    if (!bridgeRef.current) return;
    const config = bridgeRef.current.getModelConfig(name) as unknown as ModelConfigView;
    dispatch({ type: "SELECT_MODEL", name, config });
  };

  const loadPrimaryGpu = async (name: string) => {
    if (!bridgeRef.current) return;
    const spec = bridgeRef.current.getGpuSpec(name) as unknown as GpuSpecView;
    dispatch({ type: "SELECT_GPU_PRIMARY", name, spec });
  };

  const loadSecondaryGpu = async (name: string | null) => {
    if (!bridgeRef.current) {
      dispatch({ type: "SELECT_GPU_SECONDARY", name: null, spec: null });
      return;
    }
    if (name === null) {
      dispatch({ type: "SELECT_GPU_SECONDARY", name: null, spec: null });
      return;
    }
    const spec = bridgeRef.current.getGpuSpec(name) as unknown as GpuSpecView;
    dispatch({ type: "SELECT_GPU_SECONDARY", name, spec });
  };

  const runProjection = async () => {
    if (!bridgeRef.current || !state.modelConfig) return;
    dispatch({ type: "PROJECTION_START" });
    try {
      const modelPayload =
        state.numLayersOverride !== null && state.numLayersOverride !== state.modelConfig.architecture.num_layers
          ? {
              ...state.modelConfig,
              architecture: {
                ...state.modelConfig.architecture,
                num_layers: state.numLayersOverride,
              },
            }
          : (state.selectedModel ?? state.modelConfig.name);

      const output = bridgeRef.current.runProjection({
        model: modelPayload,
        parallel: state.parallel,
        workload: state.workload,
        ranks: state.ranks,
      });
      dispatch({ type: "PROJECTION_SUCCESS", output });
    } catch (err) {
      dispatch({ type: "PROJECTION_ERROR", error: err instanceof Error ? err.message : String(err) });
    }
  };

  const ctx: Ctx = { state, dispatch, loadModel, loadPrimaryGpu, loadSecondaryGpu, runProjection };
  return <ProjectionCtx.Provider value={ctx}>{children}</ProjectionCtx.Provider>;
}

export function useProjection(): Ctx {
  const ctx = useContext(ProjectionCtx);
  if (!ctx) {
    throw new Error("useProjection must be used inside <ProjectionProvider>");
  }
  return ctx;
}
