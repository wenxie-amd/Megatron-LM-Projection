import { describe, expect, it } from "vitest";

import {
  INITIAL_STATE,
  type State,
  clientValidate,
  deriveView,
  parseLayoutList,
  parseRankList,
  reducer,
  suggestLayout,
  suggestLayoutBalanced,
} from "./store";

describe("parseRankList", () => {
  it("parses comma-separated ranks", () => {
    expect(parseRankList("0, 4, 10, 63", 8)).toEqual({ ranks: [0, 4, 10, 63], error: null });
  });

  it("accepts whitespace separation", () => {
    expect(parseRankList("0  4   10", 8)).toEqual({ ranks: [0, 4, 10], error: null });
  });

  it("rejects non-integers", () => {
    const { error } = parseRankList("0, 4, oops", 8);
    expect(error).toMatch(/'oops'/);
  });

  it("rejects negative integers", () => {
    const { error } = parseRankList("0, -1", 8);
    expect(error).toMatch(/'-1'/);
  });

  it("rejects too many ranks", () => {
    const { error } = parseRankList("0,1,2,3,4,5,6,7,8", 8);
    expect(error).toMatch(/at most 8/);
  });

  it("rejects duplicates", () => {
    const { error } = parseRankList("0, 0, 1", 8);
    expect(error).toMatch(/unique/);
  });
});

describe("parseLayoutList", () => {
  it("parses positive integers", () => {
    expect(parseLayoutList("9, 8, 8, 7")).toEqual([9, 8, 8, 7]);
  });
  it("returns null for non-positive entries", () => {
    expect(parseLayoutList("8, 0, 8, 8")).toBeNull();
  });
  it("returns null for empty input", () => {
    expect(parseLayoutList("")).toBeNull();
  });
});

describe("suggestLayout / suggestLayoutBalanced", () => {
  it("evenly splits when num_layers is divisible", () => {
    expect(suggestLayout(32, 4)).toEqual([8, 8, 8, 8]);
  });
  it("front + back get the +1 surplus", () => {
    expect(suggestLayout(35, 4)).toEqual([9, 9, 8, 9]);
    expect(suggestLayout(34, 4)).toEqual([9, 8, 8, 9]);
    expect(suggestLayout(33, 4)).toEqual([9, 8, 8, 8]);
  });
  it("balanced layout absorbs cheap dense layers onto chunk 0", () => {
    // DSV3-like: 61 layers, 3 dense, 58 MoE, dense/MoE cost ratio ~ 2.76.
    const layout = suggestLayoutBalanced(61, 3, 8, { dense: 1, moe: 2.76 });
    expect(layout.reduce((a, b) => a + b, 0)).toBe(61);
    expect(layout.length).toBe(8);
    // chunk 0 should be larger than the others (absorbs dense + extra MoE).
    expect(layout[0]).toBeGreaterThan(layout[1]);
  });
  it("falls back to simple split for dense models", () => {
    const layout = suggestLayoutBalanced(32, 0, 4, { dense: 1, moe: 1 });
    expect(layout).toEqual([8, 8, 8, 8]);
  });
});

function _withState(patch: Partial<State>): State {
  return { ...INITIAL_STATE, ...patch };
}

describe("deriveView", () => {
  it("computes DP from world_size / (TP*PP*CP)", () => {
    const state = reducer(
      reducer(INITIAL_STATE, { type: "SET_NUM_GPUS", value: 16 }),
      {
        type: "SET_PARALLEL",
        patch: { tensor_model_parallel_size: 2, pipeline_model_parallel_size: 2, context_parallel_size: 1 },
      },
    );
    expect(deriveView(state)).toMatchObject({
      world_size: 16,
      data_parallel_size: 4,
      gradient_accumulation_steps: 16,
    });
  });

  it("reports DP=0 when world_size doesn't cleanly divide", () => {
    const state = reducer(
      reducer(INITIAL_STATE, { type: "SET_NUM_GPUS", value: 7 }),
      { type: "SET_PARALLEL", patch: { tensor_model_parallel_size: 2 } },
    );
    expect(deriveView(state).data_parallel_size).toBe(0);
  });

  it("computes EDP = DP / EP", () => {
    let s = reducer(INITIAL_STATE, { type: "SET_NUM_GPUS", value: 32 });
    s = reducer(s, { type: "SET_PARALLEL", patch: { expert_model_parallel_size: 8 } });
    expect(deriveView(s).expert_data_parallel_size).toBe(32 / 8);
  });
});

describe("clientValidate", () => {
  it("returns nothing for a valid default state", () => {
    const s = reducer(INITIAL_STATE, { type: "SET_NUM_GPUS", value: 8 });
    expect(clientValidate(s)).toEqual([]);
  });

  it("flags world_size not divisible by TP*PP*CP", () => {
    let s = reducer(INITIAL_STATE, { type: "SET_NUM_GPUS", value: 7 });
    s = reducer(s, { type: "SET_PARALLEL", patch: { tensor_model_parallel_size: 2 } });
    expect(clientValidate(s).some((e) => e.includes("not divisible"))).toBe(true);
  });

  it("flags gbs not divisible by dp*mbs", () => {
    let s = reducer(INITIAL_STATE, { type: "SET_NUM_GPUS", value: 8 });
    s = reducer(s, { type: "SET_WORKLOAD", patch: { global_batch_size: 50, micro_batch_size: 3 } });
    expect(clientValidate(s).some((e) => e.includes("global_batch_size=50"))).toBe(true);
  });

  it("flags recompute=full without method or num_layers", () => {
    const s = _withState({
      modelConfig: {
        ...INITIAL_STATE.modelConfig!,
        name: "x",
        architecture: { num_layers: 32 } as never,
      } as never,
      workload: { ...INITIAL_STATE.workload, recompute_granularity: "full" },
    });
    const errors = clientValidate(s);
    expect(errors.some((e) => e.includes("recompute_method"))).toBe(true);
  });

  it("flags torch_fsdp2 + PP>1", () => {
    let s = reducer(INITIAL_STATE, { type: "SET_NUM_GPUS", value: 8 });
    s = reducer(s, {
      type: "SET_PARALLEL",
      patch: { optimizer_kind: "torch_fsdp2", pipeline_model_parallel_size: 2 },
    });
    expect(clientValidate(s).some((e) => e.includes("torch_fsdp2"))).toBe(true);
  });
});
