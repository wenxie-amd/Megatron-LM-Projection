/**
 * Bridge contract test (Node, no browser, no Pyodide).
 *
 * What this test guards against:
 * 1. The wheel artifact actually gets built into `web/public/wheels/`.
 * 2. The TS types align with the fixture JSON.
 *
 * The full Pyodide-in-browser round-trip is verified by Playwright in M3.
 */

import { existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import type { ProjectionOutput } from "./types";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..", "..");
const wheelsDir = join(repoRoot, "web", "public", "wheels");
const fixturePath = join(
  repoRoot,
  "projection",
  "tests",
  "fixtures",
  "llama3_1_8b",
  "default_bf16_tp1_pp1.json",
);

describe("bridge artifact", () => {
  it("has a projection wheel ready for Pyodide", () => {
    expect(existsSync(wheelsDir)).toBe(true);
    const wheels = readdirSync(wheelsDir).filter((f) => f.endsWith(".whl"));
    expect(wheels.length).toBeGreaterThan(0);
    expect(wheels[0]).toMatch(/^projection-.*-py3-none-any\.whl$/);
  });
});

describe("ProjectionOutput type aligns with fixture", () => {
  it("can describe a synthetic output that matches the fixture shape", () => {
    const fixture = JSON.parse(readFileSync(fixturePath, "utf-8"));
    const expected = fixture.expected.rank_0_memory_bytes;

    const synthetic: ProjectionOutput = {
      model_config: { name: fixture.model },
      parallel: {
        precision: fixture.parallel.precision,
        tensor_model_parallel_size: fixture.parallel.tensor_model_parallel_size,
        sequence_parallel: fixture.parallel.sequence_parallel,
        pipeline_model_parallel_size: fixture.parallel.pipeline_model_parallel_size,
        context_parallel_size: fixture.parallel.context_parallel_size,
        expert_model_parallel_size: 1,
        data_parallel_size: fixture.parallel.data_parallel_size,
        optimizer_kind: fixture.parallel.optimizer_kind,
      },
      workload: fixture.workload,
      rank_reports: [
        {
          global_rank: 0,
          rank_coord: { tp: 0, cp: 0, ep: 0, dp: 0, pp: 0 },
          partition: {
            num_layers_on_rank: 32,
            has_embedding: true,
            has_final_norm: true,
            has_output_projection: true,
          },
          param_count: fixture.expected.param_count_total,
          param_breakdown: Object.entries(fixture.expected.param_breakdown).map(
            ([name, count]) => ({ name, count: count as number }),
          ),
          memory: {
            param_bytes: expected.param_bytes,
            activation_bytes: expected.activation_bytes,
            optimizer: {
              grad_buffer_bytes: expected.optimizer_grad_buffer_bytes,
              main_param_bytes: expected.optimizer_main_param_bytes,
              state_bytes: expected.optimizer_state_bytes,
              total_bytes:
                expected.optimizer_grad_buffer_bytes +
                expected.optimizer_main_param_bytes +
                expected.optimizer_state_bytes,
            },
            total_bytes: expected.total_bytes,
            precision: fixture.parallel.precision,
          },
        },
      ],
    };

    expect(synthetic.rank_reports[0].param_count).toBe(8_030_261_248);
    expect(synthetic.rank_reports[0].memory.total_bytes).toBe(expected.total_bytes);
  });
});
