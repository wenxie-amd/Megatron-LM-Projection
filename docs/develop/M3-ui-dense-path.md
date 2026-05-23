# M3 — Steps 1–4 UI (dense path)

## What landed

- `web/src/state/store.ts` + `web/src/state/context.tsx`: a single `useReducer` store that owns the whole wizard state (model selection, GPU selection, parallel config, workload, ranks, projection result). The Pyodide bridge is the only side-effectful piece, lifted into a context provider.
- `web/src/components/`:
  - `StepNav.tsx` — top-of-page step tabs.
  - `Field.tsx` — `NumberField`, `TextField`, `SelectField`, `CheckboxField` primitives.
  - `ParamPie.tsx` — pie chart of per-module parameter counts (Recharts).
  - `ModelStructure.tsx` — text-level structure render (full model + single layer).
  - `RankBars.tsx` — stacked bar chart of per-rank memory components.
- `web/src/steps/`:
  - `Step1Model.tsx` — model dropdown, grouped YAML display, `num_layers` editable, proxy badge, pie chart, structure render.
  - `Step2Machine.tsx` — vendor + GPU dropdown, #GPUs, primary + optional secondary spec table.
  - `Step3Training.tsx` — distributed strategy (TP/SP/PP/CP/DP/EP, PP-layout-vs-PP+VPP toggle with divisibility validators), Workload section, Hyperparameters placeholder.
  - `Step4Memory.tsx` — rank list parser (≤8), `Run projection` button, stacked bar chart + per-rank breakdown tables.
  - `Step5Script.tsx` — placeholder (filled in M5).
- More GPU YAMLs: `h200`, `b200`, `mi300x`, `mi325x`.
- Test coverage:
  - `web/src/state/store.test.ts` — `parseRankList`, `parseLayoutList` edge cases.
  - Python pytest still drives the actual memory math; the UI just renders it.

## Decisions worth remembering

- State lives in one reducer to keep step-to-step data flow obvious. Steps read state, dispatch actions. The bridge call is in the context provider's `runProjection` method.
- `proxy` label shows the moment `num_layers` deviates from the canonical value.
- PP/VPP validators run client-side and surface red hints inline; the same validation also runs on the Python side via `validate_parallel_config` so the bridge call still raises if the UI was bypassed.
- Recharts handles both charts. The chunk-size warning during build is recharts; acceptable for v1.

## Deferred to M5

- FSDP option selectability — initially disabled in the dropdown; turned on in M5 with conflict detection.
- Info icons on individual fields — added in M5.
