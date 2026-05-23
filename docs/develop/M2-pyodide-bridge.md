# M2 — Pyodide bridge

## What landed

- `projection/src/projection/api.py`: the stable JSON-in / JSON-out surface consumed by the bridge. Functions: `list_models`, `list_gpus`, `get_model_config`, `get_gpu_spec`, `get_model_breakdown`, `run_projection`.
- `web/src/pyodide/bridge.ts`: lazy loader that fetches `pyodide.js` from jsdelivr, loads pyyaml + pydantic + micropip, installs our `projection-*.whl`, and exposes a typed JS facade (`Bridge.runProjection`, `Bridge.getModelConfig`, etc.).
- `web/src/pyodide/types.ts`: TypeScript mirror of the Python API contract (`ProjectionInput`, `ProjectionOutput`, `RankReport`, …).
- `web/scripts/build-wheel.mjs` (added in M0) ensures the wheel is always fresh in `web/public/wheels/`.
- `web/src/pyodide/bridge.test.ts`: Vitest unit test that asserts (a) the wheel artifact lives in `web/public/wheels/`, (b) the TS types align with the fixture JSON.

## Deferred to M3+

- Full in-browser Playwright smoke test — added when the actual UI exists to drive. M3+ UI work piggy-backs on the same bridge and exercises it end-to-end during manual click-through.

## Bridge contract example

The TS side calls `bridge.runProjection(input)` and receives a typed `ProjectionOutput`. Numeric overflow isn't a concern: param counts and byte counts both fit comfortably under `2^53`.

## Decisions worth remembering

- Pyodide is loaded from CDN (smaller deploy footprint, faster initial visit).
- Dict→object conversion uses `Object.fromEntries` (Pyodide canonical pattern).
- Bridge is a singleton (`pyodidePromise` cache) — first call pays the load cost, subsequent calls are instant.
