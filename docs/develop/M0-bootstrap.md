# M0 — Bootstrap

## What landed

- `projection/` Python package, uv-managed (`pyproject.toml` with pyyaml + pydantic + pytest-dev).
- `web/` Vite + React + TypeScript scaffold.
- GitHub Actions:
  - `.github/workflows/ci.yml` runs `uv sync` + `pytest`, then `npm ci` + `vitest` + `npm run build`.
  - `.github/workflows/deploy.yml` builds and publishes `web/dist/` to GitHub Pages on push to `main`.
- `web/scripts/build-wheel.mjs` rebuilds the Python wheel and copies it into `web/public/wheels/` before every web build.
- `web/vite.config.ts` honours `VITE_BASE_PATH` so GH Pages can host at `/Megatron-LM-Projection/` while local dev stays at `/`.

## Decisions worth remembering

- Pyodide is loaded from the official jsdelivr CDN at runtime, not bundled — keeps the deploy lean.
- `web/public/wheels/*.whl` is git-ignored; it's always rebuilt from `projection/`.
- Pre-commit hooks already exclude `third_party/`; we did not extend them for `web/` to keep the repo's existing tooling untouched.

## Verification

- `uv run pytest` in `projection/` passes (2 smoke tests).
- `npm run build` in `web/` produces `dist/index.html` with a "hello" placeholder + `dist/wheels/projection-*.whl`.
