// Rebuilds the projection Python wheel and copies it into web/public/wheels/.
// Pyodide installs from this URL at runtime via micropip.

import { execSync } from "node:child_process";
import { mkdirSync, readdirSync, copyFileSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const pythonDir = join(repoRoot, "projection");
const distDir = join(pythonDir, "dist");
const targetDir = join(repoRoot, "web", "public", "wheels");

mkdirSync(distDir, { recursive: true });
mkdirSync(targetDir, { recursive: true });

console.log("[build-wheel] running `uv build --wheel` in", pythonDir);
execSync("uv build --wheel", { cwd: pythonDir, stdio: "inherit" });

for (const f of readdirSync(targetDir)) {
  if (f.endsWith(".whl")) rmSync(join(targetDir, f));
}
for (const f of readdirSync(distDir)) {
  if (f.endsWith(".whl")) copyFileSync(join(distDir, f), join(targetDir, f));
}
console.log("[build-wheel] copied wheels into", targetDir);
