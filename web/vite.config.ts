import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `base` is overridden by CI for GitHub Pages (e.g. /Megatron-LM-Projection/).
// Locally (`npm run dev` / `vite build` without env) it defaults to "/".
export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? "/",
  plugins: [react()],
});
