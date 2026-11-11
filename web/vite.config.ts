import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development the front end runs on port 5173 and forwards /api to the
// Python back end on port 8000. In production FastAPI serves the built files itself.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist", sourcemap: false },
});
