import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发期：Vite dev server 转发 API / WebSocket 到后端（uvicorn 默认 8000）
// 构建产物交由 FastAPI StaticFiles 托管（见 tunefield/serve/，F1 交付）
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true, // 覆盖 /api/events WebSocket
      },
      "/v1": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
