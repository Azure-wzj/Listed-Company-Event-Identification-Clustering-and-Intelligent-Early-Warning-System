import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 后端地址可通过环境变量覆盖：VITE_API_TARGET=http://localhost:8000 npm run dev
const apiTarget = process.env.VITE_API_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [vue()],
  // 构建时间戳：界面上展示，便于确认浏览器加载的是不是最新版本
  define: {
    __BUILD_TIME__: JSON.stringify(
      new Date().toLocaleString('zh-CN', { hour12: false }),
    ),
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
