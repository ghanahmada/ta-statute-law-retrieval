import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: '../app/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/auth': 'http://localhost:8000',
      '/pairs': 'http://localhost:8000',
      '/admin': 'http://localhost:8000',
    },
  },
})
