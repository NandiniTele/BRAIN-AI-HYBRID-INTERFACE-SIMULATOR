import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Suppress noisy ECONNREFUSED proxy logs during the ~10s backend startup window
const silenceProxyErrors = (proxy: any) => {
  // 1. Remove listeners attached before configure()
  proxy.removeAllListeners('error');

  // 2. Prevent Vite from attaching its own noisy error listener after configure()
  const originalOn = proxy.on.bind(proxy);
  proxy.on = function (event: string, handler: any) {
    if (event === 'error') return this;
    return originalOn(event, handler);
  };

  // 3. Attach our own silent error handler
  originalOn('error', (err: any, _req: any, res: any, _target?: any) => {
    if (res && typeof res.writeHead === 'function' && !res.headersSent) {
      res.writeHead(503, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Backend proxy error' }));
    }
  });

  // 4. Suppress socket-level errors (e.g. WS proxy ECONNRESET)
  originalOn('proxyReqWs', (_proxyReq: any, _req: any, socket: any) => {
    socket.removeAllListeners('error');
    socket.on('error', () => { /* swallow */ });
  });

  // 5. Suppress response-socket errors (e.g. WS proxy ECONNABORTED on close)
  originalOn('proxyRes', (_proxyRes: any, _req: any, res: any) => {
    if (res?.socket) {
      res.socket.removeAllListeners('error');
      res.socket.on('error', () => { /* swallow */ });
    }
  });
};

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Proxy all REST API calls to the FastAPI backend
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        configure: silenceProxyErrors,
      },
      // Proxy WebSocket connection — timeout:0 disables Vite's ~2-min idle kill
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
        timeout: 0,
        configure: silenceProxyErrors,
      },
    },
  },
})

