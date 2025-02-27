import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite';
import svgr from 'vite-plugin-svgr';
import tsconfigPaths from 'vite-tsconfig-paths';

// https://vite.dev/config/
export default defineConfig({
  server: {
    allowedHosts: true,
    port: 3000,
  },
  plugins: [
    react(),
    tailwindcss() as any,
    svgr(),
    tsconfigPaths(),
  ],
})
