import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    // Aumenta el límite de advertencia a 2000 KB (2 MB)
    chunkSizeWarningLimit: 2000,
    
    // Opcional: Divide el código para que las librerías pesadas (como React y Recharts) 
    // vayan en un archivo separado de tu código del Project-GP
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            return 'vendor';
          }
        }
      }
    }
  }
})
