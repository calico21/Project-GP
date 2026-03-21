#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Project-GP — React Dashboard Integration Script
# Run from your project root: bash visualization/dashboard_react/setup.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DASH_DIR="$SCRIPT_DIR"

echo "══════════════════════════════════════════════════"
echo " Project-GP — React Dashboard Setup"
echo " Project root: $PROJECT_ROOT"
echo " Dashboard dir: $DASH_DIR"
echo "══════════════════════════════════════════════════"

# ── 1. Check Node.js ─────────────────────────────────────────────────────
if ! command -v node &>/dev/null; then
    echo "❌ Node.js not found. Install v18+ from https://nodejs.org"
    exit 1
fi
NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VER" -lt 18 ]; then
    echo "❌ Node.js v18+ required (found v$(node -v))"
    exit 1
fi
echo "✓ Node.js $(node -v)"

# ── 2. Initialize package.json if missing ────────────────────────────────
cd "$DASH_DIR"
if [ ! -f package.json ]; then
    echo "→ Initializing package.json..."
    cat > package.json << 'PKGJSON'
{
  "name": "project-gp-dashboard",
  "version": "3.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --host 0.0.0.0 --port 3000",
    "build": "vite build",
    "preview": "vite preview --host 0.0.0.0 --port 3000"
  }
}
PKGJSON
fi

# ── 3. Install dependencies ──────────────────────────────────────────────
echo "→ Installing dependencies..."
npm install \
    react@^18 react-dom@^18 \
    recharts@^2 \
    three@^0.160 \
    @vitejs/plugin-react vite \
    --save

# ── 4. Create Vite config ───────────────────────────────────────────────
cat > vite.config.js << 'VITE'
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  root: '.',
  base: './',
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  server: {
    port: 3000,
    open: true,
  },
});
VITE

# ── 5. Create index.html entry point ────────────────────────────────────
cat > index.html << 'HTML'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Project-GP | Engineering Dashboard</title>
  <style>
    body { margin: 0; padding: 0; background: #05070b; overflow: hidden; }
    #root { width: 100vw; height: 100vh; }
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
</html>
HTML

# ── 6. Create src/main.jsx bootstrap ────────────────────────────────────
mkdir -p src
cat > src/main.jsx << 'MAIN'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
MAIN

# ── 7. Create .gitignore ────────────────────────────────────────────────
cat > .gitignore << 'GI'
node_modules/
dist/
.vite/
GI

echo ""
echo "══════════════════════════════════════════════════"
echo " ✅ Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Copy your module files into src/"
echo "      (theme.js, data.js, components.jsx,"
echo "       App.jsx, OverviewModule.jsx, etc.)"
echo ""
echo "   2. Start dev server:"
echo "      cd visualization/dashboard_react"
echo "      npm run dev"
echo ""
echo "   3. Build for production:"
echo "      npm run build"
echo "      (outputs to dist/ — serve with any static server)"
echo "══════════════════════════════════════════════════"
