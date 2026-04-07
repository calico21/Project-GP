#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Project-GP Simulator v2 → v3 Migration
# Run from project root: bash simulator/migrate_v3.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASH_DIR="$(cd "$SIM_DIR/../visualization/dashboard_react/src" && pwd)"

echo "══════════════════════════════════════════════════"
echo " Project-GP Simulator v2 → v3 Migration"
echo "══════════════════════════════════════════════════"

# ── 1. Backup existing files ─────────────────────────────────────────────
echo ""
echo "→ Step 1: Backing up existing files..."
for f in physics_server.py sim_protocol.py; do
    if [ -f "$SIM_DIR/$f" ]; then
        cp "$SIM_DIR/$f" "$SIM_DIR/${f}.v2.bak"
        echo "  ✓ Backed up $f → ${f}.v2.bak"
    fi
done

# ── 2. New files to ADD (from Batch 1 + 2) ───────────────────────────────
echo ""
echo "→ Step 2: New files to add to simulator/:"
echo "  ADD  sim_config.py           — centralised configuration"
echo "  ADD  ws_bridge.py            — WebSocket bridge for dashboard"
echo "  ADD  ros2_bridge.py          — ROS 2 bridge for driverless team"
echo ""
echo "  New file for dashboard:"
echo "  ADD  src/hooks/useLiveTelemetry.js — React WebSocket hook"

# ── 3. Files to REPLACE ──────────────────────────────────────────────────
echo ""
echo "→ Step 3: Files to REPLACE:"
echo "  REPLACE  physics_server.py   — v3: 28-dim setup, fixed ABS, named indices"
echo "  REPLACE  README.md           — v3: new architecture docs"

# ── 4. Files to DELETE (legacy v1) ───────────────────────────────────────
echo ""
echo "→ Step 4: Legacy files to DELETE:"
LEGACY_FILES=(
    "udp_server.py"       # v1 physics server (11-float protocol)
    "dummy_client.py"     # v1 test client (incompatible protocol)
    "SIM_ROADMAP.md"      # UE5 fantasy — replaced by README.md
    "patch_install.py"    # one-time patches already applied
)
for f in "${LEGACY_FILES[@]}"; do
    if [ -f "$SIM_DIR/$f" ]; then
        echo "  DELETE  $f"
    else
        echo "  (skip)  $f — already absent"
    fi
done

# ── 5. Install dependencies ──────────────────────────────────────────────
echo ""
echo "→ Step 5: Install WebSocket dependency:"
echo "  pip install websockets --break-system-packages"

# ── 6. Dashboard hook installation ───────────────────────────────────────
echo ""
echo "→ Step 6: Install dashboard WebSocket hook:"
echo "  mkdir -p $DASH_DIR/hooks/"
echo "  cp useLiveTelemetry.js $DASH_DIR/hooks/"

# ── 7. Quick verification ────────────────────────────────────────────────
echo ""
echo "→ Step 7: Verify (after copying files):"
echo "  python -c \"from simulator.sim_config import S, DEFAULT_SETUP_28; print(f'Setup dim: {len(DEFAULT_SETUP_28)}'); print(f'VX index: {S.VX}')\""
echo ""
echo "══════════════════════════════════════════════════"
echo " File Summary"
echo "══════════════════════════════════════════════════"
echo "  ADD:     sim_config.py, ws_bridge.py, ros2_bridge.py, useLiveTelemetry.js"
echo "  REPLACE: physics_server.py, README.md"
echo "  DELETE:  udp_server.py, dummy_client.py, SIM_ROADMAP.md, patch_install.py"
echo ""
echo " After migration, run the full stack:"
echo "   Terminal 1:  python simulator/physics_server.py --track fsg_autocross"
echo "   Terminal 2:  python simulator/ws_bridge.py"
echo "   Terminal 3:  python simulator/control_interface.py --mode keyboard"
echo "   Browser:     Open dashboard → toggle LIVE mode"
echo "══════════════════════════════════════════════════"