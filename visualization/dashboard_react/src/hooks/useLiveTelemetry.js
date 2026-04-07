/**
 * src/hooks/useLiveTelemetry.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Project-GP Dashboard — Live Telemetry WebSocket Hook
 *
 * Connects to ws_bridge.py and provides real-time physics data to all modules.
 * Replaces seeded RNG generators when mode === "LIVE".
 *
 * Usage in any module:
 *
 *   import { useLiveTelemetry } from "../hooks/useLiveTelemetry.js";
 *
 *   function MyModule({ mode }) {
 *     const { frame, history, connected, stats } = useLiveTelemetry(mode);
 *
 *     // frame    = latest telemetry frame (null if not connected)
 *     // history  = last N frames as an array (for time-series charts)
 *     // connected = boolean
 *     // stats    = { fps, latency_ms }
 *
 *     if (mode === "LIVE" && frame) {
 *       return <div>Speed: {frame.speed_kmh.toFixed(1)} km/h</div>;
 *     }
 *     // else: fall back to existing synthetic data
 *   }
 *
 * Architecture:
 *   ws_bridge.py (Python) ──WebSocket JSON──→ useLiveTelemetry (React)
 *                                              ├── frame (latest)
 *                                              ├── history (ring buffer)
 *                                              └── sendCommand(cmd)
 */

import { useState, useEffect, useRef, useCallback } from "react";

// ── Configuration ───────────────────────────────────────────────────────────

const WS_URL = (() => {
  // Auto-detect: in dev (localhost:5173), connect to ws://localhost:8765
  // In production (Vercel), the WS bridge won't be available — graceful fallback
  const host = typeof window !== "undefined" ? window.location.hostname : "localhost";
  const port = 8765;
  return `ws://${host}:${port}`;
})();

const HISTORY_SIZE    = 600;   // ~10 seconds at 60 Hz
const RECONNECT_DELAY = 3000;  // ms between reconnection attempts
const MAX_RECONNECTS  = 10;

// ── Frame shape (for TypeScript-like documentation) ─────────────────────────
// Each frame from ws_bridge.py contains:
//   frame_id, sim_time,
//   x, y, z, roll, pitch, yaw,
//   vx, vy, vz, ax, ay, az, wz,
//   z_fl, z_fr, z_rl, z_rr,
//   Fz_fl, Fz_fr, Fz_rl, Fz_rr,
//   Fy_fl, Fy_fr, Fy_rl, Fy_rr,
//   slip_fl, slip_fr, slip_rl, slip_rr,
//   kappa_rl, kappa_rr,
//   omega_fl, omega_fr, omega_rl, omega_rr,
//   T_fl, T_fr, T_rl, T_rr,
//   delta, throttle, brake_norm,
//   speed_kmh, lat_g, lon_g,
//   grip_f, grip_r,
//   energy_kj, downforce, drag,
//   lap_time, lap_number, sector

// ── Hook ────────────────────────────────────────────────────────────────────

export function useLiveTelemetry(mode) {
  const [frame, setFrame]         = useState(null);
  const [connected, setConnected] = useState(false);
  const [stats, setStats]         = useState({ fps: 0, latency_ms: 0 });

  const historyRef    = useRef([]);
  const wsRef         = useRef(null);
  const reconnectRef  = useRef(0);
  const frameCountRef = useRef(0);
  const lastStatRef   = useRef(Date.now());

  // Stable reference for history (avoids re-render on every frame)
  const getHistory = useCallback(() => historyRef.current, []);

  // ── Send command to physics server via ws_bridge ──────────────────────

  const sendCommand = useCallback((cmd) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(cmd));
    }
  }, []);

  // Convenience methods
  const resetCar   = useCallback(() => sendCommand({ type: "reset" }),  [sendCommand]);
  const pauseSim   = useCallback(() => sendCommand({ type: "pause" }),  [sendCommand]);
  const resumeSim  = useCallback(() => sendCommand({ type: "resume" }), [sendCommand]);

  const applySetup = useCallback((preset) => {
    sendCommand({ type: "setup", preset });
  }, [sendCommand]);

  const sendControl = useCallback((steer, throttle, brake) => {
    sendCommand({ type: "control", steer, throttle, brake });
  }, [sendCommand]);

  // ── WebSocket lifecycle ───────────────────────────────────────────────

  useEffect(() => {
    if (mode !== "LIVE") {
      // Clean up if switching away from LIVE mode
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setConnected(false);
      return;
    }

    let isMounted = true;

    function connect() {
      if (!isMounted) return;
      if (reconnectRef.current >= MAX_RECONNECTS) {
        console.warn("[LiveTelemetry] Max reconnection attempts reached.");
        return;
      }

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!isMounted) return;
        console.log("[LiveTelemetry] Connected to", WS_URL);
        setConnected(true);
        reconnectRef.current = 0;
      };

      ws.onmessage = (event) => {
        if (!isMounted) return;

        try {
          const f = JSON.parse(event.data);

          // Update latest frame (triggers re-render)
          setFrame(f);

          // Append to ring buffer (does NOT trigger re-render)
          const h = historyRef.current;
          h.push(f);
          if (h.length > HISTORY_SIZE) {
            h.splice(0, h.length - HISTORY_SIZE);
          }

          // FPS counter
          frameCountRef.current++;
          const now = Date.now();
          const elapsed = now - lastStatRef.current;
          if (elapsed > 2000) {
            setStats({
              fps: Math.round((frameCountRef.current / elapsed) * 1000),
              latency_ms: 0, // could add round-trip measurement
            });
            frameCountRef.current = 0;
            lastStatRef.current = now;
          }
        } catch (e) {
          // Malformed JSON — skip
        }
      };

      ws.onclose = () => {
        if (!isMounted) return;
        setConnected(false);
        reconnectRef.current++;
        console.log(
          `[LiveTelemetry] Disconnected. Reconnecting in ${RECONNECT_DELAY}ms ` +
          `(attempt ${reconnectRef.current}/${MAX_RECONNECTS})...`
        );
        setTimeout(connect, RECONNECT_DELAY);
      };

      ws.onerror = (err) => {
        console.error("[LiveTelemetry] WebSocket error:", err);
        ws.close();
      };
    }

    connect();

    return () => {
      isMounted = false;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [mode]);

  return {
    frame,
    history: historyRef.current,  // direct ref for perf — call getHistory() in callbacks
    getHistory,
    connected,
    stats,
    sendCommand,
    sendControl,
    resetCar,
    pauseSim,
    resumeSim,
    applySetup,
  };
}


// ── Helper: Convert live frame to module-specific data shapes ────────────────
// These adapters let existing modules consume live data without rewriting
// their chart/KPI logic.

/**
 * Adapt a live frame to the shape expected by TelemetryModule's track data.
 */
export function frameToTrackPoint(frame) {
  if (!frame) return null;
  return {
    x: frame.x,
    y: frame.y,
    curvature: 0, // derived from yaw rate / speed
    speed: frame.speed_kmh / 3.6,
    s: frame.sim_time * (frame.speed_kmh / 3.6), // approximate arc-length
  };
}

/**
 * Adapt a live frame to the shape expected by SuspensionModule.
 */
export function frameToSuspData(frame) {
  if (!frame) return null;
  return {
    t:    frame.sim_time,
    steer: frame.delta,
    spd:  frame.speed_kmh / 3.6,
    ax:   frame.ax / 9.81,
    ay:   frame.ay / 9.81,
    zFL:  frame.z_fl * 1000,  // mm
    zFR:  frame.z_fr * 1000,
    zRL:  frame.z_rl * 1000,
    zRR:  frame.z_rr * 1000,
    roll: frame.roll * (180 / Math.PI),
    pitch: frame.pitch * (180 / Math.PI),
  };
}

/**
 * Adapt a live frame to tire physics data.
 */
export function frameToTireData(frame) {
  if (!frame) return null;
  return {
    T_fl: frame.T_fl,
    T_fr: frame.T_fr,
    T_rl: frame.T_rl,
    T_rr: frame.T_rr,
    Fz_fl: frame.Fz_fl,
    Fz_fr: frame.Fz_fr,
    Fz_rl: frame.Fz_rl,
    Fz_rr: frame.Fz_rr,
    slip_fl: frame.slip_fl,
    slip_fr: frame.slip_fr,
    slip_rl: frame.slip_rl,
    slip_rr: frame.slip_rr,
    kappa_rl: frame.kappa_rl,
    kappa_rr: frame.kappa_rr,
  };
}

export default useLiveTelemetry;