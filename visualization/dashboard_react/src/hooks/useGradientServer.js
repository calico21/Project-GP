/**
 * src/hooks/useGradientServer.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Project-GP Dashboard — Live Gradient Computation Hook
 *
 * Connects to gradient_server.py and provides ∂(lap_time)/∂(setup) data
 * to the DifferentiableInsightsModule and SuspensionExplorerModule.
 *
 * Usage:
 *   import { useGradientServer } from "../hooks/useGradientServer.js";
 *
 *   function MyModule() {
 *     const { computeGradient, gradient, lapTime, loading, connected } = useGradientServer();
 *
 *     // Trigger computation with a setup vector
 *     const handleSliderChange = (setup28) => computeGradient(setup28);
 *
 *     // gradient = { param_name: sensitivity_value, ... }
 *     // lapTime = scalar [s]
 *   }
 */

import { useState, useCallback, useRef, useEffect } from "react";

// ── Configuration ───────────────────────────────────────────────────────────

const GRADIENT_URL = (() => {
  const host = typeof window !== "undefined" ? window.location.hostname : "localhost";
  return `http://${host}:8766`;
})();

const DEBOUNCE_MS = 300;  // debounce slider input to avoid flooding the server

// ── Hook ────────────────────────────────────────────────────────────────────

export function useGradientServer() {
  const [gradient, setGradient]   = useState(null);   // { param: sensitivity }
  const [ranked, setRanked]       = useState([]);      // sorted by |sensitivity|
  const [lapTime, setLapTime]     = useState(null);    // scalar [s]
  const [computeMs, setComputeMs] = useState(null);    // backend compute time
  const [loading, setLoading]     = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError]         = useState(null);

  const debounceRef = useRef(null);
  const abortRef    = useRef(null);

  // ── Health check on mount ──────────────────────────────────────────────

  useEffect(() => {
    let mounted = true;

    async function checkHealth() {
      try {
        const res = await fetch(`${GRADIENT_URL}/health`, { signal: AbortSignal.timeout(3000) });
        if (mounted && res.ok) {
          setConnected(true);
          setError(null);
        }
      } catch {
        if (mounted) {
          setConnected(false);
          setError("Gradient server not available (run scripts/gradient_server.py)");
        }
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 10000);  // re-check every 10s

    return () => { mounted = false; clearInterval(interval); };
  }, []);

  // ── Compute gradient (debounced) ───────────────────────────────────────

  const computeGradient = useCallback((setup28) => {
    // Clear any pending debounce
    if (debounceRef.current) clearTimeout(debounceRef.current);

    debounceRef.current = setTimeout(async () => {
      // Abort any in-flight request
      if (abortRef.current) abortRef.current.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setLoading(true);
      setError(null);

      try {
        const res = await fetch(`${GRADIENT_URL}/gradient`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ setup: Array.from(setup28) }),
          signal: controller.signal,
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();

        setGradient(data.gradients);
        setRanked(data.ranked || []);
        setLapTime(data.lap_time);
        setComputeMs(data.compute_ms);
        setConnected(true);

      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message);
          setConnected(false);
        }
      } finally {
        setLoading(false);
      }
    }, DEBOUNCE_MS);
  }, []);

  // ── Fetch default gradient (for initial display) ──────────────────────

  const fetchDefault = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${GRADIENT_URL}/default`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setGradient(data.gradients);
      setRanked(data.ranked || []);
      setLapTime(data.lap_time);
      setComputeMs(data.compute_ms);
      setConnected(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    computeGradient,
    fetchDefault,
    gradient,
    ranked,
    lapTime,
    computeMs,
    loading,
    connected,
    error,
  };
}


/**
 * Format a sensitivity value for display.
 * Returns e.g. "-0.0234 s/(N/mm)" for a spring rate sensitivity.
 */
export function formatSensitivity(value, unit) {
  if (value == null) return "—";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(4)} s/${unit || "unit"}`;
}
