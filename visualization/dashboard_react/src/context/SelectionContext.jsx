// ═══════════════════════════════════════════════════════════════════════════════
// src/context/SelectionContext.jsx
// ═══════════════════════════════════════════════════════════════════════════════
//
// Cross-chart linked selection bus. Provides shared state for:
//
//   • selectedSetupId   — Pareto point currently inspected (click in scatter)
//   • comparisonId      — second Pareto point for A/B diff (shift-click)
//   • timeRange         — [t0, t1] brush selection on time-series charts
//   • hoveredParam      — parameter name highlighted in sensitivity heatmap
//   • liveStep          — current simulation step counter (Live mode)
//
// Usage in any module:
//
//   import { useSelection } from "../context/SelectionContext.jsx";
//   const { selectedSetupId, setSelectedSetupId } = useSelection();
//
// Wrap your <App> in <SelectionProvider> (done in App.jsx patch).
// ═══════════════════════════════════════════════════════════════════════════════

import React, { createContext, useContext, useState, useCallback } from "react";

const SelectionContext = createContext(null);

export function SelectionProvider({ children }) {
  // ── Pareto selection ────────────────────────────────────────────────
  const [selectedSetupId, setSelectedSetupId] = useState(null);
  const [comparisonId, setComparisonId]       = useState(null);

  // ── Time-series brush ───────────────────────────────────────────────
  const [timeRange, setTimeRange] = useState(null);   // [t0, t1] or null

  // ── Sensitivity hover ───────────────────────────────────────────────
  const [hoveredParam, setHoveredParam] = useState(null);

  // ── Live mode step counter ──────────────────────────────────────────
  const [liveStep, setLiveStep] = useState(0);

  // ── Convenience: clear all selections at once ───────────────────────
  const clearAll = useCallback(() => {
    setSelectedSetupId(null);
    setComparisonId(null);
    setTimeRange(null);
    setHoveredParam(null);
  }, []);

  // ── Shift-click handler for Pareto scatter ──────────────────────────
  // Normal click → sets primary selection.
  // If primary is already set and a different point is clicked → comparison.
  const handleParetoClick = useCallback((id, isShift = false) => {
    if (isShift && selectedSetupId !== null && id !== selectedSetupId) {
      setComparisonId(id);
    } else {
      setSelectedSetupId(id);
      setComparisonId(null);
    }
  }, [selectedSetupId]);

  const value = {
    // Pareto
    selectedSetupId, setSelectedSetupId,
    comparisonId,    setComparisonId,
    handleParetoClick,

    // Time
    timeRange, setTimeRange,

    // Hover
    hoveredParam, setHoveredParam,

    // Live
    liveStep, setLiveStep,

    // Utility
    clearAll,
  };

  return (
    <SelectionContext.Provider value={value}>
      {children}
    </SelectionContext.Provider>
  );
}

/**
 * Hook to access the selection context.
 * Throws if used outside SelectionProvider.
 */
export function useSelection() {
  const ctx = useContext(SelectionContext);
  if (!ctx) {
    throw new Error("useSelection() must be used inside <SelectionProvider>");
  }
  return ctx;
}

export default SelectionContext;