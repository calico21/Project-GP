// ═══════════════════════════════════════════════════════════════════════════════
// src/context/SelectionContext.jsx — v5.0
// ═══════════════════════════════════════════════════════════════════════════════
//
// Cross-chart linked selection bus. Provides shared state for:
//
//   ── v4.0 fields (unchanged) ────────────────────────────────────────────
//   • selectedSetupId   — Pareto point currently inspected (click in scatter)
//   • comparisonId      — second Pareto point for A/B diff (shift-click)
//   • timeRange         — [t0, t1] brush selection on time-series charts
//   • hoveredParam      — parameter name highlighted in sensitivity heatmap
//   • liveStep          — current simulation step counter (Live mode)
//
//   ── v5.0 additions ─────────────────────────────────────────────────────
//   • aeroSlice         — { pitch, rh_f, rh_r, roll, yaw } aero map slice
//   • selectedCell      — { segment, cell } accumulator cell drill-down
//   • thermalNode       — selected node in coolant loop schematic
//   • powerTimestamp    — synced timestamp for power budget Sankey
//   • activeModule      — currently active module key (for cross-linking)
//
// Usage in any module:
//
//   import { useSelection } from "../context/SelectionContext.jsx";
//   const { selectedSetupId, aeroSlice, setAeroSlice } = useSelection();
//
// Wrap your <App> in <SelectionProvider> (done in App.jsx).
// ═══════════════════════════════════════════════════════════════════════════════

import React, { createContext, useContext, useState, useCallback } from "react";

const SelectionContext = createContext(null);

// Default aero slice — nominal operating point
const DEFAULT_AERO_SLICE = {
pitch: 0,
rh_f: 28,
rh_r: 32,
roll: 0,
yaw: 0,
};

export function SelectionProvider({ children }) {
// ── Pareto selection (v4.0) ─────────────────────────────────────────
const [selectedSetupId, setSelectedSetupId] = useState(null);
const [comparisonId, setComparisonId]       = useState(null);

// ── Time-series brush (v4.0) ────────────────────────────────────────
const [timeRange, setTimeRange] = useState(null);   // [t0, t1] or null

// ── Sensitivity hover (v4.0) ────────────────────────────────────────
const [hoveredParam, setHoveredParam] = useState(null);

// ── Live mode step counter (v4.0) ───────────────────────────────────
const [liveStep, setLiveStep] = useState(0);

// ══════════════════════════════════════════════════════════════════════
// v5.0 ADDITIONS — Aero & Electronics cross-linking
// ══════════════════════════════════════════════════════════════════════

// ── Aero map slice plane ────────────────────────────────────────────
// Set by AerodynamicsModule when user interacts with the 5D heatmap.
// Consumed by: Telemetry aero workspace, LiveMode AeroPlatform panel,
// DifferentiableInsights gradient coupling tab.
const [aeroSlice, setAeroSlice] = useState(DEFAULT_AERO_SLICE);

// ── Accumulator cell selection ──────────────────────────────────────
// Set by ElectronicsModule CellMatrix click.
// Consumed by: cell history detail panel, thermal management node highlight.
const [selectedCell, setSelectedCell] = useState(null); // { segment, cell } or null

// ── Coolant loop thermal node ───────────────────────────────────────
// Set by ElectronicsModule CoolantSchematic click.
// Consumed by: Endurance thermal derating projection.
const [thermalNode, setThermalNode] = useState(null); // string: "radIn"|"motJacket"|etc

// ── Power budget synced timestamp ───────────────────────────────────
// Set by any time-series scrub that has power context.
// Consumed by: Electronics PowerBudget Sankey for animation lockstep.
const [powerTimestamp, setPowerTimestamp] = useState(null); // number or null

// ── Active module tracking ──────────────────────────────────────────
// Set by App.jsx on navigation. Allows modules to know who’s active
// for conditional rendering or cross-link hover states.
const [activeModule, setActiveModule] = useState("overview");

// ── Convenience: clear all selections at once ───────────────────────
const clearAll = useCallback(() => {
setSelectedSetupId(null);
setComparisonId(null);
setTimeRange(null);
setHoveredParam(null);
setAeroSlice(DEFAULT_AERO_SLICE);
setSelectedCell(null);
setThermalNode(null);
setPowerTimestamp(null);
}, []);

// ── Shift-click handler for Pareto scatter (v4.0, unchanged) ────────
const handleParetoClick = useCallback((id, isShift = false) => {
if (isShift && selectedSetupId !== null && id !== selectedSetupId) {
setComparisonId(id);
} else {
setSelectedSetupId(id);
setComparisonId(null);
}
}, [selectedSetupId]);

// ── Aero slice updater — merges partial updates ─────────────────────
const updateAeroSlice = useCallback((partial) => {
setAeroSlice(prev => ({ ...prev, ...partial }));
}, []);

const value = {
// ── v4.0 fields ──────────────────────────────────────────────────
selectedSetupId, setSelectedSetupId,
comparisonId,    setComparisonId,
handleParetoClick,
timeRange,       setTimeRange,
hoveredParam,    setHoveredParam,
liveStep,        setLiveStep,


// ── v5.0 fields ──────────────────────────────────────────────────
aeroSlice,       setAeroSlice, updateAeroSlice,
selectedCell,    setSelectedCell,
thermalNode,     setThermalNode,
powerTimestamp,  setPowerTimestamp,
activeModule,    setActiveModule,

// ── Utility ──────────────────────────────────────────────────────
clearAll,
DEFAULT_AERO_SLICE,


};

return (
<SelectionContext.Provider value={value}>
{children}
</SelectionContext.Provider>
);
}

/**

- Hook to access the selection context.
- Throws if used outside SelectionProvider.
  */
  export function useSelection() {
  const ctx = useContext(SelectionContext);
  if (!ctx) {
  throw new Error("useSelection() must be used inside <SelectionProvider>");
  }
  return ctx;
  }

export default SelectionContext;