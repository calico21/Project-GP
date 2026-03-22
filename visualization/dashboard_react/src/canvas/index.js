// ═══════════════════════════════════════════════════════════════════════════════
// src/canvas/index.js — Barrel export for all canvas visualization components
// ═══════════════════════════════════════════════════════════════════════════════
//
// Usage in any module:
//   import { ThermalQuintet, TubeTrackMap, WaveletHeatmap } from "./canvas";
//
// ═══════════════════════════════════════════════════════════════════════════════

export { default as ThermalQuintet, ThermalRadial, FlashDeltaBar } from "./ThermalQuintet.jsx";
export { default as TubeTrackMap } from "./TubeTrackMap.jsx";
export { default as WaveletHeatmap } from "./WaveletHeatmap.jsx";
export { default as HnetContour } from "./HnetContour.jsx";
export { default as DissipationMatrix } from "./DissipationMatrix.jsx";
export { default as FrictionHeatmap } from "./FrictionHeatmap.jsx";
export { default as SensitivityHeatmap } from "./SensitivityHeatmap.jsx";