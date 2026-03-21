// src/theme.js — Project-GP Design System

export const C = {
  bg: "#05070b", bg2: "#080c14", card: "rgba(10,15,25,0.55)",
  panel: "rgba(8,12,20,0.72)", surf: "rgba(14,20,32,0.6)",
  hover: "rgba(20,30,50,0.55)", glassB: "rgba(70,100,160,0.10)",
  b1: "rgba(35,48,75,0.45)", b2: "rgba(50,68,105,0.40)", b3: "rgba(75,95,140,0.35)",
  red: "#e10600", redG: "rgba(225,6,0,0.12)",
  cy: "#00d2ff", cyG: "rgba(0,210,255,0.10)",
  gn: "#00e676", gnG: "rgba(0,230,118,0.10)",
  am: "#ffab00", amG: "rgba(255,171,0,0.10)",
  pr: "#b388ff", prG: "rgba(179,136,255,0.10)",
  w: "#eef0f6", br: "#c0c8da", md: "#6e7a96", dm: "#3e4a64",
  hd: "'Outfit', sans-serif",
  dt: "'Azeret Mono', 'IBM Plex Mono', monospace",
  bd: "'Outfit', sans-serif",
};

export const GL = {
  background: C.card,
  backdropFilter: "blur(24px) saturate(1.5)",
  WebkitBackdropFilter: "blur(24px) saturate(1.5)",
  border: `1px solid ${C.glassB}`,
  borderRadius: 12,
  boxShadow: "0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03)",
};

export const GS = "rgba(25,35,55,0.5)";

export const TT = {
  backgroundColor: "rgba(8,12,20,0.95)", backdropFilter: "blur(16px)",
  border: `1px solid ${C.b2}`, borderRadius: 8,
  fontSize: 11, fontFamily: C.dt, color: C.br, padding: "10px 12px",
};

export const FONTS_URL =
  "https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Azeret+Mono:wght@400;500;600;700&display=swap";
