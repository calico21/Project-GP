import React, { useState, useEffect, useRef } from "react";
import { C, GL } from "./theme.js";

// ── FadeSlide — FIXED ───────────────────────────────────────────────
// Previous version had `children` in useEffect deps → new React element
// reference every render → infinite loop → opacity stuck at 0 → BLACK SCREEN.
// Fix: only depend on keyVal, render children directly (no state copy).
export function FadeSlide({ children, keyVal }) {
  const [visible, setVisible] = useState(true);
  const prevKey = useRef(keyVal);

  useEffect(() => {
    if (keyVal !== prevKey.current) {
      setVisible(false);
      const t = setTimeout(() => {
        prevKey.current = keyVal;
        setVisible(true);
      }, 220);
      return () => clearTimeout(t);
    }
  }, [keyVal]);

  return (
    <div style={{
      opacity: visible ? 1 : 0,
      transform: visible ? "translateY(0)" : "translateY(14px)",
      transition: "opacity 0.35s ease, transform 0.35s ease",
    }}>
      {children}
    </div>
  );
}

// ── KPI Card ────────────────────────────────────────────────────────
export function KPI({ label, value, sub, sentiment = "neutral", delay = 0 }) {
  const ac = sentiment === "positive" ? C.gn : sentiment === "negative" ? C.red : sentiment === "amber" ? C.am : C.cy;
  const bg = sentiment === "positive" ? C.gnG : sentiment === "negative" ? C.redG : sentiment === "amber" ? C.amG : C.cyG;
  const [vis, setVis] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setVis(true), 80 + delay * 100);
    return () => clearTimeout(t);
  }, [delay]);

  return (
    <div
      style={{
        ...GL, borderTop: `2px solid ${ac}`, textAlign: "center",
        padding: "18px 14px 14px", cursor: "default",
        opacity: vis ? 1 : 0,
        transform: vis ? "translateY(0) scale(1)" : "translateY(10px) scale(0.97)",
        transition: `all 0.45s cubic-bezier(0.25,0.1,0.25,1) ${delay * 70}ms`,
      }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = ac; e.currentTarget.style.boxShadow = `0 8px 40px ${ac}18, inset 0 1px 0 rgba(255,255,255,0.05)`; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = C.glassB; e.currentTarget.style.boxShadow = GL.boxShadow; }}
    >
      <div style={{ fontSize: 9, fontWeight: 600, color: ac, textTransform: "uppercase", letterSpacing: 3, marginBottom: 8, fontFamily: C.dt }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 800, color: C.w, fontFamily: C.hd, lineHeight: 1.1 }}>{value}</div>
      {sub && <div style={{ fontSize: 9, marginTop: 8, padding: "3px 10px", display: "inline-block", borderRadius: 20, fontWeight: 600, color: ac, background: bg, border: `1px solid ${ac}25`, fontFamily: C.dt }}>{sub}</div>}
    </div>
  );
}

// ── Section header ──────────────────────────────────────────────────
export function Sec({ title, children, right }) {
  return (
    <div style={{ marginBottom: 22 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <div style={{ width: 3, height: 16, background: `linear-gradient(180deg, ${C.red}, ${C.red}40)`, borderRadius: 2, flexShrink: 0 }} />
        <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: 2, textTransform: "uppercase", color: C.br, fontFamily: C.hd }}>{title}</span>
        <div style={{ flex: 1, height: 1, background: `linear-gradient(90deg, ${C.b1}, transparent)` }} />
        {right}
      </div>
      {children}
    </div>
  );
}

// ── Glass chart wrapper ─────────────────────────────────────────────
export function GC({ children, style }) {
  return <div style={{ ...GL, padding: "14px 10px 8px", ...style }}>{children}</div>;
}

// ── Pill button ─────────────────────────────────────────────────────
export function Pill({ active, label, onClick, color }) {
  return (
    <button onClick={onClick} style={{
      background: active ? `${color || C.red}14` : "transparent",
      border: `1px solid ${active ? color || C.red : C.b1}`,
      borderRadius: 20, padding: "6px 18px", cursor: "pointer",
      color: active ? color || C.red : C.md, fontFamily: C.hd,
      fontSize: 11, fontWeight: active ? 700 : 500, letterSpacing: 1,
      textTransform: "uppercase",
      transition: "all 0.25s cubic-bezier(0.25,0.1,0.25,1)",
      transform: active ? "scale(1.03)" : "scale(1)",
    }}>
      {label}
    </button>
  );
}

// ── Math sub-components ─────────────────────────────────────────────
export function Eq({ children }) {
  return (
    <div style={{
      fontFamily: C.dt, fontSize: 12, color: C.cy,
      padding: "8px 14px", margin: "8px 0",
      background: "rgba(0,210,255,0.04)", borderRadius: 8,
      border: "1px solid rgba(0,210,255,0.08)",
      letterSpacing: 0.3, lineHeight: 1.7, overflowX: "auto", whiteSpace: "pre",
    }}>
      {children}
    </div>
  );
}

export function Pt({ children }) {
  return <p style={{ fontSize: 12.5, color: C.br, lineHeight: 1.85, margin: "6px 0", fontFamily: C.bd, fontWeight: 400 }}>{children}</p>;
}

export function Hl({ children, color }) {
  return <span style={{ color: color || C.cy, fontWeight: 700, fontFamily: C.dt, fontSize: 11 }}>{children}</span>;
}

export function InfoBox({ color, title, children }) {
  return (
    <div style={{ padding: "12px 14px", borderRadius: 8, background: `${color}06`, border: `1px solid ${color}12` }}>
      <div style={{ fontSize: 9, fontWeight: 700, color, fontFamily: C.dt, letterSpacing: 2, marginBottom: 6 }}>{title}</div>
      {children}
    </div>
  );
}
