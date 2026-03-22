// ═══════════════════════════════════════════════════════════════════════════════
// src/TirePhysicsModule.jsx
// ═══════════════════════════════════════════════════════════════════════════════
//
// Multi-fidelity tire model deep-dive.
//
// Sub-tabs:
//   1. THERMAL    — 4-corner × 5-node radial diagrams + full evolution chart
//   2. GP         — Sparse GP posterior ±2σ uncertainty envelope over slip angle
//   3. HYSTERESIS — Slip-force Lissajous trace + transient vs steady-state
//   4. LOAD       — Peak μ and Cα vs Fz (Pacejka degressive characteristic)
//   5. FRICTION   — Combined slip friction carpet |F|(α,κ)
//
// Consumes:
//   - thermal5   : Array from gThermal5()
//   - gpEnv      : Array from gGPEnvelope()
//   - hysteresis : Array from gHysteresis()
//   - loadSens   : Array from gLoadSensitivity()
//   - frictionSurf : Array from gFrictionSurface() [computed internally]
//
// Composes:
//   - ThermalQuintet, ThermalRadial (canvas/ThermalQuintet.jsx)
//   - FrictionHeatmap (canvas/FrictionHeatmap.jsx)
//   - Recharts AreaChart, LineChart, ScatterChart, ComposedChart
//
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo } from "react";
import {
  AreaChart, Area, LineChart, Line, ScatterChart, Scatter,
  ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Cell, Legend,
} from "recharts";
import { C, GL, GS, TT, AX, thermalColor } from "./theme.js";
import { Sec, GC, Pill } from "./components.jsx";
import { ThermalQuintet, ThermalRadial } from "./canvas";
import { FrictionHeatmap } from "./canvas";
import { gFrictionSurface } from "./data.js";

// ─── Sub-tab configuration ───────────────────────────────────────────────────

const TABS = [
  { key: "thermal",    label: "5-Node Thermal" },
  { key: "gp",         label: "GP Uncertainty" },
  { key: "hysteresis", label: "Slip Hysteresis" },
  { key: "load",       label: "Load Sensitivity" },
  { key: "friction",   label: "Friction Surface" },
];

const CORNER_KEYS = ["fl", "fr", "rl", "rr"];
const CORNER_LABELS = ["FL", "FR", "RL", "RR"];
const NODE_NAMES = ["Flash", "Surface", "Bulk", "Carcass", "Gas"];
const NODE_COLORS = [C.th_crit, C.th_hot, C.th_warm, C.th_cool, C.th_cold];
const NODE_DASH = ["4 2", "", "", "", ""];

// ═════════════════════════════════════════════════════════════════════════════
// THERMAL SUB-TAB
// ═════════════════════════════════════════════════════════════════════════════

function ThermalTab({ thermal5 }) {
  const [activeCorner, setActiveCorner] = useState("fl");

  // Last frame for per-corner summary cards
  const lastFrame = thermal5[thermal5.length - 1];

  // Downsample for chart performance
  const chartData = useMemo(
    () => thermal5.filter((_, i) => i % 2 === 0),
    [thermal5],
  );

  return (
    <div>
      {/* ── Per-corner summary cards with radial diagrams ──────────── */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(4, 1fr)",
        gap: 8, marginBottom: 14,
      }}>
        {CORNER_KEYS.map((key, ci) => {
          const temps = lastFrame?.[key];
          if (!temps) return null;
          const isActive = activeCorner === key;

          return (
            <GC
              key={key}
              style={{
                textAlign: "center",
                borderTop: `2px solid ${thermalColor(temps[1])}`,
                cursor: "pointer",
                outline: isActive ? `1px solid ${C.cy}` : "none",
                transition: "outline 0.15s",
              }}
            >
              <div
                onClick={() => setActiveCorner(key)}
                style={{ padding: "4px 0" }}
              >
                <span style={{
                  fontSize: 8, fontWeight: 700, letterSpacing: 2,
                  color: thermalColor(temps[1]),
                  fontFamily: C.dt, textTransform: "uppercase",
                }}>
                  {CORNER_LABELS[ci]}
                </span>

                <ThermalRadial temps={temps} label="" size={110} />

                {/* Per-node readout table */}
                <div style={{
                  display: "grid", gridTemplateColumns: "1fr 1fr",
                  gap: 1, marginTop: 4, padding: "0 6px",
                }}>
                  {NODE_NAMES.map((name, ni) => (
                    <div key={name} style={{
                      fontSize: 7, fontFamily: C.dt,
                      display: "flex", justifyContent: "space-between",
                      padding: "1px 2px",
                    }}>
                      <span style={{ color: C.dm }}>{name}</span>
                      <span style={{ color: thermalColor(temps[ni]), fontWeight: 600 }}>
                        {temps[ni].toFixed(1)}°
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </GC>
          );
        })}
      </div>

      {/* ── Full thermal evolution chart for selected corner ──────── */}
      <Sec
        title={`Thermal Evolution — ${activeCorner.toUpperCase()} — All 5 Layers`}
        right={
          <div style={{ display: "flex", gap: 4 }}>
            {CORNER_KEYS.map((k, i) => (
              <Pill
                key={k}
                active={activeCorner === k}
                label={CORNER_LABELS[i]}
                onClick={() => setActiveCorner(k)}
                color={C.cy}
              />
            ))}
          </div>
        }
      >
        <GC>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="s" {...AX} label={{ value: "Distance (m)", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis {...AX} domain={[20, 165]} label={{ value: "Temperature (°C)", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />

              {/* Optimal grip band */}
              <ReferenceArea
                y1={85} y2={105}
                fill={C.gn} fillOpacity={0.04}
                label={{ value: "Optimal grip band", fill: C.gn, fontSize: 7, fontFamily: C.dt, position: "insideTopRight" }}
              />

              {/* Degradation threshold */}
              <ReferenceLine
                y={140} stroke={C.red} strokeDasharray="4 2"
                label={{ value: "Degradation onset", fill: C.red, fontSize: 7, fontFamily: C.dt }}
              />

              {/* 5 thermal layers */}
              {NODE_NAMES.map((name, ni) => (
                <Line
                  key={name}
                  type="monotone"
                  dataKey={d => d[activeCorner]?.[ni]}
                  stroke={NODE_COLORS[ni]}
                  strokeWidth={ni === 1 ? 2.5 : 1.5}
                  dot={false}
                  name={name}
                  strokeDasharray={NODE_DASH[ni]}
                />
              ))}

              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
            </LineChart>
          </ResponsiveContainer>
          <div style={{
            fontSize: 8, color: C.dm, fontFamily: C.dt,
            padding: "4px 8px", lineHeight: 1.6,
          }}>
            <span style={{ color: C.th_crit }}>Flash</span> (dashed) spikes under high slip — Jaeger transient.{" "}
            <span style={{ color: C.th_hot }}>Surface</span> (bold) is the primary grip driver.{" "}
            Large flash-to-surface delta = thermal abuse not yet soaked through.
          </div>
        </GC>
      </Sec>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// GP UNCERTAINTY SUB-TAB
// ═════════════════════════════════════════════════════════════════════════════

function GPTab({ gpEnv }) {
  return (
    <Sec title="Sparse GP Posterior ±2σ — Lateral Force vs Slip Angle">
      <GC>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={gpEnv} margin={{ top: 12, right: 20, bottom: 24, left: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} />
            <XAxis
              dataKey="alpha" {...AX}
              label={{ value: "Slip Angle α (°)", position: "bottom", fill: C.dm, fontSize: 9 }}
            />
            <YAxis
              {...AX}
              label={{ value: "Fy (N)", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }}
            />
            <Tooltip contentStyle={TT} />

            {/* ±2σ band (filled area between upper and lower) */}
            <Area type="monotone" dataKey="upper" stroke="none" fill={C.sl_gp} fillOpacity={0.12} name="+2σ bound" />
            <Area type="monotone" dataKey="lower" stroke="none" fill={C.sl_gp} fillOpacity={0.12} name="-2σ bound" />

            {/* Band boundary lines */}
            <Line type="monotone" dataKey="upper" stroke={C.sl_gp} strokeWidth={0.5} dot={false} name="+2σ" strokeDasharray="3 2" />
            <Line type="monotone" dataKey="lower" stroke={C.sl_gp} strokeWidth={0.5} dot={false} name="-2σ" strokeDasharray="3 2" />

            {/* Mean prediction (Pacejka + PINN) */}
            <Line type="monotone" dataKey="mu" stroke={C.am} strokeWidth={2.5} dot={false} name="Pacejka + PINN mean" />

            <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
          </AreaChart>
        </ResponsiveContainer>

        {/* Sigma magnitude subplot */}
        <ResponsiveContainer width="100%" height={100}>
          <AreaChart data={gpEnv} margin={{ top: 4, right: 20, bottom: 8, left: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} />
            <XAxis dataKey="alpha" {...AX} />
            <YAxis {...AX} />
            <Tooltip contentStyle={TT} />
            <Area type="monotone" dataKey="sigma" stroke={C.sl_gp} fill={C.sl_gp} fillOpacity={0.15} strokeWidth={1.5} name="σ (N)" />
          </AreaChart>
        </ResponsiveContainer>

        <div style={{
          fontSize: 8, color: C.dm, fontFamily: C.dt,
          padding: "6px 8px", lineHeight: 1.6,
        }}>
          Purple band: Matérn 5/2 Sparse GP posterior ±2σ.
          Narrow band = high confidence (dense inducing points).
          Wide band at extremes = extrapolation zone where LCB penalty is active (capped at 0.15 to prevent force collapse to 44% of Pacejka).
          The lower σ subplot shows uncertainty magnitude — peaks at ±15° where the tire operates near the friction limit.
        </div>
      </GC>
    </Sec>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// HYSTERESIS SUB-TAB
// ═════════════════════════════════════════════════════════════════════════════

function HysteresisTab({ hysteresis }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
      {/* Lissajous trace: slip angle (X) vs Fy (Y) */}
      <Sec title="Slip–Force Hysteresis Loop (Lissajous)">
        <GC>
          <ResponsiveContainer width="100%" height={360}>
            <ScatterChart margin={{ top: 12, right: 16, bottom: 24, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis
                dataKey="alpha" name="α (°)" {...AX}
                label={{ value: "Slip Angle α (°)", position: "bottom", fill: C.dm, fontSize: 9 }}
              />
              <YAxis
                dataKey="Fy" name="Fy (N)" {...AX}
                label={{ value: "Fy (N)", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }}
              />
              <Tooltip contentStyle={TT} />
              <Scatter data={hysteresis} name="Transient trace">
                {hysteresis.map((d, i) => (
                  <Cell
                    key={i}
                    fill={C.sl_lat}
                    opacity={0.12 + 0.88 * (i / hysteresis.length)}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div style={{
            fontSize: 8, color: C.dm, fontFamily: C.dt,
            padding: "6px 8px", lineHeight: 1.6,
          }}>
            Trace opacity = time (dim → bright). Loop width reveals the
            relaxation length effect: wider loop = more transient lag.
            The tire is operating in the pneumatic-trail-dominant regime
            when the loop is asymmetric about the origin.
          </div>
        </GC>
      </Sec>

      {/* Transient vs steady-state time-domain comparison */}
      <Sec title="Transient vs Steady-State Response">
        <GC>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart
              data={hysteresis.filter((_, i) => i % 2 === 0)}
              margin={{ top: 8, right: 12, bottom: 8, left: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="t" {...AX} />
              <YAxis {...AX} />
              <Tooltip contentStyle={TT} />
              <Line type="monotone" dataKey="Fy" stroke={C.sl_lat} strokeWidth={2} dot={false} name="Fy transient" />
              <Line type="monotone" dataKey="Fy_ss" stroke={C.am} strokeWidth={1} dot={false} name="Fy steady-state" strokeDasharray="4 2" />
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
            </LineChart>
          </ResponsiveContainer>

          {/* Slip angle input overlay */}
          <ResponsiveContainer width="100%" height={100}>
            <LineChart
              data={hysteresis.filter((_, i) => i % 2 === 0)}
              margin={{ top: 4, right: 12, bottom: 8, left: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="t" {...AX} />
              <YAxis {...AX} />
              <Tooltip contentStyle={TT} />
              <Line type="monotone" dataKey="alpha" stroke={C.pr} strokeWidth={1.5} dot={false} name="α input (°)" />
              <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
            </LineChart>
          </ResponsiveContainer>

          <div style={{
            fontSize: 8, color: C.dm, fontFamily: C.dt,
            padding: "4px 8px", lineHeight: 1.6,
          }}>
            Top: force response. Dashed = pure Pacejka (instantaneous).
            Solid = transient model with first-order relaxation (τ ≈ 0.05s).
            Bottom: sinusoidal slip angle input driving both models.
          </div>
        </GC>
      </Sec>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// LOAD SENSITIVITY SUB-TAB
// ═════════════════════════════════════════════════════════════════════════════

function LoadTab({ loadSens }) {
  return (
    <Sec title="Load Sensitivity — Peak μ and Cornering Stiffness vs Fz">
      <GC>
        <ResponsiveContainer width="100%" height={380}>
          <ComposedChart data={loadSens} margin={{ top: 12, right: 24, bottom: 24, left: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} />
            <XAxis
              dataKey="Fz" {...AX}
              label={{ value: "Vertical Load Fz (N)", position: "bottom", fill: C.dm, fontSize: 9 }}
            />
            <YAxis
              yAxisId="l" {...AX}
              label={{ value: "μ_peak", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }}
            />
            <YAxis
              yAxisId="r" orientation="right" {...AX}
              label={{ value: "Cα (N/rad)", angle: 90, position: "insideRight", fill: C.dm, fontSize: 9 }}
            />
            <Tooltip contentStyle={TT} />
            <Line yAxisId="l" type="monotone" dataKey="mu" stroke={C.am} strokeWidth={2.5} dot={false} name="μ_peak (degressive)" />
            <Line yAxisId="r" type="monotone" dataKey="Ca" stroke={C.cy} strokeWidth={2} dot={false} name="Cα cornering stiffness" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
          </ComposedChart>
        </ResponsiveContainer>
        <div style={{
          fontSize: 9, color: C.dm, fontFamily: C.dt,
          padding: "6px 8px", lineHeight: 1.7,
        }}>
          Classic Pacejka degressive load sensitivity (PDY1/PDY2 coefficients, Hoosier R20 TTC-fitted).
          Higher vertical load → lower peak friction coefficient but higher absolute force.
          This is the fundamental reason <span style={{ color: C.am }}>load transfer hurts total grip</span>:
          the loaded tire gains less than the unloaded tire loses.
          Cornering stiffness Cα saturates — above ~1500N, adding more load gives diminishing returns for transient response.
        </div>
      </GC>
    </Sec>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// FRICTION SURFACE SUB-TAB
// ═════════════════════════════════════════════════════════════════════════════

function FrictionTab() {
  // Compute friction surface on mount (25×25 grid = 625 points)
  const frictionData = useMemo(() => gFrictionSurface(25), []);

  return (
    <Sec title="Combined Slip Friction Surface — |F|(α, κ)">
      <GC style={{ padding: 10 }}>
        <div style={{
          display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap",
        }}>
          <FrictionHeatmap data={frictionData} width={380} height={380} />
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{
              fontSize: 10, fontWeight: 700, color: C.br, fontFamily: C.hd,
              letterSpacing: 1, textTransform: "uppercase", marginBottom: 8,
            }}>
              Interpretation
            </div>
            <div style={{
              fontSize: 8, color: C.dm, fontFamily: C.dt, lineHeight: 1.7,
            }}>
              The friction circle topology is visible as the bright ridge of peak force
              running diagonally across the surface. Pure lateral force (α only) and pure
              longitudinal force (κ only) each reach their individual maxima on the axes,
              but combined slip reduces both — the coupling penalty.
            </div>
            <div style={{
              marginTop: 10, padding: "8px 10px",
              background: `${C.cy}08`, borderRadius: 8,
              border: `1px solid ${C.cy}15`,
            }}>
              <div style={{
                fontSize: 8, fontWeight: 700, color: C.cy,
                fontFamily: C.dt, letterSpacing: 1, marginBottom: 4,
              }}>
                PINN CORRECTION
              </div>
              <div style={{
                fontSize: 8, color: C.md, fontFamily: C.dt, lineHeight: 1.6,
              }}>
                The PINN module adds a deterministic correction layer on top of the base
                Pacejka surface (8-feature input including T_norm). Brightness modulation
                in the heatmap indicates where the PINN residual is largest — typically at
                high combined slip where Pacejka MF6.2 underestimates the coupling effect.
              </div>
            </div>
            <div style={{
              marginTop: 10, padding: "8px 10px",
              background: `${C.sl_gp}08`, borderRadius: 8,
              border: `1px solid ${C.sl_gp}15`,
            }}>
              <div style={{
                fontSize: 8, fontWeight: 700, color: C.sl_gp,
                fontFamily: C.dt, letterSpacing: 1, marginBottom: 4,
              }}>
                GP UNCERTAINTY LAYER
              </div>
              <div style={{
                fontSize: 8, color: C.md, fontFamily: C.dt, lineHeight: 1.6,
              }}>
                The Matérn 5/2 Sparse GP wraps the PINN output with stochastic bounds.
                LCB penalty (capped at 0.15) reduces predicted force in high-uncertainty
                regions, biasing the WMPC toward conservative trajectories where the tire
                model is least confident.
              </div>
            </div>
          </div>
        </div>
      </GC>
    </Sec>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// MAIN MODULE EXPORT
// ═════════════════════════════════════════════════════════════════════════════

export default function TirePhysicsModule({
  thermal5, gpEnv, hysteresis, loadSens,
}) {
  const [tab, setTab] = useState("thermal");

  return (
    <div>
      {/* ── Tab Switcher ──────────────────────────────────────────────── */}
      <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
        {TABS.map(t => (
          <Pill
            key={t.key}
            active={tab === t.key}
            label={t.label}
            onClick={() => setTab(t.key)}
            color={C.am}
          />
        ))}
      </div>

      {/* ── Tab Content ───────────────────────────────────────────────── */}
      {tab === "thermal"    && <ThermalTab thermal5={thermal5} />}
      {tab === "gp"         && <GPTab gpEnv={gpEnv} />}
      {tab === "hysteresis" && <HysteresisTab hysteresis={hysteresis} />}
      {tab === "load"       && <LoadTab loadSens={loadSens} />}
      {tab === "friction"   && <FrictionTab />}
    </div>
  );
}