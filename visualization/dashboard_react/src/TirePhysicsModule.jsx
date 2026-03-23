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
import { KPI, Sec, GC, Pill, FadeSlide } from "./components.jsx";
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
  { key: "mz",         label: "Aligning Moment" },
  { key: "slipTarget", label: "Slip Target" },
  { key: "relax",      label: "Relaxation" },
  { key: "wear",       label: "Tire Wear" },
  { key: "gpCoverage", label: "GP Coverage" },
  { key: "tempGrip",   label: "Temp vs Grip" },
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
function gMzTrace(n = 200) {
  const data = [];
  for (let i = 0; i < n; i++) {
    const alpha = (i / n - 0.5) * 24;
    const alphaRad = alpha * Math.PI / 180;
    const Bt = 10.9, Ct = 1.18, Dt = 0.032;
    const trail = Dt * Math.cos(Ct * Math.atan(Bt * alphaRad));
    const Fy = 3000 * Math.sin(1.4 * Math.atan(12 * alphaRad));
    const Mz = -trail * Fy;
    const Fy_light = 2200 * Math.sin(1.4 * Math.atan(12 * alphaRad));
    const Fy_heavy = 3600 * Math.sin(1.4 * Math.atan(12 * alphaRad));
    const Mz_light = -(Dt * 1.15 * Math.cos(Ct * Math.atan(Bt * 0.9 * alphaRad))) * Fy_light;
    const Mz_heavy = -(Dt * 0.85 * Math.cos(Ct * Math.atan(Bt * 1.1 * alphaRad))) * Fy_heavy;
    data.push({
      alpha: +alpha.toFixed(1), trail: +(trail * 1000).toFixed(1),
      Mz: +Mz.toFixed(0), Mz_light: +Mz_light.toFixed(0), Mz_heavy: +Mz_heavy.toFixed(0),
    });
  }
  return data;
}

function MzTab() {
  const mzData = useMemo(() => gMzTrace(), []);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
      <Sec title="Aligning Moment Mz vs Slip Angle [Nm]">
        <GC><ResponsiveContainer width="100%" height={280}>
          <LineChart data={mzData} margin={{ top: 8, right: 16, bottom: 20, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis dataKey="alpha" {...AX} label={{ value: "Slip Angle α [°]", position: "bottom", fill: C.dm, fontSize: 9 }}/>
            <YAxis {...AX} label={{ value: "Mz [Nm]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }}/>
            <Tooltip contentStyle={TT}/>
            <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3"/>
            <ReferenceLine x={0} stroke={C.dm} strokeDasharray="3 3"/>
            <Line dataKey="Mz" stroke={C.cy} strokeWidth={2} dot={false} name="Mz (nominal Fz)"/>
            <Line dataKey="Mz_light" stroke={C.gn} strokeWidth={1.2} dot={false} name="Mz (light load)" strokeDasharray="4 2"/>
            <Line dataKey="Mz_heavy" stroke={C.am} strokeWidth={1.2} dot={false} name="Mz (heavy load)" strokeDasharray="4 2"/>
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }}/>
          </LineChart>
        </ResponsiveContainer></GC>
      </Sec>

      <Sec title="Pneumatic Trail vs Slip Angle [mm]">
        <GC><ResponsiveContainer width="100%" height={280}>
          <ComposedChart data={mzData} margin={{ top: 8, right: 16, bottom: 20, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS}/>
            <XAxis dataKey="alpha" {...AX} label={{ value: "Slip Angle α [°]", position: "bottom", fill: C.dm, fontSize: 9 }}/>
            <YAxis {...AX} label={{ value: "Trail [mm]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }}/>
            <Tooltip contentStyle={TT}/>
            <ReferenceLine y={0} stroke={C.red} strokeDasharray="4 4" label={{ value: "ZERO TRAIL = LIMIT", fill: C.red, fontSize: 7 }}/>
            <Area dataKey="trail" stroke={C.cy} fill={`${C.cy}12`} strokeWidth={2} dot={false} name="Pneumatic Trail"/>
          </ComposedChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "6px 8px", lineHeight: 1.6 }}>
          Trail drops to zero at peak Fy — the driver feels this as loss of self-aligning torque.
          Below zero, the contact patch centre has migrated past the tread centreline.
        </div></GC>
      </Sec>
    </div>
  );
}
function gSlipTargetData() {
  const data = [];
  for (let i = 0; i < 100; i++) {
    const alpha = (i / 99) * 16; // 0-16 deg
    const alphaRad = alpha * Math.PI / 180;
 
    // Fy from Pacejka
    const Fy = 3000 * Math.sin(1.4 * Math.atan(12 * alphaRad));
    const muEff = Fy / 735; // normalized by nominal Fz
 
    // MPC target slip (slightly below peak for safety margin)
    const peakAlpha = 7.2; // degrees — peak slip angle
    const mpcTarget = alpha <= peakAlpha ? muEff * 0.98 : muEff * 0.92; // backs off past peak
 
    // Actual operating slip (with noise — from sim data)
    const R = Math.sin(i * 17.3) * 0.5 + 0.5; // deterministic pseudo-random
    const actual = muEff * (0.85 + R * 0.12);
 
    data.push({
      alpha: +alpha.toFixed(1),
      muCurve: +muEff.toFixed(3),
      mpcTarget: alpha < 14 ? +mpcTarget.toFixed(3) : null,
      actual: alpha > 1 && alpha < 13 ? +actual.toFixed(3) : null,
      gap: alpha > 1 && alpha < 13 ? +((mpcTarget - actual) * 100).toFixed(1) : null,
    });
  }
  return data;
}
 
function SlipTargetTab() {
  const data = React.useMemo(() => gSlipTargetData(), []);
  const peakMu = Math.max(...data.map(d => d.muCurve));
  const peakAlpha = data.find(d => d.muCurve >= peakMu - 0.001)?.alpha || 7;
 
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
      <Sec title="μ-Slip Curve — MPC Target vs Actual Operating Point">
        <GC><ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 24, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} />
            <XAxis dataKey="alpha" {...AX} label={{ value: "Slip Angle α [°]", position: "bottom", fill: C.dm, fontSize: 9 }} />
            <YAxis {...AX} domain={[0, "auto"]} label={{ value: "μ_eff (Fy/Fz)", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
            <Tooltip contentStyle={TT} />
            <ReferenceLine x={peakAlpha} stroke={C.red} strokeDasharray="4 2" label={{ value: `peak α=${peakAlpha}°`, fill: C.red, fontSize: 7 }} />
            <ReferenceArea x1={peakAlpha} x2={16} fill={C.red} fillOpacity={0.03} label={{ value: "SATURATION", fill: C.red, fontSize: 7, position: "insideTop" }} />
            <Line dataKey="muCurve" stroke={C.dm} strokeWidth={2.5} dot={false} name="μ(α) Pacejka" strokeDasharray="6 3" />
            <Line dataKey="mpcTarget" stroke={C.gn} strokeWidth={2} dot={false} name="MPC Target" />
            <Scatter dataKey="actual" data={data.filter(d => d.actual != null)} fill={C.cy} fillOpacity={0.5} r={3} name="Actual Operating" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
          </ComposedChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "6px 8px", lineHeight: 1.6 }}>
          The MPC targets 98% of peak grip below the peak slip angle, backing off to 92% in the
          post-peak saturation zone. The gap between green (target) and cyan dots (actual) represents
          untapped performance — driver skill or controller latency.
        </div></GC>
      </Sec>
 
      <Sec title="Grip Utilization Gap [%]">
        <GC><ResponsiveContainer width="100%" height={300}>
          <AreaChart data={data.filter(d => d.gap != null)} margin={{ top: 8, right: 16, bottom: 24, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} />
            <XAxis dataKey="alpha" {...AX} label={{ value: "Slip Angle α [°]", position: "bottom", fill: C.dm, fontSize: 9 }} />
            <YAxis {...AX} label={{ value: "Gap [%]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
            <Tooltip contentStyle={TT} />
            <ReferenceLine y={0} stroke={C.dm} />
            <ReferenceArea y1={-5} y2={5} fill={C.gn} fillOpacity={0.04} label={{ value: "±5% tolerance", fill: C.gn, fontSize: 7 }} />
            <Area dataKey="gap" stroke={C.am} fill={`${C.am}12`} strokeWidth={1.5} dot={false} name="Target − Actual [%]" />
          </AreaChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "6px 8px", lineHeight: 1.6 }}>
          Positive gap = leaving grip on the table (conservative). Negative gap = exceeding target
          (risk of saturation). The green band shows the ±5% acceptable tolerance.
        </div></GC>
      </Sec>
    </div>
  );
}
 
 
// ── Relaxation Length Tab ────────────────────────────────────────────
 
function gRelaxData(n = 200) {
  const data = [];
  // Step steer input: 0→8° at t=0.5s
  for (let i = 0; i < n; i++) {
    const t = i * 0.01; // 0-2s at 100Hz
    const alphaInput = t < 0.5 ? 0 : 8.0; // step input [deg]
 
    // Steady-state Fy
    const Fy_ss = 3000 * Math.sin(1.4 * Math.atan(12 * (alphaInput * Math.PI / 180)));
 
    // Transient response with relaxation length σ
    // τ = σ_α / V_x, Fy_transient = Fy_ss * (1 - exp(-(t-0.5)/τ))
    const sigma_fast = 0.15; // m — low relaxation length (stiff carcass)
    const sigma_slow = 0.45; // m — high relaxation length (soft carcass)
    const Vx = 15; // m/s
    const tau_fast = sigma_fast / Vx;
    const tau_slow = sigma_slow / Vx;
 
    const Fy_fast = t < 0.5 ? 0 : Fy_ss * (1 - Math.exp(-(t - 0.5) / tau_fast));
    const Fy_slow = t < 0.5 ? 0 : Fy_ss * (1 - Math.exp(-(t - 0.5) / tau_slow));
    const Fy_actual = t < 0.5 ? 0 : Fy_ss * (1 - Math.exp(-(t - 0.5) / (0.25 / Vx))); // σ=0.25m actual
 
    data.push({
      t: +t.toFixed(3),
      alphaInput: +alphaInput.toFixed(1),
      Fy_ss: +Fy_ss.toFixed(0),
      Fy_fast: +Fy_fast.toFixed(0),
      Fy_slow: +Fy_slow.toFixed(0),
      Fy_actual: +Fy_actual.toFixed(0),
    });
  }
  return data;
}
 
function gRelaxSweep() {
  // Relaxation length vs speed
  const data = [];
  for (let v = 2; v <= 30; v += 1) {
    const sigma = 0.25; // m — constant carcass property
    const tau = sigma / v; // time constant decreases with speed
    const tSettle = tau * 3; // 95% settling time
    const f3dB = 1 / (2 * Math.PI * tau); // bandwidth
    data.push({
      speed: v,
      tau: +(tau * 1000).toFixed(1), // ms
      tSettle: +(tSettle * 1000).toFixed(0), // ms
      bandwidth: +f3dB.toFixed(1), // Hz
    });
  }
  return data;
}
 
function RelaxTab() {
  const relaxData = React.useMemo(() => gRelaxData(), []);
  const relaxSweep = React.useMemo(() => gRelaxSweep(), []);
 
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
      <Sec title="Step Steer Response — Effect of Relaxation Length">
        <GC><ResponsiveContainer width="100%" height={280}>
          <LineChart data={relaxData} margin={{ top: 8, right: 16, bottom: 24, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} />
            <XAxis dataKey="t" {...AX} label={{ value: "Time [s]", position: "bottom", fill: C.dm, fontSize: 9 }} />
            <YAxis {...AX} label={{ value: "Fy [N]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
            <Tooltip contentStyle={TT} />
            <ReferenceLine x={0.5} stroke={C.dm} strokeDasharray="3 3" label={{ value: "step input", fill: C.dm, fontSize: 7 }} />
            <Line dataKey="Fy_ss" stroke={C.dm} strokeWidth={1} dot={false} name="Fy steady-state" strokeDasharray="6 3" />
            <Line dataKey="Fy_fast" stroke={C.gn} strokeWidth={1.5} dot={false} name="σ=0.15m (stiff)" />
            <Line dataKey="Fy_actual" stroke={C.cy} strokeWidth={2} dot={false} name="σ=0.25m (actual)" />
            <Line dataKey="Fy_slow" stroke={C.red} strokeWidth={1.5} dot={false} name="σ=0.45m (soft)" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
          </LineChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "6px 8px", lineHeight: 1.6 }}>
          The relaxation length σ_α determines how quickly lateral force builds after a steering input.
          At V=15 m/s with σ=0.25m, the time constant τ = σ/V = 16.7ms (95% settling in ~50ms).
          This lag directly explains the hysteresis loop width in the Slip Hysteresis tab.
        </div></GC>
      </Sec>
 
      <Sec title="Time Constant & Bandwidth vs Speed">
        <GC><ResponsiveContainer width="100%" height={280}>
          <ComposedChart data={relaxSweep} margin={{ top: 8, right: 40, bottom: 24, left: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GS} />
            <XAxis dataKey="speed" {...AX} label={{ value: "Speed [m/s]", position: "bottom", fill: C.dm, fontSize: 9 }} />
            <YAxis yAxisId="tau" {...AX} label={{ value: "τ [ms]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
            <YAxis yAxisId="bw" orientation="right" {...AX} label={{ value: "f_3dB [Hz]", angle: 90, position: "insideRight", fill: C.dm, fontSize: 9 }} />
            <Tooltip contentStyle={TT} />
            <Line yAxisId="tau" dataKey="tau" stroke={C.cy} strokeWidth={2} dot={false} name="τ [ms]" />
            <Line yAxisId="tau" dataKey="tSettle" stroke={C.am} strokeWidth={1.5} dot={false} name="t_settle [ms]" strokeDasharray="4 2" />
            <Line yAxisId="bw" dataKey="bandwidth" stroke={C.gn} strokeWidth={2} dot={false} name="f_3dB [Hz]" />
            <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
          </ComposedChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "6px 8px", lineHeight: 1.6 }}>
          At low speed (parking lot, 3 m/s): τ=83ms, bandwidth=1.9Hz — the tire feels sluggish.
          At high speed (straight, 25 m/s): τ=10ms, bandwidth=15.9Hz — nearly instant response.
          The WMPC accounts for this speed-dependent lag in its prediction horizon.
        </div></GC>
      </Sec>
    </div>
  );
}

function gWearData(nLaps = 22) {
  const R = (seed) => { let s = seed; return () => { s = (s * 16807) % 2147483647; return (s & 0x7fffffff) / 0x7fffffff; }; };
  const rng = R(777);
  const data = [];
  let tread = [4.0, 4.0, 4.0, 4.0]; // mm tread depth per corner
  let grip = [100, 100, 100, 100]; // % of peak
  for (let lap = 0; lap <= nLaps; lap++) {
    // Wear model: front wears faster (steering + braking), outside (FL for RHD tracks) fastest
    const wearRate = [0.065, 0.058, 0.048, 0.042]; // mm/lap base rate
    for (let c = 0; c < 4; c++) {
      const lapWear = wearRate[c] * (0.9 + rng() * 0.2);
      tread[c] = Math.max(0.5, tread[c] - lapWear);
      // Grip correlates with tread: plateau 4.0-2.0mm, then cliff below 1.5mm
      grip[c] = tread[c] > 2.0 ? 100 - (4.0 - tread[c]) * 1.5 : 100 - 3.0 - (2.0 - tread[c]) * 15;
    }
    data.push({
      lap, dist: +(lap * 1.37).toFixed(1),
      tFL: +tread[0].toFixed(2), tFR: +tread[1].toFixed(2), tRL: +tread[2].toFixed(2), tRR: +tread[3].toFixed(2),
      gFL: +grip[0].toFixed(1), gFR: +grip[1].toFixed(1), gRL: +grip[2].toFixed(1), gRR: +grip[3].toFixed(1),
      avgGrip: +((grip[0] + grip[1] + grip[2] + grip[3]) / 4).toFixed(1),
    });
  }
  return data;
}
 
function WearTab() {
  const wearData = React.useMemo(() => gWearData(), []);
  const finalGrip = wearData[wearData.length - 1]?.avgGrip || 100;
  const finalTread = Math.min(
    wearData[wearData.length - 1]?.tFL || 4,
    wearData[wearData.length - 1]?.tFR || 4,
    wearData[wearData.length - 1]?.tRL || 4,
    wearData[wearData.length - 1]?.tRR || 4,
  );
  const cliffLap = wearData.findIndex(d => d.avgGrip < 90);
 
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Final Grip" value={`${finalGrip.toFixed(1)}%`} sub="end of 22km" sentiment={finalGrip > 90 ? "positive" : finalGrip > 85 ? "amber" : "negative"} delay={0} />
        <KPI label="Min Tread" value={`${finalTread.toFixed(1)}mm`} sub={finalTread > 1.5 ? "above cliff" : "IN CLIFF ZONE"} sentiment={finalTread > 1.5 ? "positive" : "negative"} delay={1} />
        <KPI label="Cliff Onset" value={cliffLap > 0 ? `Lap ${cliffLap}` : "None"} sub="grip < 90%" sentiment={cliffLap < 0 || cliffLap > 18 ? "positive" : "amber"} delay={2} />
        <KPI label="Grip Lost" value={`${(100 - finalGrip).toFixed(1)}%`} sub="total degradation" sentiment={(100 - finalGrip) < 8 ? "positive" : "amber"} delay={3} />
      </div>
 
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="Tread Depth per Corner [mm]">
          <GC><ResponsiveContainer width="100%" height={240}>
            <LineChart data={wearData} margin={{ top: 8, right: 16, bottom: 20, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="dist" {...AX} label={{ value: "Distance [km]", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis {...AX} domain={[0, 4.5]} label={{ value: "Tread [mm]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <ReferenceArea y1={0} y2={1.5} fill={C.red} fillOpacity={0.05} label={{ value: "CLIFF", fill: C.red, fontSize: 7 }} />
              <Line dataKey="tFL" stroke={C.cy} strokeWidth={1.5} dot={false} name="FL" />
              <Line dataKey="tFR" stroke={C.gn} strokeWidth={1.5} dot={false} name="FR" />
              <Line dataKey="tRL" stroke={C.am} strokeWidth={1.5} dot={false} name="RL" />
              <Line dataKey="tRR" stroke={C.red} strokeWidth={1.5} dot={false} name="RR" />
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
            </LineChart>
          </ResponsiveContainer></GC>
        </Sec>
 
        <Sec title="Grip Level per Corner [%]">
          <GC><ResponsiveContainer width="100%" height={240}>
            <LineChart data={wearData} margin={{ top: 8, right: 16, bottom: 20, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="dist" {...AX} label={{ value: "Distance [km]", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis {...AX} domain={[70, 102]} label={{ value: "Grip [%]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <ReferenceLine y={90} stroke={C.am} strokeDasharray="4 2" label={{ value: "90% threshold", fill: C.am, fontSize: 7 }} />
              <Line dataKey="gFL" stroke={C.cy} strokeWidth={1.5} dot={false} name="FL" />
              <Line dataKey="gFR" stroke={C.gn} strokeWidth={1.5} dot={false} name="FR" />
              <Line dataKey="gRL" stroke={C.am} strokeWidth={1.5} dot={false} name="RL" />
              <Line dataKey="gRR" stroke={C.red} strokeWidth={1.5} dot={false} name="RR" />
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
            </LineChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "6px 8px", lineHeight: 1.6 }}>
            The R25B tread exhibits a plateau region (4.0→2.0mm) where grip loss is linear (~1.5%/mm).
            Below 1.5mm, the compound enters the "cliff" — grip degrades 15%/mm due to reduced tread
            block deformation and thermal mass loss. The front-left wears fastest due to combined
            steering scrub + braking load transfer on right-hand dominant tracks.
          </div></GC>
        </Sec>
      </div>
    </div>
  );
}
 
/*--- END WearTab ---*/


function gInducingPoints(n = 50) {
  // Mirrors tire_model.py: Z = tanh(Z_raw) * scale + shift
  // 5D: alpha, kappa, gamma, Fz, Vx
  const R = (seed) => { let s = seed; return () => { s = (s * 16807) % 2147483647; return (s & 0x7fffffff) / 0x7fffffff; }; };
  const rng = R(8888);
  const pts = [];
  for (let i = 0; i < n; i++) {
    const alpha = Math.tanh((rng() - 0.5) * 2) * 14.3;  // ±14.3° (0.25 rad)
    const kappa = Math.tanh((rng() - 0.5) * 2) * 0.20;  // ±0.20
    const gamma = Math.tanh((rng() - 0.5) * 2) * 4.6;   // ±4.6° (0.08 rad)
    const Fz = 400 + rng() * 800;                         // 400-1200 N
    const Vx = 2 + rng() * 20;                            // 2-22 m/s
    // Posterior variance — lower near points, higher in gaps
    const sigma = 0.02 + rng() * 0.06;
    pts.push({
      alpha: +alpha.toFixed(1), kappa: +kappa.toFixed(3),
      gamma: +gamma.toFixed(1), Fz: +Fz.toFixed(0), Vx: +Vx.toFixed(1),
      sigma: +sigma.toFixed(3),
    });
  }
  return pts;
}
 
function gCoverageGrid(inducingPts, res = 30) {
  // Compute variance proxy on (alpha, kappa) grid
  const data = [];
  for (let ai = 0; ai < res; ai++) {
    for (let ki = 0; ki < res; ki++) {
      const alpha = (ai / (res - 1)) * 28 - 14; // -14 to +14
      const kappa = (ki / (res - 1)) * 0.4 - 0.2; // -0.2 to +0.2
      // Approximate GP variance: distance to nearest inducing point
      let minDist = Infinity;
      for (const pt of inducingPts) {
        const da = (alpha - pt.alpha) / 14;
        const dk = (kappa - pt.kappa) / 0.2;
        const dist = Math.sqrt(da * da + dk * dk);
        if (dist < minDist) minDist = dist;
      }
      const variance = Math.min(1, minDist * 2.5);
      data.push({ ai, ki, alpha: +alpha.toFixed(1), kappa: +kappa.toFixed(3), variance: +variance.toFixed(3) });
    }
  }
  return data;
}
 
function GPCoverageTab() {
  const inducingPts = React.useMemo(() => gInducingPoints(), []);
  const coverageGrid = React.useMemo(() => gCoverageGrid(inducingPts), [inducingPts]);
  const meanSigma = inducingPts.reduce((a, p) => a + p.sigma, 0) / inducingPts.length;
  const highUncertainty = coverageGrid.filter(d => d.variance > 0.6).length;
 
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Inducing Points" value="50" sub="Sparse GP" sentiment="neutral" delay={0} />
        <KPI label="Mean σ" value={meanSigma.toFixed(3)} sub="posterior std" sentiment={meanSigma < 0.05 ? "positive" : "amber"} delay={1} />
        <KPI label="Coverage Gaps" value={`${((highUncertainty / coverageGrid.length) * 100).toFixed(0)}%`} sub="high uncertainty cells" sentiment={highUncertainty / coverageGrid.length < 0.2 ? "positive" : "amber"} delay={2} />
        <KPI label="Input Dim" value="5D" sub="α, κ, γ, Fz, Vx" sentiment="neutral" delay={3} />
      </div>
 
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="Inducing Point Distribution (α vs κ)">
          <GC><ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 12, right: 16, bottom: 24, left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis type="number" dataKey="alpha" {...AX} domain={[-16, 16]}
                label={{ value: "Slip Angle α [°]", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis type="number" dataKey="kappa" {...AX} domain={[-0.25, 0.25]}
                label={{ value: "Slip Ratio κ", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <ReferenceLine x={0} stroke={C.dm} strokeDasharray="3 3" />
              <ReferenceLine y={0} stroke={C.dm} strokeDasharray="3 3" />
              {/* Operating envelope */}
              <ReferenceArea x1={-10} x2={10} y1={-0.15} y2={0.15} fill={C.gn} fillOpacity={0.03}
                label={{ value: "Typical FS envelope", fill: C.gn, fontSize: 7, position: "insideTopLeft" }} />
              <Scatter data={inducingPts} name="Inducing Points">
                {inducingPts.map((pt, i) => (
                  <Cell key={i} fill={pt.sigma < 0.04 ? C.gn : pt.sigma < 0.06 ? C.am : C.red}
                    fillOpacity={0.7} r={5} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "6px 8px", lineHeight: 1.6 }}>
            Each dot is an inducing point of the Sparse GP. <span style={{ color: C.gn }}>Green</span> = low σ (confident).
            <span style={{ color: C.red }}> Red</span> = high σ (uncertain). The green box shows the typical FS operating
            envelope. Points outside it provide extrapolation safety.
            After BUGFIX-4, inducing points use tanh mapping for symmetric coverage.
          </div></GC>
        </Sec>
 
        <Sec title="Uncertainty Heatmap (α × κ grid)">
          <GC style={{ padding: "8px" }}>
            <div style={{ display: "grid", gridTemplateColumns: `repeat(30, 1fr)`, gap: 0, aspectRatio: "1" }}>
              {coverageGrid.map((d, i) => (
                <div key={i} style={{
                  background: d.variance < 0.3 ? `rgba(35,209,96,${0.2 + d.variance})`
                    : d.variance < 0.6 ? `rgba(255,176,32,${0.1 + d.variance * 0.6})`
                    : `rgba(255,56,56,${0.1 + d.variance * 0.5})`,
                  aspectRatio: "1",
                }} title={`α=${d.alpha}° κ=${d.kappa} σ²=${d.variance}`} />
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, fontSize: 7, fontFamily: C.dt, color: C.dm, padding: "0 2px" }}>
              <span>α = -14°</span><span>α = 0°</span><span>α = +14°</span>
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 8, fontSize: 8, fontFamily: C.dt }}>
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: C.gn }} />
                <span style={{ color: C.dm }}>Low σ² (confident)</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: C.am }} />
                <span style={{ color: C.dm }}>Medium</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: C.red }} />
                <span style={{ color: C.dm }}>High σ² (blind spot)</span>
              </div>
            </div>
          </GC>
        </Sec>
      </div>
    </div>
  );
}
 
 
// ── Temperature vs Grip Correlation Tab ─────────────────────────────
 
function gTempGripData() {
  const data = [];
  for (let T = 20; T <= 140; T += 2) {
    const T_opt = 90; // from tire_coeffs.py T_OPT
    const beta = 0.0008; // from tire_coeffs.py BETA_T
    // Thermal grip factor: exp(-beta * (T - T_opt)^2)
    const gripFactor = Math.exp(-beta * (T - T_opt) * (T - T_opt));
    // Per-node breakdown
    const surfaceGrip = gripFactor * (1 + 0.02 * Math.sin((T - 60) * 0.1));
    const bulkGrip = gripFactor * (1 - 0.01 * Math.max(0, T - 110) / 30);
    // Viscous vs hysteretic components
    const viscous = 0.4 * Math.exp(-0.001 * (T - 80) * (T - 80));
    const hysteretic = 0.6 * Math.exp(-0.0005 * (T - 95) * (T - 95));
 
    data.push({
      T, gripFactor: +gripFactor.toFixed(4),
      gripPct: +(gripFactor * 100).toFixed(1),
      surfaceGrip: +(surfaceGrip * 100).toFixed(1),
      bulkGrip: +(bulkGrip * 100).toFixed(1),
      viscous: +(viscous * 100).toFixed(1),
      hysteretic: +(hysteretic * 100).toFixed(1),
      combined: +((viscous + hysteretic) * 100 / 1.0).toFixed(1),
    });
  }
  return data;
}
 
function TempGripTab() {
  const tempData = React.useMemo(() => gTempGripData(), []);
  const peakGrip = tempData.reduce((best, d) => d.gripPct > best.gripPct ? d : best, tempData[0]);
  const window90 = tempData.filter(d => d.gripPct >= 90);
  const windowLow = window90.length > 0 ? window90[0].T : 0;
  const windowHigh = window90.length > 0 ? window90[window90.length - 1].T : 0;
 
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="T_optimal" value={`${peakGrip.T}°C`} sub="peak grip temperature" sentiment="positive" delay={0} />
        <KPI label="Peak Grip" value={`${peakGrip.gripPct}%`} sub="at T_opt" sentiment="positive" delay={1} />
        <KPI label="90% Window" value={`${windowLow}-${windowHigh}°C`} sub={`${windowHigh - windowLow}°C range`} sentiment={windowHigh - windowLow > 40 ? "positive" : "amber"} delay={2} />
        <KPI label="β (width)" value="0.0008" sub="K⁻² Gaussian" sentiment="neutral" delay={3} />
      </div>
 
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <Sec title="Thermal Grip Factor vs Temperature">
          <GC><ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={tempData} margin={{ top: 8, right: 16, bottom: 24, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="T" {...AX} label={{ value: "Temperature [°C]", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis {...AX} domain={[0, 105]} label={{ value: "Grip [%]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <ReferenceArea x1={windowLow} x2={windowHigh} fill={C.gn} fillOpacity={0.06}
                label={{ value: "90% window", fill: C.gn, fontSize: 7, position: "insideTop" }} />
              <ReferenceLine x={90} stroke={C.gn} strokeDasharray="4 2" label={{ value: "T_opt=90°C", fill: C.gn, fontSize: 7 }} />
              <Area dataKey="gripPct" stroke={C.cy} fill={`${C.cy}15`} strokeWidth={2.5} dot={false} name="Grip Factor [%]" />
            </ComposedChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "6px 8px", lineHeight: 1.6 }}>
            Gaussian thermal model: μ(T) = μ_peak · exp(−β·(T − T_opt)²).
            β = 0.0008 K⁻² from tire_coeffs.py.
            The green band shows the temperature window where grip remains above 90%.
            Below 55°C: compound not activated ("cold tire"). Above 125°C: thermal degradation.
          </div></GC>
        </Sec>
 
        <Sec title="Viscous vs Hysteretic Grip Components">
          <GC><ResponsiveContainer width="100%" height={280}>
            <AreaChart data={tempData} margin={{ top: 8, right: 16, bottom: 24, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GS} />
              <XAxis dataKey="T" {...AX} label={{ value: "Temperature [°C]", position: "bottom", fill: C.dm, fontSize: 9 }} />
              <YAxis {...AX} domain={[0, 70]} label={{ value: "Contribution [%]", angle: -90, position: "insideLeft", fill: C.dm, fontSize: 9 }} />
              <Tooltip contentStyle={TT} />
              <Area type="monotone" dataKey="viscous" stackId="1" stroke={C.cy} fill={`${C.cy}20`} strokeWidth={1.5} name="Viscous (low-T dominant)" />
              <Area type="monotone" dataKey="hysteretic" stackId="1" stroke={C.am} fill={`${C.am}20`} strokeWidth={1.5} name="Hysteretic (high-T dominant)" />
              <Legend wrapperStyle={{ fontSize: 8, fontFamily: C.hd }} />
            </AreaChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, padding: "6px 8px", lineHeight: 1.6 }}>
            At low temperatures, viscous friction dominates (rubber flows into surface asperities).
            At high temperatures, hysteretic friction dominates (bulk deformation energy loss).
            The crossover around 85°C explains why optimal grip is at ~90°C — both mechanisms
            contribute simultaneously at the peak.
          </div></GC>
        </Sec>
      </div>
    </div>
  );
}

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
      {tab === "mz"         && <MzTab />}
      {tab === "slipTarget" && <SlipTargetTab />}
      {tab === "relax"      && <RelaxTab />}
      {tab === "wear" && <WearTab />}
      {tab === "gpCoverage" && <GPCoverageTab />}
      {tab === "tempGrip"   && <TempGripTab />}
    </div>
  );
}