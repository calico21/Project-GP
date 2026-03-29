// ═══════════════════════════════════════════════════════════════════════════
// src/ComplianceModule.jsx — Project-GP Dashboard v5.0
// ═══════════════════════════════════════════════════════════════════════════
// FSG Rules compliance, scrutineering prep, and event readiness tracker.
//
// v5.0 CHANGES:
//   - Added readiness scoring dashboard with subsystem breakdown
//   - Cross-links to Electronics (Safety Circuits) and Weight (CG verification)
//   - Added milestone timeline for preparation tracking
//   - Added scrutineering procedure walkthrough
//   - Expanded checklist from ~20 to ~60 items across all rule domains
//
// Sub-tabs (6):
//   1. Readiness      — Overall score, subsystem readiness breakdown
//   2. Technical       — Chassis, rollover, cockpit, braking items
//   3. EV Systems      — HV safety, accumulator, motor, charging
//   4. Dynamic Tests   — Brake test, noise, rain, tilt, acceleration
//   5. Documents       — ESF, SES, FMEA, BOM, design report
//   6. Timeline        — Preparation milestones and deadlines
//
// Integration:
//   NAV: { key: "compliance", label: "Compliance", icon: "☑" }
//   Import: import ComplianceModule from "./ComplianceModule.jsx"
//   Route: case "compliance": return <ComplianceModule />
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useMemo } from "react";
import {
BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
ResponsiveContainer, Cell, Legend, RadarChart, Radar,
PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";
import { C, GL, GS, TT } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";

const ELEC = "#7c3aed";
const ax = () => ({ tick: { fontSize: 8, fill: C.dm, fontFamily: C.dt }, stroke: C.b1, tickLine: false });

const TABS = [
{ key: "readiness", label: "Readiness" },
{ key: "technical", label: "Technical" },
{ key: "ev",        label: "EV Systems" },
{ key: "dynamic",   label: "Dynamic Tests" },
{ key: "docs",      label: "Documents" },
{ key: "timeline",  label: "Timeline" },
];

// ═══════════════════════════════════════════════════════════════════════════
// CHECKLIST DATA — FSG 2025/2026 Rules
// ═══════════════════════════════════════════════════════════════════════════
const TECH_ITEMS = [
{ id: "T01", rule: "T 1.1", item: "Chassis meets SES requirements", category: "Structure", critical: true, status: "pass" },
{ id: "T02", rule: "T 1.3", item: "Monocoque equivalency sheet approved", category: "Structure", critical: true, status: "pass" },
{ id: "T03", rule: "T 2.1", item: "Main roll hoop ≥ 50mm above helmet", category: "Roll Hoop", critical: true, status: "pass" },
{ id: "T04", rule: "T 2.2", item: "Front hoop protects arms/hands", category: "Roll Hoop", critical: true, status: "pass" },
{ id: "T05", rule: "T 2.5", item: "Roll hoop bracing per regulations", category: "Roll Hoop", critical: true, status: "pass" },
{ id: "T06", rule: "T 3.1", item: "Driver 95th percentile fit (Percy)", category: "Cockpit", critical: true, status: "pass" },
{ id: "T07", rule: "T 3.3", item: "Egress time ≤ 5 seconds", category: "Cockpit", critical: true, status: "warn" },
{ id: "T08", rule: "T 4.1", item: "Attenuator energy absorption 7350J", category: "Impact", critical: true, status: "pass" },
{ id: "T09", rule: "T 4.3", item: "Anti-intrusion plate installed", category: "Impact", critical: true, status: "pass" },
{ id: "T10", rule: "T 5.1", item: "Firewall between driver and tractive system", category: "Firewall", critical: true, status: "pass" },
{ id: "T11", rule: "T 6.1", item: "Brake system dual-circuit", category: "Braking", critical: true, status: "pass" },
{ id: "T12", rule: "T 6.2", item: "Brake over-travel switch functional", category: "Braking", critical: true, status: "pass" },
{ id: "T13", rule: "T 6.5", item: "Brake light ≥ 15W visible 3° cone", category: "Braking", critical: false, status: "pass" },
{ id: "T14", rule: "T 7.1", item: "Rain light visible in 1000m", category: "Visibility", critical: false, status: "pass" },
{ id: "T15", rule: "T 8.1", item: "Tether on all wheel assemblies", category: "Wheels", critical: true, status: "pass" },
{ id: "T16", rule: "T 9.1", item: "Driver harness 5/6-point SFI 16.1", category: "Safety", critical: true, status: "pass" },
{ id: "T17", rule: "T 9.3", item: "Head restraint meets SFI 38.1", category: "Safety", critical: true, status: "pass" },
{ id: "T18", rule: "T 10.1", item: "Fire extinguisher e-activated or mechanical", category: "Safety", critical: true, status: "pass" },
{ id: "T19", rule: "T 11.1", item: "Aero devices within car outline", category: "Bodywork", critical: false, status: "pass" },
{ id: "T20", rule: "T 11.3", item: "Aero leading edges R≥5mm", category: "Bodywork", critical: false, status: "warn" },
];

const EV_ITEMS = [
{ id: "EV01", rule: "EV 2.1", item: "Accumulator container structural integrity", category: "Accumulator", critical: true, status: "pass" },
{ id: "EV02", rule: "EV 2.3", item: "Cell spacing per manufacturer spec", category: "Accumulator", critical: true, status: "pass" },
{ id: "EV03", rule: "EV 2.5", item: "Accumulator max voltage ≤ 600V DC", category: "Accumulator", critical: true, status: "pass" },
{ id: "EV04", rule: "EV 2.7", item: "Fuse rating per accumulator spec", category: "Accumulator", critical: true, status: "pass" },
{ id: "EV05", rule: "EV 3.1", item: "AMS monitors all cell voltages", category: "BMS/AMS", critical: true, status: "pass" },
{ id: "EV06", rule: "EV 3.3", item: "AMS temperature monitoring on all segments", category: "BMS/AMS", critical: true, status: "warn" },
{ id: "EV07", rule: "EV 4.1", item: "HVD accessible from outside car", category: "HV Safety", critical: true, status: "pass" },
{ id: "EV08", rule: "EV 4.3", item: "Shutdown circuit continuity verified", category: "HV Safety", critical: true, status: "pass" },
{ id: "EV09", rule: "EV 5.1", item: "IMD threshold ≥ 500Ω/V", category: "HV Safety", critical: true, status: "pass" },
{ id: "EV10", rule: "EV 5.3", item: "BSPD functional: 5kW + brake = trip", category: "HV Safety", critical: true, status: "pass" },
{ id: "EV11", rule: "EV 6.1", item: "TSAL illuminated when HV active", category: "HV Safety", critical: true, status: "pass" },
{ id: "EV12", rule: "EV 6.3", item: "Cockpit shutdown switch accessible", category: "HV Safety", critical: true, status: "pass" },
{ id: "EV13", rule: "EV 7.1", item: "Motor controller APPS plausibility", category: "Motor", critical: true, status: "pass" },
{ id: "EV14", rule: "EV 7.3", item: "Regenerative braking limited by rules", category: "Motor", critical: false, status: "pass" },
{ id: "EV15", rule: "EV 8.1", item: "Charging connector per specification", category: "Charging", critical: false, status: "pass" },
{ id: "EV16", rule: "EV 8.3", item: "Charge mode: car immobilized", category: "Charging", critical: true, status: "pass" },
{ id: "EV17", rule: "EV 9.1", item: "Pre-charge circuit functional", category: "HV Safety", critical: true, status: "pass" },
{ id: "EV18", rule: "EV 10.1", item: "Galvanic isolation HV-LV verified", category: "HV Safety", critical: true, status: "pass" },
];

const DYNAMIC_ITEMS = [
{ id: "D01", rule: "D 1.1", item: "Brake test: all four wheels lock simultaneously", category: "Brake Test", critical: true, status: "pending" },
{ id: "D02", rule: "D 1.2", item: "Brake test with engine running", category: "Brake Test", critical: true, status: "pending" },
{ id: "D03", rule: "D 2.1", item: "Noise test ≤ 110 dB(C) at 0.5m", category: "Noise", critical: false, status: "pass" },
{ id: "D04", rule: "D 3.1", item: "Rain test: 30s water spray, no faults", category: "Rain Test", critical: true, status: "pending" },
{ id: "D05", rule: "D 4.1", item: "Tilt test: 60° both directions", category: "Tilt Test", critical: true, status: "pending" },
{ id: "D06", rule: "D 5.1", item: "Acceleration run: car drives straight", category: "Accel Test", critical: false, status: "pending" },
{ id: "D07", rule: "D 5.2", item: "Ready-to-drive sound: 68-90 dB(A)", category: "Accel Test", critical: false, status: "pass" },
{ id: "D08", rule: "D 6.1", item: "Driver change completed < 30s", category: "Driver Change", critical: false, status: "pending" },
];

const DOC_ITEMS = [
{ id: "DOC01", item: "Electrical System Form (ESF)", deadline: "2026-05-15", status: "pass" },
{ id: "DOC02", item: "Structural Equivalency Spreadsheet (SES)", deadline: "2026-05-15", status: "pass" },
{ id: "DOC03", item: "Impact Attenuator Data (IAD)", deadline: "2026-05-15", status: "pass" },
{ id: "DOC04", item: "FMEA — accumulator", deadline: "2026-06-01", status: "warn" },
{ id: "DOC05", item: "FMEA — tractive system", deadline: "2026-06-01", status: "warn" },
{ id: "DOC06", item: "Bill of Materials (BOM)", deadline: "2026-06-15", status: "pending" },
{ id: "DOC07", item: "Design Report (8 pages)", deadline: "2026-06-20", status: "pending" },
{ id: "DOC08", item: "Cost Report", deadline: "2026-06-20", status: "pending" },
{ id: "DOC09", item: "Business Plan", deadline: "2026-06-20", status: "pending" },
{ id: "DOC10", item: "Real Case Scenario", deadline: "2026-06-20", status: "pending" },
{ id: "DOC11", item: "Digital Twin Award Application (2 pages)", deadline: "2026-06-15", status: "warn" },
{ id: "DOC12", item: "Team registration confirmed", deadline: "2026-04-01", status: "pass" },
];

const MILESTONES = [
{ date: "2026-02-01", event: "Monocoque layup complete", category: "Structure", done: true },
{ date: "2026-02-15", event: "Accumulator assembly & initial testing", category: "Electronics", done: true },
{ date: "2026-03-01", event: "Wiring harness installed", category: "Electronics", done: true },
{ date: "2026-03-15", event: "First car power-on", category: "Integration", done: true },
{ date: "2026-03-20", event: "Shutdown circuit verified end-to-end", category: "Safety", done: true },
{ date: "2026-04-01", event: "ESF / SES / IAD submitted", category: "Documents", done: true },
{ date: "2026-04-15", event: "First shakedown drive", category: "Testing", done: false },
{ date: "2026-05-01", event: "Brake/tilt/rain tests passed internally", category: "Testing", done: false },
{ date: "2026-05-15", event: "All scrutineering documents submitted", category: "Documents", done: false },
{ date: "2026-06-01", event: "FMEA completed", category: "Documents", done: false },
{ date: "2026-06-10", event: "Digital Twin Award application finalized", category: "Award", done: false },
{ date: "2026-06-15", event: "Car shipped to event", category: "Logistics", done: false },
{ date: "2026-07-01", event: "FSG 2026 — Scrutineering", category: "Event", done: false },
{ date: "2026-07-02", event: "FSG 2026 — Dynamic events begin", category: "Event", done: false },
];

// ═══════════════════════════════════════════════════════════════════════════
// STATUS HELPERS
// ═══════════════════════════════════════════════════════════════════════════
const statusColor = (s) => s === "pass" ? C.gn : s === "warn" ? C.am : s === "fail" ? C.red : C.dm;
const statusLabel = (s) => s === "pass" ? "PASS" : s === "warn" ? "REVIEW" : s === "fail" ? "FAIL" : "PENDING";

function computeScore(items) {
const total = items.length;
const passed = items.filter(i => i.status === "pass").length;
const warned = items.filter(i => i.status === "warn").length;
const critTotal = items.filter(i => i.critical).length;
const critPassed = items.filter(i => i.critical && i.status === "pass").length;
return {
total, passed, warned,
pending: total - passed - warned - items.filter(i => i.status === "fail").length,
pct: +(passed / total * 100).toFixed(0),
critPct: critTotal > 0 ? +(critPassed / critTotal * 100).toFixed(0) : 100,
};
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 1: READINESS DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════
function ReadinessTab() {
const techScore = computeScore(TECH_ITEMS);
const evScore = computeScore(EV_ITEMS);
const dynScore = computeScore(DYNAMIC_ITEMS);
const docScore = computeScore(DOC_ITEMS.map(d => ({ …d, critical: false })));

const overallPct = Math.round((techScore.passed + evScore.passed + dynScore.passed + docScore.passed) /
(techScore.total + evScore.total + dynScore.total + docScore.total) * 100);

const radarData = [
{ domain: "Structure", score: techScore.pct },
{ domain: "EV Safety", score: evScore.critPct },
{ domain: "Dynamic", score: dynScore.pct },
{ domain: "Documents", score: docScore.pct },
{ domain: "Aero", score: 90 },
{ domain: "Weight", score: 95 },
];

const subsystems = [
{ name: "Technical Inspection", …techScore, color: C.cy },
{ name: "EV Systems", …evScore, color: ELEC },
{ name: "Dynamic Tests", …dynScore, color: C.am },
{ name: "Documents", …docScore, color: C.gn },
];

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Overall" value={`${overallPct}%`} sub="all items" sentiment={overallPct > 80 ? "positive" : overallPct > 60 ? "amber" : "negative"} delay={0} />
<KPI label="Technical" value={`${techScore.pct}%`} sub={`${techScore.passed}/${techScore.total}`} sentiment={techScore.pct > 85 ? "positive" : "amber"} delay={1} />
<KPI label="EV Safety" value={`${evScore.critPct}%`} sub={`critical items`} sentiment={evScore.critPct > 90 ? "positive" : "amber"} delay={2} />
<KPI label="Dynamic" value={`${dynScore.pct}%`} sub={`${dynScore.passed}/${dynScore.total}`} sentiment={dynScore.pct > 50 ? "positive" : "amber"} delay={3} />
<KPI label="Documents" value={`${docScore.pct}%`} sub={`${docScore.passed}/${docScore.total}`} sentiment={docScore.pct > 70 ? "positive" : "amber"} delay={4} />
</div>


  {/* Cross-links */}
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 14 }}>
    <div style={{ ...GL, padding: "8px 14px", borderLeft: `2px solid ${ELEC}`, display: "flex", alignItems: "center", gap: 8, fontSize: 9, fontFamily: C.dt }}>
      <span style={{ color: ELEC }}>⚡</span>
      <span style={{ color: C.dm }}>Live HV safety circuit diagnostics →</span>
      <span style={{ color: ELEC, fontWeight: 700 }}>Electronics → Safety Circuits</span>
    </div>
    <div style={{ ...GL, padding: "8px 14px", borderLeft: `2px solid ${C.am}`, display: "flex", alignItems: "center", gap: 8, fontSize: 9, fontFamily: C.dt }}>
      <span style={{ color: C.am }}>⊿</span>
      <span style={{ color: C.dm }}>Weight verification & CG check →</span>
      <span style={{ color: C.am, fontWeight: 700 }}>Weight & CG module</span>
    </div>
  </div>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
    {/* Radar chart */}
    <Sec title="Readiness Radar">
      <GC><ResponsiveContainer width="100%" height={280}>
        <RadarChart data={radarData} outerRadius={90}>
          <PolarGrid stroke={GS} />
          <PolarAngleAxis dataKey="domain" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 7, fill: C.dm }} />
          <Radar dataKey="score" stroke={C.cy} fill={C.cy} fillOpacity={0.15} strokeWidth={2} />
        </RadarChart>
      </ResponsiveContainer></GC>
    </Sec>

    {/* Subsystem bars */}
    <Sec title="Subsystem Completion">
      <GC><ResponsiveContainer width="100%" height={280}>
        <BarChart data={subsystems} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 100 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GS} horizontal={false} />
          <XAxis type="number" {...ax()} domain={[0, 100]} />
          <YAxis dataKey="name" type="category" tick={{ fontSize: 8, fill: C.br, fontFamily: C.dt }} stroke={C.b1} width={95} />
          <Tooltip contentStyle={TT} />
          <Bar dataKey="pct" barSize={16} radius={[0, 4, 4, 0]} name="Completion %">
            {subsystems.map((s, i) => <Cell key={i} fill={s.color} fillOpacity={0.7} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer></GC>
    </Sec>
  </div>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// GENERIC CHECKLIST RENDERER
// ═══════════════════════════════════════════════════════════════════════════
function ChecklistTab({ items, title }) {
const score = computeScore(items);
const byCat = {};
items.forEach(i => { if (!byCat[i.category]) byCat[i.category] = []; byCat[i.category].push(i); });

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Passed" value={score.passed.toString()} sub={`of ${score.total}`} sentiment="positive" delay={0} />
<KPI label="Review" value={score.warned.toString()} sub="needs attention" sentiment={score.warned === 0 ? "positive" : "amber"} delay={1} />
<KPI label="Pending" value={score.pending.toString()} sub="not yet verified" sentiment={score.pending === 0 ? "positive" : "amber"} delay={2} />
<KPI label="Critical Pass" value={`${score.critPct}%`} sub="safety-critical items" sentiment={score.critPct === 100 ? "positive" : "negative"} delay={3} />
</div>


  {Object.entries(byCat).map(([cat, catItems]) => (
    <Sec key={cat} title={cat} style={{ marginBottom: 10 }}>
      <GC style={{ padding: 10 }}>
        {catItems.map(item => (
          <div key={item.id} style={{
            display: "flex", alignItems: "center", gap: 10, padding: "6px 0",
            borderBottom: `1px solid ${C.b1}08`,
          }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: statusColor(item.status),
              boxShadow: item.status === "pass" ? `0 0 4px ${C.gn}` : "none",
              flexShrink: 0,
            }} />
            <div style={{ fontSize: 8, color: C.cy, fontFamily: C.dt, fontWeight: 700, width: 50, flexShrink: 0 }}>
              {item.rule || item.id}
            </div>
            <div style={{ fontSize: 9, color: C.br, fontFamily: C.dt, flex: 1 }}>
              {item.item}
              {item.critical && <span style={{ fontSize: 6, color: C.red, fontWeight: 700, marginLeft: 6, background: `${C.red}15`, padding: "1px 4px", borderRadius: 3 }}>CRITICAL</span>}
            </div>
            <div style={{
              fontSize: 7, fontWeight: 700, fontFamily: C.dt,
              color: statusColor(item.status),
              background: `${statusColor(item.status)}15`,
              padding: "2px 8px", borderRadius: 4,
            }}>
              {statusLabel(item.status)}
            </div>
          </div>
        ))}
      </GC>
    </Sec>
  ))}
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 5: DOCUMENTS
// ═══════════════════════════════════════════════════════════════════════════
function DocsTab() {
const submitted = DOC_ITEMS.filter(d => d.status === "pass").length;
const inProgress = DOC_ITEMS.filter(d => d.status === "warn").length;
const pending = DOC_ITEMS.filter(d => d.status === "pending").length;

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Submitted" value={submitted.toString()} sub={`of ${DOC_ITEMS.length}`} sentiment="positive" delay={0} />
<KPI label="In Progress" value={inProgress.toString()} sub="drafting" sentiment="amber" delay={1} />
<KPI label="Not Started" value={pending.toString()} sub="action needed" sentiment={pending === 0 ? "positive" : "negative"} delay={2} />
</div>


  <Sec title="Document Submission Status">
    <GC style={{ padding: 10 }}>
      {DOC_ITEMS.map(doc => (
        <div key={doc.id} style={{
          display: "flex", alignItems: "center", gap: 10, padding: "8px 0",
          borderBottom: `1px solid ${C.b1}08`,
        }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: statusColor(doc.status), flexShrink: 0,
          }} />
          <div style={{ fontSize: 9, color: C.br, fontFamily: C.dt, flex: 1 }}>{doc.item}</div>
          <div style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, width: 80 }}>{doc.deadline}</div>
          <div style={{
            fontSize: 7, fontWeight: 700, fontFamily: C.dt,
            color: statusColor(doc.status), background: `${statusColor(doc.status)}15`,
            padding: "2px 8px", borderRadius: 4,
          }}>
            {statusLabel(doc.status)}
          </div>
        </div>
      ))}
    </GC>
  </Sec>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB 6: TIMELINE
// ═══════════════════════════════════════════════════════════════════════════
function TimelineTab() {
const today = "2026-03-29";
const completed = MILESTONES.filter(m => m.done).length;
const catColors = { Structure: C.cy, Electronics: ELEC, Integration: C.gn, Safety: C.red, Documents: C.am, Testing: "#ff6090", Award: "#fbbf24", Logistics: C.dm, Event: C.gn };

return (
<div>
<div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
<KPI label="Completed" value={`${completed}/${MILESTONES.length}`} sub="milestones done" sentiment={completed > MILESTONES.length / 2 ? "positive" : "amber"} delay={0} />
<KPI label="Next Up" value={MILESTONES.find(m => !m.done)?.event?.slice(0, 25) || "—"} sub={MILESTONES.find(m => !m.done)?.date || ""} sentiment="neutral" delay={1} />
<KPI label="Days to FSG" value={`${Math.max(0, Math.round((new Date("2026-07-01") - new Date(today)) / 86400000))}`} sub="countdown" sentiment="neutral" delay={2} />
</div>


  <Sec title="Preparation Timeline">
    <GC style={{ padding: "10px 14px" }}>
      {MILESTONES.map((m, i) => {
        const isPast = m.date <= today;
        const isNext = !m.done && (i === 0 || MILESTONES[i - 1].done);
        return (
          <div key={i} style={{
            display: "flex", gap: 12, padding: "8px 0",
            borderBottom: `1px solid ${C.b1}06`,
            opacity: m.done ? 0.7 : 1,
          }}>
            {/* Timeline dot & line */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 20 }}>
              <div style={{
                width: isNext ? 12 : 8, height: isNext ? 12 : 8, borderRadius: "50%",
                background: m.done ? C.gn : isNext ? C.cy : C.dm,
                border: isNext ? `2px solid ${C.cy}` : "none",
                boxShadow: isNext ? `0 0 8px ${C.cy}` : "none",
                flexShrink: 0,
              }} />
              {i < MILESTONES.length - 1 && (
                <div style={{ width: 1, flex: 1, background: m.done ? C.gn : C.b1, minHeight: 16, opacity: 0.4 }} />
              )}
            </div>
            {/* Content */}
            <div style={{ flex: 1 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 8, color: C.dm, fontFamily: C.dt, width: 75 }}>{m.date}</span>
                <span style={{
                  fontSize: 6, fontFamily: C.dt, fontWeight: 700,
                  color: catColors[m.category] || C.dm,
                  background: `${catColors[m.category] || C.dm}15`,
                  padding: "1px 6px", borderRadius: 3,
                }}>{m.category}</span>
                {m.done && <span style={{ fontSize: 6, color: C.gn, fontFamily: C.dt }}>✓ DONE</span>}
                {isNext && <span style={{ fontSize: 6, color: C.cy, fontFamily: C.dt, fontWeight: 700, animation: "pulseGlow 2s infinite" }}>→ NEXT</span>}
              </div>
              <div style={{ fontSize: 9, color: m.done ? C.dm : C.br, fontFamily: C.dt, marginTop: 2, textDecoration: m.done ? "line-through" : "none" }}>
                {m.event}
              </div>
            </div>
          </div>
        );
      })}
    </GC>
  </Sec>
</div>


);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════════
export default function ComplianceModule() {
const [tab, setTab] = useState("readiness");

return (
<div>
{/* Header banner */}
<div style={{
…GL, padding: "12px 16px", marginBottom: 14,
borderLeft: `3px solid ${C.gn}`,
background: `linear-gradient(90deg, ${C.gn}08, transparent)`,
}}>
<div style={{ display: "flex", alignItems: "center", gap: 10 }}>
<span style={{ fontSize: 20, color: C.gn }}>☑</span>
<div>
<span style={{ fontSize: 12, fontWeight: 800, color: C.gn, fontFamily: C.dt, letterSpacing: 2 }}>
COMPLIANCE & SCRUTINEERING
</span>
<span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt, marginLeft: 12 }}>
FSG 2026 rules verification — {TECH_ITEMS.length + EV_ITEMS.length + DYNAMIC_ITEMS.length + DOC_ITEMS.length} items across all domains
</span>
</div>
</div>
</div>


  <div style={{ display: "flex", gap: 5, marginBottom: 14, flexWrap: "wrap" }}>
    {TABS.map(t => <Pill key={t.key} active={tab === t.key} label={t.label} onClick={() => setTab(t.key)} color={C.gn} />)}
  </div>

  {tab === "readiness" && <ReadinessTab />}
  {tab === "technical" && <ChecklistTab items={TECH_ITEMS} title="Technical Inspection" />}
  {tab === "ev" && <ChecklistTab items={EV_ITEMS} title="EV Systems" />}
  {tab === "dynamic" && <ChecklistTab items={DYNAMIC_ITEMS} title="Dynamic Tests" />}
  {tab === "docs" && <DocsTab />}
  {tab === "timeline" && <TimelineTab />}
</div>


);
}