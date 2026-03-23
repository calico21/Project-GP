// ═══════════════════════════════════════════════════════════════════════════
// src/ComplianceModule.jsx — Project-GP Dashboard v4.2
// ═══════════════════════════════════════════════════════════════════════════
// FSG Rules compliance checklist & scrutineering preparation tracker.
//
// Integration:
//   NAV: { key: "compliance", label: "Compliance", icon: "☑" }
//   Import: import ComplianceModule from "./ComplianceModule.jsx"
//   Route: case "compliance": return <ComplianceModule />
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState } from "react";
import { C, GL } from "./theme.js";
import { KPI, Sec, GC, Pill } from "./components.jsx";

const TABS = [
  { key: "tech", label: "Technical Inspection" },
  { key: "dynamic", label: "Dynamic Tests" },
  { key: "docs", label: "Documents" },
  { key: "safety", label: "Safety Systems" },
];

// ═══════════════════════════════════════════════════════════════════════════
// CHECKLIST DATA — FSG 2025 Rules
// ═══════════════════════════════════════════════════════════════════════════
const TECH_ITEMS = [
  { id: "T1", rule: "T 1.1", item: "Chassis meets SES requirements", category: "Structure", critical: true },
  { id: "T2", rule: "T 1.3", item: "Monocoque equivalency sheet approved", category: "Structure", critical: true },
  { id: "T3", rule: "T 2.1", item: "Main roll hoop height ≥ 50mm above helmet", category: "Roll Hoop", critical: true },
  { id: "T4", rule: "T 2.2", item: "Front hoop protects arms/hands", category: "Roll Hoop", critical: true },
  { id: "T5", rule: "T 2.5", item: "Roll hoop bracing per regulations", category: "Roll Hoop", critical: true },
  { id: "T6", rule: "T 3.1", item: "Driver 95th percentile fit (Percy)", category: "Cockpit", critical: true },
  { id: "T7", rule: "T 3.3", item: "Egress time < 5 seconds", category: "Cockpit", critical: true },
  { id: "T8", rule: "T 4.1", item: "Minimum wheelbase 1525mm", category: "Dimensions", critical: false },
  { id: "T9", rule: "T 4.2", item: "Minimum track width 75% of wheelbase", category: "Dimensions", critical: false },
  { id: "T10", rule: "T 5.1", item: "Suspension travel ≥ 25mm bump & droop", category: "Suspension", critical: false },
  { id: "T11", rule: "T 5.2", item: "All suspension joints visible/testable", category: "Suspension", critical: false },
  { id: "T12", rule: "T 6.1", item: "Ground clearance variable over range", category: "Chassis", critical: false },
  { id: "T13", rule: "T 7.1", item: "Brake system dual circuit", category: "Brakes", critical: true },
  { id: "T14", rule: "T 7.2", item: "Brake over-travel switch wired to shutdown", category: "Brakes", critical: true },
  { id: "T15", rule: "T 8.1", item: "Exposed edge radius ≥ 3mm", category: "Bodywork", critical: false },
  { id: "T16", rule: "T 8.3", item: "No sharp edges in driver vicinity", category: "Bodywork", critical: false },
];

const DYNAMIC_TESTS = [
  { id: "D1", test: "Tilt Test (60°)", criteria: "No fluid leaks, no wheel lift", category: "Stability", critical: true },
  { id: "D2", test: "Brake Test", criteria: "All 4 wheels lock simultaneously", category: "Braking", critical: true },
  { id: "D3", test: "Rain Test", criteria: "No water ingress to HV components after 2min spray", category: "EV Safety", critical: true },
  { id: "D4", test: "Noise Test", criteria: "< 110 dB(C) at 1m (EV: inverter whine check)", category: "Noise", critical: false },
  { id: "D5", test: "Ready-to-Drive Sound", criteria: "Clearly audible between 1-5m, 68-90 dB(C)", category: "Safety", critical: true },
  { id: "D6", test: "BSPD Test", criteria: "Brake + >5kW = shutdown within 500ms", category: "EV Safety", critical: true },
  { id: "D7", test: "IMD Test", criteria: "Isolation > 500Ω/V, shutdown < 30s", category: "EV Safety", critical: true },
  { id: "D8", test: "Accumulator Isolation", criteria: "HV+ to chassis, HV- to chassis both > 500Ω/V", category: "EV Safety", critical: true },
];

const DOCUMENTS = [
  { id: "DOC1", name: "Structural Equivalency Sheet (SES)", deadline: "8 weeks before", critical: true },
  { id: "DOC2", name: "Electrical System Form (ESF)", deadline: "8 weeks before", critical: true },
  { id: "DOC3", name: "Failure Modes & Effects Analysis (FMEA)", deadline: "4 weeks before", critical: true },
  { id: "DOC4", name: "Design Report (business logic)", deadline: "4 weeks before", critical: false },
  { id: "DOC5", name: "Cost Report", deadline: "4 weeks before", critical: false },
  { id: "DOC6", name: "Impact Attenuator Data (IA)", deadline: "8 weeks before", critical: true },
  { id: "DOC7", name: "Accumulator Design Document", deadline: "8 weeks before", critical: true },
  { id: "DOC8", name: "Real Case Scenario", deadline: "2 weeks before", critical: false },
];

const SAFETY_SYSTEMS = [
  { id: "S1", system: "TSAL (Tractive System Active Light)", rule: "EV 6.1", status: "Flashing when TS active" },
  { id: "S2", system: "Shutdown Circuit (SDC)", rule: "EV 7.1", status: "Series connection of all safety devices" },
  { id: "S3", system: "BSPD", rule: "EV 7.6", status: "Non-programmable, hardwired" },
  { id: "S4", system: "IMD (Insulation Monitoring)", rule: "EV 8.1", status: "Bender ISOMETER or equivalent" },
  { id: "S5", system: "AIRs (Accumulator Isolation Relays)", rule: "EV 5.5", status: "2× normally-open contactors" },
  { id: "S6", system: "Pre-charge Circuit", rule: "EV 5.8", status: "< 60V DC before AIR close" },
  { id: "S7", system: "HVD (HV Disconnect)", rule: "EV 5.3", status: "Accessible without tools" },
  { id: "S8", system: "Accumulator Container", rule: "EV 4.1", status: "IP65, fire-resistant" },
  { id: "S9", system: "Inertia Switch", rule: "T 9.2", status: "Triggers SDC on impact" },
  { id: "S10", system: "Rain Light", rule: "EV 6.4", status: "Red, > 15W equivalent, flashing" },
  { id: "S11", system: "Master Switches", rule: "EV 7.3", status: "Cockpit + external, red spark symbol" },
  { id: "S12", system: "Brake Light", rule: "T 6.5", status: "Visible in bright sunlight" },
];

// ═══════════════════════════════════════════════════════════════════════════
// CHECKLIST COMPONENT
// ═══════════════════════════════════════════════════════════════════════════
function ChecklistItem({ item, checked, onToggle, showRule = true }) {
  return (
    <div onClick={onToggle} style={{
      display: "flex", alignItems: "center", gap: 10, padding: "8px 12px",
      borderBottom: `1px solid ${C.b1}08`, cursor: "pointer",
      background: checked ? `${C.gn}06` : "transparent",
      transition: "background 0.15s",
    }}>
      <div style={{
        width: 18, height: 18, borderRadius: 4, flexShrink: 0,
        border: `2px solid ${checked ? C.gn : C.b1}`,
        background: checked ? C.gn : "transparent",
        display: "flex", alignItems: "center", justifyContent: "center",
        transition: "all 0.15s",
      }}>
        {checked && <span style={{ color: C.bg, fontSize: 11, fontWeight: 800 }}>✓</span>}
      </div>
      {showRule && item.rule && (
        <span style={{ fontSize: 8, fontFamily: C.dt, color: C.dm, fontWeight: 700, letterSpacing: 1, width: 48, flexShrink: 0 }}>
          {item.rule}
        </span>
      )}
      <span style={{
        flex: 1, fontSize: 10, fontFamily: C.dt, color: checked ? C.md : C.br,
        textDecoration: checked ? "line-through" : "none", opacity: checked ? 0.6 : 1,
      }}>
        {item.item || item.test || item.name || item.system}
      </span>
      {item.critical && (
        <span style={{
          fontSize: 7, fontFamily: C.dt, fontWeight: 700, color: C.red,
          background: `${C.red}10`, padding: "2px 6px", borderRadius: 4,
          border: `1px solid ${C.red}20`, letterSpacing: 1,
        }}>
          CRITICAL
        </span>
      )}
      {item.category && (
        <span style={{ fontSize: 7, fontFamily: C.dt, color: C.dm, letterSpacing: 1 }}>
          {item.category}
        </span>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════
function TechTab({ checked, toggle }) {
  const total = TECH_ITEMS.length;
  const done = TECH_ITEMS.filter(i => checked[i.id]).length;
  const critTotal = TECH_ITEMS.filter(i => i.critical).length;
  const critDone = TECH_ITEMS.filter(i => i.critical && checked[i.id]).length;

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Overall" value={`${done}/${total}`} sub={`${(done / total * 100).toFixed(0)}%`} sentiment={done === total ? "positive" : "amber"} delay={0} />
        <KPI label="Critical" value={`${critDone}/${critTotal}`} sub={critDone === critTotal ? "all passed" : `${critTotal - critDone} remaining`} sentiment={critDone === critTotal ? "positive" : "negative"} delay={1} />
        <KPI label="Status" value={critDone === critTotal ? "PASS" : "INCOMPLETE"} sub="tech inspection" sentiment={critDone === critTotal ? "positive" : "negative"} delay={2} />
      </div>
      <GC style={{ padding: 0, overflow: "hidden" }}>
        {TECH_ITEMS.map(item => (
          <ChecklistItem key={item.id} item={item} checked={!!checked[item.id]} onToggle={() => toggle(item.id)} />
        ))}
      </GC>
    </div>
  );
}

function DynamicTab({ checked, toggle }) {
  const total = DYNAMIC_TESTS.length;
  const done = DYNAMIC_TESTS.filter(i => checked[i.id]).length;
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Tests Passed" value={`${done}/${total}`} sub={`${(done / total * 100).toFixed(0)}%`} sentiment={done === total ? "positive" : "amber"} delay={0} />
        <KPI label="Tilt Test" value={checked["D1"] ? "PASS" : "PENDING"} sub="60° no fluid leak" sentiment={checked["D1"] ? "positive" : "amber"} delay={1} />
        <KPI label="Brake Test" value={checked["D2"] ? "PASS" : "PENDING"} sub="4-wheel lockup" sentiment={checked["D2"] ? "positive" : "amber"} delay={2} />
      </div>
      <GC style={{ padding: 0, overflow: "hidden" }}>
        {DYNAMIC_TESTS.map(item => (
          <div key={item.id}>
            <ChecklistItem item={item} checked={!!checked[item.id]} onToggle={() => toggle(item.id)} showRule={false} />
            {!checked[item.id] && (
              <div style={{ padding: "0 12px 8px 40px", fontSize: 8, color: C.dm, fontFamily: C.dt }}>
                Criteria: {item.criteria}
              </div>
            )}
          </div>
        ))}
      </GC>
    </div>
  );
}

function DocsTab({ checked, toggle }) {
  const total = DOCUMENTS.length;
  const done = DOCUMENTS.filter(i => checked[i.id]).length;
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Submitted" value={`${done}/${total}`} sub={`${(done / total * 100).toFixed(0)}%`} sentiment={done === total ? "positive" : "amber"} delay={0} />
        <KPI label="SES + ESF" value={checked["DOC1"] && checked["DOC2"] ? "SUBMITTED" : "MISSING"} sub="8-week deadline" sentiment={checked["DOC1"] && checked["DOC2"] ? "positive" : "negative"} delay={1} />
      </div>
      <GC style={{ padding: 0, overflow: "hidden" }}>
        {DOCUMENTS.map(item => (
          <div key={item.id} style={{ borderBottom: `1px solid ${C.b1}08` }}>
            <ChecklistItem item={item} checked={!!checked[item.id]} onToggle={() => toggle(item.id)} showRule={false} />
            <div style={{ padding: "0 12px 6px 40px", fontSize: 8, color: C.dm, fontFamily: C.dt }}>
              Deadline: {item.deadline}
            </div>
          </div>
        ))}
      </GC>
    </div>
  );
}

function SafetyTab({ checked, toggle }) {
  const total = SAFETY_SYSTEMS.length;
  const done = SAFETY_SYSTEMS.filter(i => checked[i.id]).length;
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 14 }}>
        <KPI label="Verified" value={`${done}/${total}`} sub="safety systems" sentiment={done === total ? "positive" : "amber"} delay={0} />
        <KPI label="SDC Complete" value={checked["S2"] ? "YES" : "NO"} sub="shutdown circuit" sentiment={checked["S2"] ? "positive" : "negative"} delay={1} />
        <KPI label="EV Ready" value={done >= 10 ? "YES" : "NO"} sub="for scrutineering" sentiment={done >= 10 ? "positive" : "negative"} delay={2} />
      </div>
      <GC style={{ padding: 0, overflow: "hidden" }}>
        {SAFETY_SYSTEMS.map(item => (
          <div key={item.id} style={{ borderBottom: `1px solid ${C.b1}08` }}>
            <ChecklistItem item={{ ...item, item: item.system }} checked={!!checked[item.id]} onToggle={() => toggle(item.id)} showRule={false} />
            <div style={{ padding: "0 12px 6px 40px", fontSize: 8, color: C.dm, fontFamily: C.dt }}>
              Rule: {item.rule} · {item.status}
            </div>
          </div>
        ))}
      </GC>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════════════
export default function ComplianceModule() {
  const [tab, setTab] = useState("tech");
  const [checked, setChecked] = useState({});
  const toggle = id => setChecked(prev => ({ ...prev, [id]: !prev[id] }));

  const allItems = [...TECH_ITEMS, ...DYNAMIC_TESTS, ...DOCUMENTS, ...SAFETY_SYSTEMS];
  const totalAll = allItems.length;
  const doneAll = allItems.filter(i => checked[i.id]).length;
  const critItems = allItems.filter(i => i.critical);
  const critDone = critItems.filter(i => checked[i.id]).length;
  const readiness = critDone === critItems.length ? "GO" : doneAll / totalAll > 0.7 ? "ALMOST" : "NOT READY";

  return (
    <div>
      {/* Status banner */}
      <div style={{
        ...GL, padding: "12px 16px", marginBottom: 14,
        borderLeft: `3px solid ${readiness === "GO" ? C.gn : readiness === "ALMOST" ? C.am : C.red}`,
        display: "flex", alignItems: "center", gap: 16,
      }}>
        <div style={{
          width: 10, height: 10, borderRadius: 5,
          background: readiness === "GO" ? C.gn : readiness === "ALMOST" ? C.am : C.red,
          boxShadow: `0 0 10px ${readiness === "GO" ? C.gn : readiness === "ALMOST" ? C.am : C.red}`,
        }} />
        <span style={{ fontSize: 12, fontWeight: 800, color: readiness === "GO" ? C.gn : readiness === "ALMOST" ? C.am : C.red, fontFamily: C.dt, letterSpacing: 2 }}>
          SCRUTINEERING: {readiness}
        </span>
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 9, color: C.dm, fontFamily: C.dt }}>
          {doneAll}/{totalAll} items · {critDone}/{critItems.length} critical
        </span>
      </div>

      {/* Progress bar */}
      <div style={{ ...GL, padding: "10px 14px", marginBottom: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
          <span style={{ fontSize: 8, fontWeight: 700, color: C.dm, fontFamily: C.dt, letterSpacing: 1.5 }}>OVERALL PROGRESS</span>
          <span style={{ fontSize: 10, fontWeight: 700, color: C.cy, fontFamily: C.dt }}>{(doneAll / totalAll * 100).toFixed(0)}%</span>
        </div>
        <div style={{ height: 8, background: C.b1, borderRadius: 4, overflow: "hidden" }}>
          <div style={{ width: `${doneAll / totalAll * 100}%`, height: "100%", background: `linear-gradient(90deg, ${C.red}, ${C.am} 50%, ${C.gn})`, borderRadius: 4, transition: "width 0.3s" }} />
        </div>
      </div>

      <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
        {TABS.map(t => <Pill key={t.key} active={tab === t.key} label={t.label} onClick={() => setTab(t.key)} color={C.gn} />)}
      </div>

      {tab === "tech" && <TechTab checked={checked} toggle={toggle} />}
      {tab === "dynamic" && <DynamicTab checked={checked} toggle={toggle} />}
      {tab === "docs" && <DocsTab checked={checked} toggle={toggle} />}
      {tab === "safety" && <SafetyTab checked={checked} toggle={toggle} />}
    </div>
  );
}