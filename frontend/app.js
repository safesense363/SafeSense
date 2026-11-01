/**
 * SafeSense AI - Dashboard Application Script
 * Polls /api/health and /api/events, drives KPI cards, zone canvas, event table, and sim buttons.
 */
"use strict";

const API_BASE       = "http://localhost:8000";
const POLL_EVENTS_MS = 3000;
const POLL_HEALTH_MS = 30000;
const MAX_TABLE_ROWS = 50;

let autoRefresh        = true;
let refreshTimer       = null;
let healthTimer        = null;
let currentFilter      = "all";
let lastAlertTimestamp = null;
let alertBannerDismissed = false;
let inferenceFresh = false;

const $ = id => document.getElementById(id);
const DOM = {
  statusDot:        $("system-status-dot"),
  statusText:       $("system-status-text"),
  healthApi:        $("health-api"),
  healthCamera:     $("health-camera"),
  healthModel:      $("health-model"),
  healthDatabase:   $("health-database"),
  healthInference:  $("health-inference"),
  healthNote:       $("health-note"),
  headerDevice:     $("header-device"),
  headerModel:      $("header-model"),
  alertBanner:      $("alert-banner"),
  alertTitle:       $("alert-title"),
  alertReason:      $("alert-reason"),
  btnDismissAlert:  $("btn-dismiss-alert"),
  kpiActivity:      $("kpi-activity"),
  activityBadge:    $("activity-id-badge"),
  activityBar:      $("activity-confidence-bar"),
  kpiConfidence:    $("kpi-confidence"),
  kpiLatency:       $("kpi-latency"),
  kpiRiskLevel:     $("kpi-risk-level"),
  riskScoreCircle:  $("risk-score-circle"),
  kpiRiskScore:     $("kpi-risk-score"),
  kpiRiskReason:    $("kpi-risk-reason"),
  kpiIncident:      $("kpi-incident-status"),
  kpiAlertCount:    $("kpi-alert-count"),
  kpiZoneStatus:    $("kpi-zone-status"),
  kpiZoneDesc:      $("kpi-zone-desc"),
  zoneCanvas:       $("zone-canvas"),
  probList:         $("prob-list"),
  eventTbody:       $("event-tbody"),
  eventDetailsPlaceholder: $("event-details-placeholder"),
  eventDetailsList: $("event-details-list"),
  btnToggleRefresh: $("btn-toggle-refresh"),
  filterAll:        $("filter-all"),
  filterAlerts:     $("filter-alerts"),
  filterCritical:   $("filter-critical"),
  btnSimWalk:       $("btn-sim-walk"),
  btnSimRun:        $("btn-sim-run"),
  btnSimZone:       $("btn-sim-zone"),
  btnSimFall:       $("btn-sim-fall"),
};

const RISK_META = {
  NORMAL:    { cls: "status-normal",    icon: "OK",       label: "NORMAL"    },
  UNCERTAIN: { cls: "status-uncertain", icon: "?",        label: "UNCERTAIN" },
  UNKNOWN:   { cls: "status-uncertain", icon: "?",        label: "UNKNOWN"   },
  WARNING:   { cls: "status-warning",   icon: "WARN",     label: "WARNING"   },
  HIGH:      { cls: "status-high",      icon: "HIGH",     label: "HIGH"      },
  CRITICAL:  { cls: "status-critical",  icon: "CRITICAL", label: "CRITICAL"  },
};
const RISK_CLASSES = ["status-normal","status-uncertain","status-warning","status-high","status-critical"];

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-IN", { hour12:false, hour:"2-digit", minute:"2-digit", second:"2-digit" });
}
function fmtPct(v) { return (v * 100).toFixed(1) + "%"; }
function clearRiskClasses(el) { RISK_CLASSES.forEach(c => el.classList.remove(c)); }

function setStatus(online) {
  DOM.statusDot.className    = "pulse-dot " + (online ? "dot-online" : "dot-offline");
  DOM.statusText.textContent = online ? "API ONLINE" : "API OFFLINE";
}

function setHealthValue(element, text, state) {
  element.textContent = text;
  element.dataset.state = state;
}

/* ─── Zone Canvas ─── */
let zones = [];

function drawCanvas() {
  const canvas = DOM.zoneCanvas;
  const ctx    = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(10,20,40,0.95)";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "rgba(100,140,200,0.08)";
  ctx.lineWidth = 1;
  for (let gx = 0; gx <= W; gx += 40) { ctx.beginPath(); ctx.moveTo(gx,0); ctx.lineTo(gx,H); ctx.stroke(); }
  for (let gy = 0; gy <= H; gy += 40) { ctx.beginPath(); ctx.moveTo(0,gy); ctx.lineTo(W,gy); ctx.stroke(); }
  zones.forEach(z => {
    const zx=z.x*W, zy=z.y*H, zw=z.w*W, zh=z.h*H;
    ctx.fillStyle = z.fill; ctx.fillRect(zx,zy,zw,zh);
    ctx.strokeStyle = z.stroke; ctx.lineWidth=1.5; ctx.strokeRect(zx,zy,zw,zh);
    ctx.fillStyle = z.stroke; ctx.font="10px Inter,sans-serif"; ctx.fillText(z.label,zx+6,zy+14);
  });
}

async function pollRiskConfig() {
  try {
    const response = await fetch(
      `${API_BASE}/api/config/risk`,
      { signal: AbortSignal.timeout(5000) },
    );
    if (!response.ok) throw new Error(`Risk config HTTP ${response.status}`);
    const config = await response.json();
    zones = (config.zones || []).flatMap(zone => {
      const box = zone.bbox;
      if (!Array.isArray(box) || box.length !== 4
        || !box.every(value => Number.isFinite(value) && value >= 0 && value <= 1)
        || box[2] < box[0] || box[3] < box[1]) {
        return [];
      }
      return [{
        label: zone.description || zone.name || "Configured zone",
        x: box[0],
        y: box[1],
        w: box[2] - box[0],
        h: box[3] - box[1],
        fill: "rgba(255,80,80,0.12)",
        stroke: "rgba(255,80,80,0.7)",
      }];
    });
    DOM.kpiZoneStatus.textContent = `${zones.length} CONFIGURED`;
    DOM.kpiZoneDesc.textContent = zones.length
      ? "Bounds are loaded from risk rules; live subject position is not available."
      : "No valid safety zones are configured.";
    drawCanvas();
  } catch (error) {
    zones = [];
    DOM.kpiZoneStatus.textContent = "CONFIGURATION UNAVAILABLE";
    DOM.kpiZoneDesc.textContent = error.message;
    drawCanvas();
  }
}

/* ─── Health ─── */
async function pollHealth() {
  try {
    const healthResponse = await fetch(
      `${API_BASE}/api/health`,
      { signal: AbortSignal.timeout(5000) },
    );
    if (!healthResponse.ok) throw new Error(`Health HTTP ${healthResponse.status}`);
    const data = await healthResponse.json();
    setStatus(true);
    DOM.headerDevice.textContent = (data.device || "cpu").toUpperCase();
    setHealthValue(DOM.healthApi, "Available", "ok");
    setHealthValue(
      DOM.healthCamera,
      data.camera_available ? "Available" : "Unavailable",
      data.camera_available ? "ok" : "warning",
    );
    setHealthValue(
      DOM.healthModel,
      data.model_loaded ? "Available" : "Unavailable",
      data.model_loaded ? "ok" : "warning",
    );
    setHealthValue(
      DOM.healthDatabase,
      data.database_available ? "Available" : "Unavailable",
      data.database_available ? "ok" : "warning",
    );
    const degraded = data.status !== "healthy";
    DOM.healthNote.textContent = degraded
      ? "One or more technical checks are degraded or unavailable. This is separate from safety incidents."
      : "Technical checks are available. Inference freshness is based on recent events; it is not a camera-process heartbeat.";
  } catch (error) {
    setStatus(false);
    setHealthValue(DOM.healthApi, "Unavailable", "warning");
    setHealthValue(DOM.healthCamera, "Unknown", "unknown");
    setHealthValue(DOM.healthModel, "Unknown", "unknown");
    setHealthValue(DOM.healthDatabase, "Unknown", "unknown");
    setHealthValue(DOM.healthInference, "Unknown", "unknown");
    DOM.healthNote.textContent = `Health check failed: ${error.message}`;
  }
}

async function pollInferenceStatus() {
  try {
    const response = await fetch(
      `${API_BASE}/api/status`,
      { signal: AbortSignal.timeout(5000) },
    );
    if (!response.ok) throw new Error(`Status HTTP ${response.status}`);
    const pipeline = await response.json();
    inferenceFresh = Boolean(pipeline.is_running);
    setHealthValue(
      DOM.healthInference,
      inferenceFresh ? "Recent event received" : "No recent inference event",
      inferenceFresh ? "ok" : "unknown",
    );
  } catch {
    inferenceFresh = false;
    setHealthValue(DOM.healthInference, "Status unavailable", "unknown");
  }
}

/* ─── Probability bars ─── */
function renderProbList(probs) {
  if (!probs || !Object.keys(probs).length) {
    DOM.probList.innerHTML = "<p class='prob-empty'>No probability data.</p>"; return;
  }
  DOM.probList.replaceChildren();
  Object.entries(probs).sort((a,b)=>b[1]-a[1]).forEach(([cls,p]) => {
    const pct = (p*100).toFixed(1);
    const row = document.createElement("div");
    row.className = "prob-row";
    const label = document.createElement("span");
    label.className = "prob-label";
    label.textContent = cls;
    const bar = document.createElement("div");
    bar.className = "prob-bar-bg";
    const fill = document.createElement("div");
    fill.className = "prob-bar-fill";
    fill.style.width = `${Math.max(Number(pct), 0.5)}%`;
    bar.append(fill);
    const value = document.createElement("span");
    value.className = "prob-val";
    value.textContent = `${pct}%`;
    row.append(label, bar, value);
    DOM.probList.append(row);
  });
}

/* ─── Table row ─── */
function buildRow(ev, index) {
  const meta    = RISK_META[ev.risk_level] || RISK_META.NORMAL;
  const reasons = (ev.reasons||[]).join(" · ") || "—";
  const row = document.createElement("tr");
  row.className = "event-row";
  row.dataset.eventIndex = String(index);
  row.tabIndex = 0;
  row.addEventListener("click", () => showEventDetails(ev));
  row.addEventListener("keydown", event => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      showEventDetails(ev);
    }
  });
  const values = [
    fmtTime(ev.timestamp),
    `${ev.activity}${ev.alert_triggered ? " — ALERT" : ""}`,
    fmtPct(ev.confidence),
    meta.label,
    ev.risk_score.toFixed(3),
    reasons,
  ];
  values.forEach((value, cellIndex) => {
    const cell = document.createElement("td");
    cell.textContent = value;
    if (cellIndex === 5) cell.className = "td-reason";
    row.append(cell);
  });
  return row;
}

function showEventDetails(event) {
  DOM.eventDetailsPlaceholder.hidden = true;
  DOM.eventDetailsList.replaceChildren();
  const fields = [
    ["Recorded at", fmtTime(event.timestamp)],
    ["Activity", event.activity],
    ["Model", event.model || "Not reported"],
    ["Risk level", event.risk_level],
    ["Risk score", Number(event.risk_score).toFixed(3)],
    ["Confidence", fmtPct(event.confidence)],
    ["Incident type", event.incident_type || "Not reported"],
    ["Rationale", (event.reasons || []).join(" · ") || "No rationale provided"],
  ];
  fields.forEach(([label, value]) => {
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = value;
    DOM.eventDetailsList.append(term, description);
  });
}

/* ─── Apply latest event to KPIs ─── */
function applyEvent(ev) {
  DOM.headerModel.textContent   = ev.model || "Demo / external";
  DOM.kpiActivity.textContent   = ev.activity.toUpperCase();
  DOM.activityBadge.textContent = `SCORE: ${fmtPct(ev.confidence)}`;
  DOM.activityBar.style.width   = fmtPct(ev.confidence);
  DOM.kpiConfidence.textContent = fmtPct(ev.confidence);
  DOM.kpiLatency.textContent    = `Latency: ${ev.latency_ms.toFixed(1)} ms`;

  const meta = RISK_META[ev.risk_level] || RISK_META.NORMAL;
  clearRiskClasses(DOM.kpiRiskLevel);
  DOM.kpiRiskLevel.classList.add(meta.cls);
  DOM.kpiRiskLevel.textContent  = meta.label;
  DOM.kpiRiskScore.textContent  = ev.risk_score.toFixed(3);
  DOM.kpiRiskReason.textContent = (ev.reasons||[]).join(" · ") || "No active risk triggers.";
  clearRiskClasses(DOM.riskScoreCircle);
  DOM.riskScoreCircle.classList.add(meta.cls);

  if (ev.alert_triggered) {
    DOM.kpiIncident.textContent = "ALERT EVENT";
    DOM.kpiAlertCount.textContent = "YES";
  } else {
    DOM.kpiIncident.textContent = "NO ALERT";
    DOM.kpiAlertCount.textContent = "NO";
  }

  renderProbList(ev.probabilities);

  if ((ev.risk_level==="HIGH"||ev.risk_level==="CRITICAL")
    && ev.alert_triggered && inferenceFresh) {
    if (ev.timestamp!==lastAlertTimestamp) { lastAlertTimestamp=ev.timestamp; alertBannerDismissed=false; }
    if (!alertBannerDismissed) {
      DOM.alertTitle.textContent  = `${meta.label} INCIDENT DETECTED`;
      DOM.alertReason.textContent = (ev.reasons||[]).join(" · ") || `${ev.activity} at ${fmtPct(ev.confidence)}`;
      DOM.alertBanner.classList.remove("hidden");
    }
  } else if (!alertBannerDismissed) {
    DOM.alertBanner.classList.add("hidden");
  }
  drawCanvas();
}

/* ─── Events polling ─── */
async function pollEvents() {
  await pollInferenceStatus();
  try {
    let url = `${API_BASE}/api/events?limit=${MAX_TABLE_ROWS}`;
    if (currentFilter==="alerts")   url+="&only_alerts=true";
    if (currentFilter==="critical") url+="&min_risk=HIGH";
    const res  = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error("HTTP "+res.status);
    const data = await res.json();
    const events = data.events||[];
    if (!events.length) {
      DOM.eventTbody.innerHTML = `<tr><td colspan="6" class="empty-state">No events match current filter.</td></tr>`;
      DOM.eventDetailsPlaceholder.hidden = false;
      DOM.eventDetailsList.replaceChildren();
      return;
    }
    DOM.eventTbody.replaceChildren(...events.map(buildRow));
    applyEvent(events[0]);
  } catch (error) {
    DOM.eventTbody.textContent = "";
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "empty-state";
    cell.textContent = `Unable to load event history: ${error.message}`;
    row.append(cell);
    DOM.eventTbody.append(row);
  }
}

/* ─── Simulation payloads ─── */
const SIM_PAYLOADS = {
  walk: { activity:"walking",  confidence:0.82, risk_level:"NORMAL",   risk_score:0.10, reasons:[],
    alert_triggered:false, incident_type:null, latency_ms:14.2,
    probabilities:{walking:0.82,standing:0.09,running:0.05,sitting:0.02,falling:0.01,lying_down:0.01} },
  run:  { activity:"running",  confidence:0.74, risk_level:"WARNING",  risk_score:0.42,
    reasons:["Running detected in lab","Velocity exceeds safe threshold"],
    alert_triggered:false, incident_type:null, latency_ms:16.1,
    probabilities:{running:0.74,walking:0.16,standing:0.06,falling:0.03,sitting:0.01,lying_down:0.00} },
  zone: { activity:"walking",  confidence:0.69, risk_level:"HIGH",     risk_score:0.78,
    reasons:["Subject entered Restricted Machinery Zone","Prolonged proximity to hazard area"],
    alert_triggered:true, incident_type:"zone_breach", latency_ms:18.5,
    probabilities:{walking:0.69,standing:0.15,running:0.10,sitting:0.04,falling:0.02,lying_down:0.00} },
  fall: { activity:"falling",  confidence:0.91, risk_level:"CRITICAL", risk_score:0.97,
    reasons:["Fall motion pattern detected","Prolonged floor contact","Subject unresponsive"],
    alert_triggered:true, incident_type:"fall", latency_ms:12.3,
    probabilities:{falling:0.91,lying_down:0.06,running:0.02,walking:0.01,standing:0.00,sitting:0.00} },
};

async function simulate(type) {
  const payload = { ...SIM_PAYLOADS[type], timestamp: Date.now()/1000, sequence_shape:[30,51] };
  try {
    const response = await fetch(`${API_BASE}/api/events`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload), signal:AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error(`Event submission failed (HTTP ${response.status})`);
    await pollEvents();
  } catch (error) {
    DOM.healthNote.textContent = `Demo event was not saved: ${error.message}`;
  }
}

/* ─── Auto-refresh ─── */
function startAutoRefresh() {
  stopAutoRefresh();
  refreshTimer = setInterval(pollEvents, POLL_EVENTS_MS);
  DOM.btnToggleRefresh.textContent = "Auto-Refresh: ON";
  DOM.btnToggleRefresh.classList.remove("btn-muted");
}
function stopAutoRefresh() { if (refreshTimer) clearInterval(refreshTimer); refreshTimer=null; }

/* ─── Filter ─── */
function setFilter(f) {
  currentFilter = f;
  [DOM.filterAll,DOM.filterAlerts,DOM.filterCritical].forEach(b=>b.classList.remove("active"));
  ({all:DOM.filterAll,alerts:DOM.filterAlerts,critical:DOM.filterCritical})[f]?.classList.add("active");
  pollEvents();
}

/* ─── Boot ─── */
function init() {
  drawCanvas();
  pollHealth();
  pollRiskConfig();
  pollEvents();
  startAutoRefresh();
  healthTimer = setInterval(pollHealth, POLL_HEALTH_MS);
  setInterval(pollRiskConfig, POLL_HEALTH_MS);

  DOM.btnToggleRefresh.addEventListener("click", () => {
    autoRefresh = !autoRefresh;
    autoRefresh ? startAutoRefresh() : (stopAutoRefresh(), DOM.btnToggleRefresh.textContent="Auto-Refresh: OFF", DOM.btnToggleRefresh.classList.add("btn-muted"));
  });
  DOM.btnDismissAlert.addEventListener("click", () => {
    alertBannerDismissed = true;
    DOM.alertBanner.classList.add("hidden");
  });
  DOM.filterAll.addEventListener("click",      () => setFilter("all"));
  DOM.filterAlerts.addEventListener("click",   () => setFilter("alerts"));
  DOM.filterCritical.addEventListener("click", () => setFilter("critical"));
  DOM.btnSimWalk.addEventListener("click", () => simulate("walk"));
  DOM.btnSimRun.addEventListener("click",  () => simulate("run"));
  DOM.btnSimZone.addEventListener("click", () => simulate("zone"));
  DOM.btnSimFall.addEventListener("click", () => simulate("fall"));
}

document.addEventListener("DOMContentLoaded", init);
