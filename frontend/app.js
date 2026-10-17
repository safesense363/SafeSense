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
let alertAcknowledged  = false;
let subjectPos         = { x: 0.4, y: 0.55 };

const $ = id => document.getElementById(id);
const DOM = {
  statusDot:        $("system-status-dot"),
  statusText:       $("system-status-text"),
  headerDevice:     $("header-device"),
  headerModel:      $("header-model"),
  alertBanner:      $("alert-banner"),
  alertTitle:       $("alert-title"),
  alertReason:      $("alert-reason"),
  btnAckAlert:      $("btn-ack-alert"),
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
  kpiCooldown:      $("kpi-cooldown"),
  kpiAlertCount:    $("kpi-alert-count"),
  kpiZoneStatus:    $("kpi-zone-status"),
  kpiZoneDesc:      $("kpi-zone-desc"),
  zoneCanvas:       $("zone-canvas"),
  probList:         $("prob-list"),
  eventTbody:       $("event-tbody"),
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
  DOM.statusText.textContent = online ? "LIVE" : "OFFLINE";
}

/* ─── Zone Canvas ─── */
const ZONES = [
  { label:"General Lab Area",     x:0,   y:0,    w:1,    h:1,    fill:"rgba(100,220,130,0.07)", stroke:"rgba(100,220,130,0.3)" },
  { label:"Restricted Machinery", x:0.6, y:0,    w:0.4,  h:0.5,  fill:"rgba(255,80,80,0.12)",  stroke:"rgba(255,80,80,0.5)"   },
  { label:"Chemical Storage",     x:0,   y:0.65, w:0.35, h:0.35, fill:"rgba(255,180,0,0.1)",   stroke:"rgba(255,180,0,0.4)"   },
];

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
  ZONES.forEach(z => {
    const zx=z.x*W, zy=z.y*H, zw=z.w*W, zh=z.h*H;
    ctx.fillStyle = z.fill; ctx.fillRect(zx,zy,zw,zh);
    ctx.strokeStyle = z.stroke; ctx.lineWidth=1.5; ctx.strokeRect(zx,zy,zw,zh);
    ctx.fillStyle = z.stroke; ctx.font="10px Inter,sans-serif"; ctx.fillText(z.label,zx+6,zy+14);
  });
  const sx = subjectPos.x * W, sy = subjectPos.y * H;
  const g = ctx.createRadialGradient(sx,sy,0,sx,sy,18);
  g.addColorStop(0,"rgba(80,200,255,0.5)"); g.addColorStop(1,"rgba(80,200,255,0)");
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(sx,sy,18,0,Math.PI*2); ctx.fill();
  ctx.fillStyle = "#50c8ff"; ctx.beginPath(); ctx.arc(sx,sy,6,0,Math.PI*2); ctx.fill();
  ctx.strokeStyle="rgba(80,200,255,0.4)"; ctx.lineWidth=1; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(sx,0); ctx.lineTo(sx,H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,sy); ctx.lineTo(W,sy); ctx.stroke();
  ctx.setLineDash([]);
}

/* ─── Health ─── */
async function pollHealth() {
  try {
    const res  = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    setStatus(true);
    DOM.headerDevice.textContent = (data.device || "cpu").toUpperCase();
    DOM.headerModel.textContent  = data.model_path ? data.model_path.replace(".pt"," (GRU)") : "Not Loaded";
    if (!data.model_loaded) { DOM.statusText.textContent="DEGRADED"; DOM.statusDot.className="pulse-dot dot-warning"; }
  } catch { setStatus(false); }
}

/* ─── Probability bars ─── */
function renderProbList(probs) {
  if (!probs || !Object.keys(probs).length) {
    DOM.probList.innerHTML = "<p class='prob-empty'>No probability data.</p>"; return;
  }
  DOM.probList.innerHTML = Object.entries(probs)
    .sort((a,b)=>b[1]-a[1])
    .map(([cls,p]) => {
      const pct = (p*100).toFixed(1);
      return `<div class="prob-row">
        <span class="prob-label">${cls}</span>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${Math.max(pct,0.5)}%"></div></div>
        <span class="prob-val">${pct}%</span></div>`;
    }).join("");
}

/* ─── Table row ─── */
function buildRow(ev) {
  const meta    = RISK_META[ev.risk_level] || RISK_META.NORMAL;
  const reasons = (ev.reasons||[]).join(" · ") || "—";
  const alert   = ev.alert_triggered ? '<span class="alert-pip">ALERT</span>' : "";
  return `<tr class="event-row" data-risk="${ev.risk_level}">
    <td class="td-time">${fmtTime(ev.timestamp)}</td>
    <td class="td-activity">${ev.activity} ${alert}</td>
    <td><span class="conf-chip">${fmtPct(ev.confidence)}</span></td>
    <td><span class="risk-badge risk-${ev.risk_level.toLowerCase()}">${meta.label}</span></td>
    <td>${ev.risk_score.toFixed(3)}</td>
    <td class="td-reason">${reasons}</td></tr>`;
}

/* ─── Zone eval ─── */
function evalZone(pos) {
  if (pos.x>0.6&&pos.y<0.5) return { status:"BREACH",  desc:"Subject in Restricted Machinery Zone!" };
  if (pos.x<0.35&&pos.y>0.65) return { status:"WARNING", desc:"Subject near Chemical Storage Zone." };
  return { status:"SECURE", desc:"All zones nominal." };
}

/* ─── Apply latest event to KPIs ─── */
function applyEvent(ev) {
  DOM.kpiActivity.textContent   = ev.activity.toUpperCase();
  DOM.activityBadge.textContent = `CONF: ${fmtPct(ev.confidence)}`;
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
    DOM.kpiIncident.textContent = "INCIDENT";
    DOM.kpiCooldown.textContent = "ACTIVE";
    DOM.kpiAlertCount.textContent = (parseInt(DOM.kpiAlertCount.textContent,10)||0) + 1;
  } else if (ev.risk_level==="NORMAL") {
    DOM.kpiIncident.textContent = "ALL CLEAR";
    DOM.kpiCooldown.textContent = "INACTIVE";
  }

  const h = [...ev.activity].reduce((a,c)=>a+c.charCodeAt(0),0);
  subjectPos = {
    x: 0.1 + (((h*7+Math.floor(ev.timestamp))%100)/100)*0.75,
    y: 0.1 + ((Math.floor(ev.risk_score*100)%80)/100)*0.75,
  };
  const zone = evalZone(subjectPos);
  DOM.kpiZoneStatus.textContent = zone.status;
  DOM.kpiZoneDesc.textContent   = zone.desc;
  renderProbList(ev.probabilities);

  if ((ev.risk_level==="HIGH"||ev.risk_level==="CRITICAL") && ev.alert_triggered) {
    if (ev.timestamp!==lastAlertTimestamp) { lastAlertTimestamp=ev.timestamp; alertAcknowledged=false; }
    if (!alertAcknowledged) {
      DOM.alertTitle.textContent  = `${meta.label} INCIDENT DETECTED`;
      DOM.alertReason.textContent = (ev.reasons||[]).join(" · ") || `${ev.activity} at ${fmtPct(ev.confidence)}`;
      DOM.alertBanner.classList.remove("hidden");
    }
  } else if (!alertAcknowledged) {
    DOM.alertBanner.classList.add("hidden");
  }
  drawCanvas();
}

/* ─── Events polling ─── */
async function pollEvents() {
  try {
    let url = `${API_BASE}/api/events?limit=${MAX_TABLE_ROWS}`;
    if (currentFilter==="alerts")   url+="&only_alerts=true";
    if (currentFilter==="critical") url+="&min_risk=HIGH";
    const res  = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error("HTTP "+res.status);
    const data = await res.json();
    setStatus(true);
    const events = data.events||[];
    if (!events.length) {
      DOM.eventTbody.innerHTML = `<tr><td colspan="6" class="empty-state">No events match current filter.</td></tr>`;
      return;
    }
    DOM.eventTbody.innerHTML = events.map(buildRow).join("");
    applyEvent(events[0]);
  } catch { setStatus(false); }
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
    await fetch(`${API_BASE}/api/events`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload), signal:AbortSignal.timeout(5000),
    });
    await pollEvents();
  } catch {
    applyEvent(payload);
    DOM.eventTbody.innerHTML = buildRow(payload) + DOM.eventTbody.innerHTML;
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
  pollEvents();
  startAutoRefresh();
  healthTimer = setInterval(pollHealth, POLL_HEALTH_MS);

  DOM.btnToggleRefresh.addEventListener("click", () => {
    autoRefresh = !autoRefresh;
    autoRefresh ? startAutoRefresh() : (stopAutoRefresh(), DOM.btnToggleRefresh.textContent="Auto-Refresh: OFF", DOM.btnToggleRefresh.classList.add("btn-muted"));
  });
  DOM.btnAckAlert.addEventListener("click", () => { alertAcknowledged=true; DOM.alertBanner.classList.add("hidden"); });
  DOM.filterAll.addEventListener("click",      () => setFilter("all"));
  DOM.filterAlerts.addEventListener("click",   () => setFilter("alerts"));
  DOM.filterCritical.addEventListener("click", () => setFilter("critical"));
  DOM.btnSimWalk.addEventListener("click", () => simulate("walk"));
  DOM.btnSimRun.addEventListener("click",  () => simulate("run"));
  DOM.btnSimZone.addEventListener("click", () => simulate("zone"));
  DOM.btnSimFall.addEventListener("click", () => simulate("fall"));
}

document.addEventListener("DOMContentLoaded", init);
