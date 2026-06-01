/* =============================================================================
   Physio-Vision  ·  app.js
   JS is a dumb terminal.  All API calls, session logic, and state mutation
   happen in Python (Bridge).  JS only renders what Python tells it.
   ============================================================================= */

"use strict";

// ---------------------------------------------------------------------------
// QWebChannel bootstrap
// ---------------------------------------------------------------------------
let backend = null;   // set after QWebChannel is ready

new QWebChannel(qt.webChannelTransport, (channel) => {
  backend = channel.objects.backend;

  // ── Subscribe to Python → JS signals ────────────────────────────────────
  backend.stats_changed.connect(onStatsChanged);
  backend.status_changed.connect(onStatusChanged);
  backend.session_finished.connect(onSessionFinished);
  backend.history_loaded.connect(onHistoryLoaded);

  // ── Seed the UI with real backend state ─────────────────────────────────
  backend.get_initial_state((stateJson) => {
    const s = JSON.parse(stateJson);
    initUi(s);
  });

  // ── Load history for Records page ───────────────────────────────────────
  backend.fetch_history();
});

// ---------------------------------------------------------------------------
// App state (UI-only; ground truth lives in Python)
// ---------------------------------------------------------------------------
const ui = {
  currentPage: "hub",
  currentExercise: null,    // key: "squat" | "sts" | "pushup" | "curl"
  sessionRunning: false,
  lastReport: null,         // holds report dict while pain dialog is open
};

// Tracks the live Chart.js instance so we can destroy it cleanly before
// re-rendering (prevents canvas memory leaks and stale-data ghosting).
let progressChartInstance = null;

// Cache of the last received records array so filter pills can re-render
// without needing another round-trip to Python.
let _cachedRecords = [];

// ---------------------------------------------------------------------------
// Exercise catalogue
// ---------------------------------------------------------------------------
const EXERCISES = [
  { key: "squat",  title: "Deep Squat",    desc: "Knee & hip mobility analysis",      icon: "🦵" },
  { key: "sts",    title: "Sit to Stand",  desc: "Geriatric fall-risk assessment",     icon: "🪑" },
  { key: "pushup", title: "Push-up",       desc: "Upper body & core stabilisation",    icon: "💪" },
  { key: "curl",   title: "Bicep Curl",    desc: "Elbow ROM & cheat classification",   icon: "🏋️" },
];

const EXERCISE_LABELS = Object.fromEntries(EXERCISES.map(e => [e.key, e.title]));

const PAIN_LABELS = [
  "No Pain", "Very Mild", "Mild", "Moderate Mild",
  "Moderate", "Moderate Severe", "Severe", "Very Severe",
  "Intense", "Extremely Intense", "Unbearable"
];

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function navigate(pageId) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => {
    n.classList.toggle("active", n.dataset.page === pageId);
  });
  const el = document.getElementById(`page-${pageId}`);
  if (el) el.classList.add("active");
  ui.currentPage = pageId;
}

function launchExercise(key) {
  ui.currentExercise = key;
  // Update analysis page header
  document.getElementById("analysis-exercise-title").textContent =
    EXERCISE_LABELS[key] || key;
  document.getElementById("analysis-exercise-icon").textContent =
    EXERCISES.find(e => e.key === key)?.icon || "🏃";

  // Reset metrics
  resetAnalysisMetrics();
  navigate("analysis");
}

// ---------------------------------------------------------------------------
// Initialise UI from Python state
// ---------------------------------------------------------------------------
function initUi(s) {
  // Set username avatar / greeting
  const initial = (s.username || "U")[0].toUpperCase();
  document.querySelectorAll(".user-avatar").forEach(el => el.textContent = initial);
  document.querySelectorAll(".user-name").forEach(el => el.textContent = s.username || "User");

  // Settings sliders & toggles
  setSlider("height-slider", "height-val", s.USER_HEIGHT_CM, (v) => `${v} cm`);
  setSlider("weight-slider", "weight-val", s.USER_WEIGHT_KG, (v) => `${v} kg`);
  setToggle("voice-toggle", s.VOICE_ON);
  setToggle("ar-toggle",    s.AR_MODE);
}

function setSlider(sliderId, valId, value, fmt) {
  const slider = document.getElementById(sliderId);
  const valEl  = document.getElementById(valId);
  if (!slider || !valEl) return;
  slider.value = value;
  valEl.textContent = fmt(value);
}

function setToggle(toggleId, value) {
  const el = document.getElementById(toggleId);
  if (el) el.checked = Boolean(value);
}

// ---------------------------------------------------------------------------
// Settings — two-way bridge
// ---------------------------------------------------------------------------
function onSettingSlider(key, sliderId, valId, fmt) {
  const slider = document.getElementById(sliderId);
  const valEl  = document.getElementById(valId);
  if (!slider || !valEl) return;
  const v = parseFloat(slider.value);
  valEl.textContent = fmt(v);
  if (backend) backend.update_setting(key, JSON.stringify(v));
}

function onSettingToggle(key, toggleId) {
  const el = document.getElementById(toggleId);
  if (!el || !backend) return;
  backend.update_setting(key, JSON.stringify(el.checked));
}

// ---------------------------------------------------------------------------
// Session control
// ---------------------------------------------------------------------------
function startSession() {
  if (!ui.currentExercise || !backend) return;
  ui.sessionRunning = true;
  updateSessionButton();
  updateStatusPill("CONNECTING…", "warning");
  setVideoFeed(true);
  backend.start_session(ui.currentExercise);
}

function stopSession() {
  if (!backend) return;
  backend.stop_session();
  ui.sessionRunning = false;
  updateSessionButton();
  updateStatusPill("OFFLINE", "");
  setVideoFeed(false);
}

function toggleSession() {
  if (ui.sessionRunning) stopSession();
  else startSession();
}

function updateSessionButton() {
  const btn = document.getElementById("session-btn");
  if (!btn) return;
  if (ui.sessionRunning) {
    btn.textContent = "⬛  Stop Session";
    btn.className = "btn btn-stop btn-full";
  } else {
    btn.textContent = "▶  Start Session";
    btn.className = "btn btn-primary btn-full";
  }
}

function setVideoFeed(active) {
  const img = document.getElementById("video_feed");
  const placeholder = document.getElementById("video-placeholder");
  if (!img || !placeholder) return;
  if (active) {
    img.src = `http://127.0.0.1:5050/video_feed?t=${Date.now()}`;
    img.style.display = "block";
    placeholder.style.display = "none";
  } else {
    img.src = "";
    img.style.display = "none";
    placeholder.style.display = "flex";
  }
}

// ---------------------------------------------------------------------------
// Python → JS signal handlers
// ---------------------------------------------------------------------------
function onStatsChanged(jsonStr) {
  const data = JSON.parse(jsonStr);

  // Cloud sync notification
  if (data.__cloud_sync) {
    toast(data.__cloud_sync === "ok" ? "Session saved to cloud ✓" : "Cloud sync failed", data.__cloud_sync === "ok" ? "success" : "error");
    return;
  }
  if (data.__session_error) {
    toast("Camera error — session aborted.", "error");
    ui.sessionRunning = false;
    updateSessionButton();
    return;
  }

  // Live metrics
  if (data.reps    !== undefined) setMetric("metric-reps",     data.reps);
  if (data.score   !== undefined) {
    setMetric("metric-score", data.score);
    updateScoreRing(data.score);
  }
  if (data.feedback !== undefined) updateFeedback(data.feedback);
  if (data.knee_angle !== undefined) setMetric("metric-knee", Math.round(data.knee_angle) + "°");
  if (data.trunk_lean !== undefined) setMetric("metric-trunk", Math.round(data.trunk_lean) + "°");
  if (data.elbow_angle !== undefined) setMetric("metric-elbow", Math.round(data.elbow_angle) + "°");
}

function onStatusChanged(text, color) {
  const pill = document.getElementById("status-pill");
  if (!pill) return;
  pill.className = "status-pill";
  pill.querySelector(".status-text").textContent = text;

  const running = ["RECORDING", "ANALYZING", "DETECTING", "CONNECTING"].some(k => text.includes(k));
  const warning = ["RESETTING", "TIMEOUT", "STAND", "HOLD", "MOVE", "GET IN"].some(k => text.includes(k));
  if (running) pill.classList.add("online");
  else if (warning) pill.classList.add("warning");
}

function onSessionFinished(jsonStr) {
  const report = JSON.parse(jsonStr);
  ui.lastReport = report;
  ui.sessionRunning = false;
  updateSessionButton();
  updateStatusPill("OFFLINE", "");
  setVideoFeed(false);
  openPainDialog(report);
}

function onHistoryLoaded(jsonStr) {
  const records = JSON.parse(jsonStr);
  _cachedRecords = records;
  populateRecords(records);
  updateKpis(records);
  renderAnalyticsChart(records, "all");
  renderAnalyticsBreakdown(records);
  renderAnalyticsKpis(records);
}

// ---------------------------------------------------------------------------
// Analytics — KPI summary row
// ---------------------------------------------------------------------------
function renderAnalyticsKpis(records) {
  if (!records.length) return;

  // Chronological order for trend calculation
  const chron  = [...records].reverse();
  const scores = chron.map(r => r.score ?? r.avg_score ?? 0);

  // Best score
  setMetric("an-best", Math.max(...scores));

  // Latest score
  setMetric("an-latest", scores[scores.length - 1] ?? "—");

  // Trend: avg of last 5 vs avg of previous 5
  const last5 = scores.slice(-5);
  const prev5 = scores.slice(-10, -5);
  if (last5.length && prev5.length) {
    const avgLast = last5.reduce((a, b) => a + b, 0) / last5.length;
    const avgPrev = prev5.reduce((a, b) => a + b, 0) / prev5.length;
    const delta   = Math.round(avgLast - avgPrev);
    const el = document.getElementById("an-trend");
    if (el) {
      el.textContent = (delta >= 0 ? "▲ +" : "▼ ") + delta;
      el.style.color = delta >= 0 ? "var(--accent-green)" : "var(--accent-red)";
    }
  } else {
    setMetric("an-trend", "—");
  }

  // Consistency: sessions recorded this calendar month
  const now      = new Date();
  const thisMonth = records.filter(r => {
    if (!r.date) return false;
    const d = new Date(r.date);
    return !isNaN(d) && d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
  }).length;
  setMetric("an-consistency", thisMonth);
}

// ---------------------------------------------------------------------------
// Analytics — main Chart.js timeline
// ---------------------------------------------------------------------------
function renderAnalyticsChart(records, filterKey) {
  filterKey = filterKey || "all";

  // Chronological order: oldest → newest = left → right on X axis
  const chron = [...records].reverse();

  // Apply exercise filter
  const filtered = filterKey === "all"
    ? chron
    : chron.filter(r => (r.exercise || "") === filterKey);

  const canvas = document.getElementById("progressChart");
  if (!canvas) return;

  // Destroy stale instance to prevent canvas memory leaks / hover ghosting
  if (progressChartInstance) {
    progressChartInstance.destroy();
    progressChartInstance = null;
  }

  if (!filtered.length) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const labels = filtered.map(r => r.date || "—");
  const scores = filtered.map(r => r.score ?? r.avg_score ?? 0);

  progressChartInstance = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Form Score",
        data: scores,
        borderColor: "#10B981",
        backgroundColor: "rgba(16, 185, 129, 0.05)",
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: "#10B981",
        pointBorderColor: "#FFFFFF",
        pointBorderWidth: 2,
        pointHoverRadius: 6,
        fill: true,
        tension: 0.4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,   // parent .analytics-canvas-wrap controls height
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0F172A",
          titleColor: "#94A3B8",
          bodyColor: "#F8FAFC",
          borderColor: "#1E293B",
          borderWidth: 1,
          padding: 10,
          titleFont: { family: "'DM Mono', monospace", size: 11 },
          bodyFont: { family: "'DM Mono', monospace", size: 13, weight: "500" },
          callbacks: {
            title: (items) => items[0]?.label || "",
            label: (item)  => `  Score: ${item.raw}/100`,
          },
        },
      },
      scales: {
        x: {
          grid:   { display: false },   // hide vertical grid lines entirely
          border: { display: false },
          ticks: {
            color: "#94A3B8",
            font:  { family: "'DM Sans', sans-serif", size: 11 },
            maxRotation: 35,
            maxTicksLimit: 12,
          },
        },
        y: {
          beginAtZero: true,
          max: 100,                     // hard ceiling — score is always /100
          grid: {
            color: "#F1F5F9",
            drawBorder: false,
          },
          border: { display: false },
          ticks: {
            color: "#94A3B8",
            font:  { family: "'DM Mono', monospace", size: 11 },
            stepSize: 20,
          },
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Analytics — per-exercise breakdown cards
// ---------------------------------------------------------------------------
function renderAnalyticsBreakdown(records) {
  const container = document.getElementById("analytics-breakdown");
  if (!container) return;
  container.innerHTML = "";

  EXERCISES.forEach(ex => {
    const subset = records.filter(r => (r.exercise || "") === ex.key);
    if (!subset.length) return;

    const scores    = subset.map(r => r.score ?? r.avg_score ?? 0);
    const avgScore  = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
    const best      = Math.max(...scores);
    const totalReps = subset.reduce((a, r) => a + (r.reps || 0), 0);
    const barWidth  = Math.min(100, avgScore);
    const barColor  = avgScore >= 80 ? "var(--accent-green)"
                    : avgScore >= 60 ? "var(--accent-amber)"
                    :                  "var(--accent-red)";

    container.insertAdjacentHTML("beforeend", `
      <div class="breakdown-card">
        <div class="breakdown-card-header">
          <div class="breakdown-card-icon">${ex.icon}</div>
          <div>
            <div class="breakdown-card-title">${ex.title}</div>
            <div class="breakdown-card-count">${subset.length} session${subset.length !== 1 ? "s" : ""}</div>
          </div>
        </div>
        <div class="score-bar-wrap">
          <div class="score-bar-fill" style="width:${barWidth}%;background:${barColor}"></div>
        </div>
        <div class="breakdown-stat-row">
          <span class="breakdown-stat-label">Avg Score</span>
          <span class="breakdown-stat-value">${avgScore}/100</span>
        </div>
        <div class="breakdown-stat-row">
          <span class="breakdown-stat-label">Best Score</span>
          <span class="breakdown-stat-value">${best}/100</span>
        </div>
        <div class="breakdown-stat-row">
          <span class="breakdown-stat-label">Total Reps</span>
          <span class="breakdown-stat-value">${totalReps}</span>
        </div>
      </div>`);
  });

  if (!container.children.length) {
    container.innerHTML = `
      <div class="analytics-empty" style="grid-column:1/-1">
        <div class="empty-icon">📊</div>
        <div class="empty-text">Complete sessions to see per-exercise breakdowns.</div>
      </div>`;
  }
}

// ---------------------------------------------------------------------------
// KPI strip (Hub)
// ---------------------------------------------------------------------------
function setMetric(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function updateScoreRing(score) {
  const el = document.getElementById("score-ring-value");
  if (!el) return;
  el.textContent = score;
  el.className = "score-ring-value";
  if (score >= 80) el.style.color = "var(--accent-green)";
  else if (score >= 60) el.style.color = "var(--accent-amber)";
  else el.style.color = "var(--accent-red)";
}

function updateFeedback(text) {
  const el = document.getElementById("feedback-chip");
  if (!el) return;
  el.textContent = text;
  el.className = "feedback-chip";
  const bad = ["Cheat", "Compensatory", "Incomplete", "Half Rep", "Drag", "Heave", "Swing"].some(k => text.includes(k));
  if (bad) el.classList.add("warn");
}

function updateStatusPill(text, cls) {
  const pill = document.getElementById("status-pill");
  if (!pill) return;
  pill.className = "status-pill" + (cls ? ` ${cls}` : "");
  const span = pill.querySelector(".status-text");
  if (span) span.textContent = text;
}

function resetAnalysisMetrics() {
  ["metric-reps", "metric-score", "metric-knee", "metric-trunk", "metric-elbow"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = "—";
  });
  updateScoreRing(0);
  const fb = document.getElementById("feedback-chip");
  if (fb) { fb.textContent = "Awaiting session…"; fb.className = "feedback-chip"; }
  updateStatusPill("OFFLINE", "");
}

// ---------------------------------------------------------------------------
// KPI strip (Hub)
// ---------------------------------------------------------------------------
function updateKpis(records) {
  const total = records.length;
  const avgScore = total
    ? Math.round(records.reduce((s, r) => s + (r.score || 0), 0) / total)
    : 0;
  const totalReps = records.reduce((s, r) => s + (r.reps || 0), 0);
  const avgPain   = total
    ? (records.reduce((s, r) => s + (r.pain_level || 0), 0) / total).toFixed(1)
    : "—";

  setMetric("kpi-sessions",   total);
  setMetric("kpi-avg-score",  avgScore);
  setMetric("kpi-total-reps", totalReps);
  setMetric("kpi-avg-pain",   avgPain);
}

// ---------------------------------------------------------------------------
// Patient Records — nested accordion
// ---------------------------------------------------------------------------
function populateRecords(records) {
  const container = document.getElementById("records-list");
  if (!container) return;
  container.innerHTML = "";

  if (!records.length) {
    container.innerHTML = `<p class="text-muted" style="text-align:center;padding:40px 0">No sessions recorded yet.</p>`;
    return;
  }

  // Newest first
  [...records].reverse().forEach((rec, i) => {
    const ex       = EXERCISES.find(e => e.key === (rec.exercise || "squat")) || EXERCISES[0];
    const scoreNum = rec.score || rec.avg_score || 0;
    const scoreClass = scoreNum >= 80 ? "high" : scoreNum >= 60 ? "mid" : "low";
    const details  = Array.isArray(rec.details) ? rec.details : [];

    const repRows = details.length
      ? details.map(d => `
          <tr>
            <td>${d.rep_num || "—"}</td>
            <td><span class="rep-score-pill ${d.score>=80?'high':d.score>=60?'mid':'low'}">${d.score ?? "—"}</span></td>
            <td>${d.issue || "—"}</td>
          </tr>`).join("")
      : `<tr><td colspan="3" style="color:var(--text-muted);font-style:italic">No rep-level data for this session.</td></tr>`;

    const html = `
      <div class="record-accordion" id="acc-${i}">
        <div class="record-header" onclick="toggleAccordion('acc-${i}')">
          <div class="record-ex-icon">${ex.icon}</div>
          <div class="record-meta">
            <div class="record-date">${rec.date || "Unknown date"} — ${ex.title}</div>
            <div class="record-summary">${rec.reps || 0} reps &nbsp;·&nbsp; Pain ${rec.pain_level ?? "—"}/10</div>
          </div>
          <div class="record-score-badge ${scoreNum>=80?'text-success':scoreNum>=60?'text-warn':'text-error'}">${scoreNum}</div>
          <span class="record-chevron">▾</span>
        </div>
        <div class="record-body">
          <table class="rep-table">
            <thead>
              <tr>
                <th>Rep #</th>
                <th>Score</th>
                <th>Feedback</th>
              </tr>
            </thead>
            <tbody>${repRows}</tbody>
          </table>
        </div>
      </div>`;

    container.insertAdjacentHTML("beforeend", html);
  });
}

function toggleAccordion(id) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle("open");
}

// ---------------------------------------------------------------------------
// Pain Scale Dialog
// ---------------------------------------------------------------------------
function openPainDialog(report) {
  const overlay = document.getElementById("pain-overlay");
  if (!overlay) return;
  overlay.classList.add("open");

  // Reset slider
  const slider = document.getElementById("pain_slider");
  if (slider) { slider.value = 0; updatePainUi(0); }
}

function closePainDialog() {
  document.getElementById("pain-overlay")?.classList.remove("open");
}

function updatePainUi(val) {
  val = parseInt(val, 10);
  const display = document.getElementById("pain-level-display");
  const label   = document.getElementById("pain-level-label");
  const img     = document.getElementById("pain_image");

  if (display) display.textContent = val;
  if (label)   label.textContent   = PAIN_LABELS[val] || "";

  // Dynamic image: pain_imgs/00.PNG … pain_imgs/10.PNG
  if (img) {
    const padded = String(val).padStart(2, "0");
    img.src = `../pain_imgs/${padded}.PNG`;
  }

  // Colour the number
  if (display) {
    if (val <= 2)      display.style.color = "var(--accent-green)";
    else if (val <= 5) display.style.color = "var(--accent-amber)";
    else               display.style.color = "var(--accent-red)";
  }
}

function submitPainScore() {
  const slider = document.getElementById("pain_slider");
  const pain   = slider ? parseInt(slider.value, 10) : 0;
  const report = ui.lastReport || {};

  closePainDialog();

  if (!backend) return;

  // Add record locally in Records page
  const newRecord = {
    date:       new Date().toLocaleDateString("en-GB", { day:"2-digit", month:"short", year:"numeric" }),
    exercise:   ui.currentExercise || "squat",
    reps:       report.reps || 0,
    score:      report.avg_score || 0,
    pain_level: pain,
    details:    report.details || [],
  };
  populateRecords([newRecord, ...currentRecords()]);

  backend.submit_pain_score(
    String(pain),
    ui.currentExercise || "",
    report.reps || 0,
    report.avg_score || 0,
    JSON.stringify(report.details || [])
  );
}

function currentRecords() {
  // Return any records already in the list (for combining with new one)
  // This is a lightweight approach — full truth is always fetched from server
  return [];
}

// ---------------------------------------------------------------------------
// Toast notifications
// ---------------------------------------------------------------------------
function toast(msg, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

// ---------------------------------------------------------------------------
// DOM ready
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // Navigation buttons
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => navigate(btn.dataset.page));
  });

  // Exercise cards
  document.querySelectorAll(".exercise-card[data-key]").forEach(card => {
    card.addEventListener("click", () => launchExercise(card.dataset.key));
  });

  // Back button on analysis page
  document.getElementById("back-btn")?.addEventListener("click", () => {
    if (ui.sessionRunning) stopSession();
    navigate("hub");
  });

  // Session start/stop
  document.getElementById("session-btn")?.addEventListener("click", toggleSession);

  // Settings sliders
  document.getElementById("height-slider")?.addEventListener("input", () =>
    onSettingSlider("USER_HEIGHT_CM", "height-slider", "height-val", v => `${parseFloat(v).toFixed(1)} cm`));
  document.getElementById("weight-slider")?.addEventListener("input", () =>
    onSettingSlider("USER_WEIGHT_KG", "weight-slider", "weight-val", v => `${parseFloat(v).toFixed(1)} kg`));

  // Settings toggles
  document.getElementById("voice-toggle")?.addEventListener("change", () =>
    onSettingToggle("VOICE_ON", "voice-toggle"));
  document.getElementById("ar-toggle")?.addEventListener("change", () =>
    onSettingToggle("AR_MODE", "ar-toggle"));

  // Pain slider
  document.getElementById("pain_slider")?.addEventListener("input", (e) =>
    updatePainUi(e.target.value));

  // Pain dialog buttons
  document.getElementById("pain-cancel-btn")?.addEventListener("click", closePainDialog);
  document.getElementById("pain-submit-btn")?.addEventListener("click", submitPainScore);

  // Analytics filter pills
  document.querySelectorAll(".filter-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".filter-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      renderAnalyticsChart(_cachedRecords, pill.dataset.filter);
    });
  });

  // Start on hub
  navigate("hub");
  setVideoFeed(false);
});