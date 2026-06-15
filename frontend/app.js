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
  backend.goals_loaded.connect(onGoalsLoaded);
  backend.achievements_loaded.connect(onAchievementsLoaded);

  // ── Seed the UI with real backend state ─────────────────────────────────
  backend.get_initial_state((stateJson) => {
    const s = JSON.parse(stateJson);
    initUi(s);
  });

  // ── Load history for Records page ───────────────────────────────────────
  backend.fetch_history();
  backend.fetch_goals();
  backend.fetch_achievements();
});

// ---------------------------------------------------------------------------
// App state (UI-only; ground truth lives in Python)
// ---------------------------------------------------------------------------
const ui = {
  currentPage: "hub",
  currentExercise: null,
  sessionRunning: false,
  lastReport: null,
  painPromptEnabled: true,
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
  { key: "squat",           title: "Deep Squat",    desc: "Knee & hip mobility analysis",                    icon: "🦵" },
  { key: "sts",             title: "Sit to Stand",  desc: "Geriatric fall-risk assessment",                  icon: "🪑" },
  { key: "pushup",          title: "Push-up",       desc: "Upper body & core stabilisation",                 icon: "💪" },
  { key: "curl",            title: "Bicep Curl",    desc: "Elbow ROM & cheat classification",                icon: "🏋️" },
  { key: "lateral_raise",   title: "Lateral Raise", desc: "Shoulder abduction & symmetry analysis",          icon: "🙆" },
  { key: "knee_extension",  title: "Knee Extension", desc: "Seated straight-leg raise & ROM analysis",       icon: "🦵" },
  { key: "wall_pushup",     title: "Wall Push-Up",   desc: "Upper body & shoulder mobility analysis",        icon: "🤜" },
  { key: "hip_march",       title: "Hip March",      desc: "Hip flexor mobility & symmetry analysis",        icon: "🏃" },
];

const EXERCISE_LABELS = Object.fromEntries(EXERCISES.map(e => [e.key, e.title]));

const PAIN_LABELS = [
  "No Pain", "Very Mild", "Mild", "Moderate Mild",
  "Moderate", "Moderate Severe", "Severe", "Very Severe",
  "Intense", "Extremely Intense", "Unbearable"
];

const ACHIEVEMENT_DESCS = {
  first_rep:         "Completed your very first session.",
  ten_sessions:      "Completed 10 sessions.",
  fifty_sessions:    "Completed 50 sessions.",
  hundred_sessions:  "Completed 100 sessions.",
  perfect_score:     "Achieved a perfect form score of 100 on a rep.",
  high_scorer:       "Averaged 90+ form score across 10+ sessions.",
  all_rounder:       "Tried all 5 exercises at least once.",
  pain_warrior:      "Completed a session reporting pain level 7 or above.",
  hundred_reps:      "Accumulated 100 total reps across all sessions.",
  five_hundred_reps: "Accumulated 500 total reps across all sessions.",
  comeback_kid:      "Returned to training after a 7+ day absence.",
  streak_7:          "Exercised on 7 consecutive calendar days.",
};

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
  // Avatar / greeting
  const initial = (s.username || "U")[0].toUpperCase();
  document.querySelectorAll(".user-avatar").forEach(el => el.textContent = initial);
  document.querySelectorAll(".user-name").forEach(el => el.textContent = s.username || "User");

  // Profile section
  const profileEl = document.getElementById("profile-username");
  if (profileEl) profileEl.textContent = s.username || "—";

  // Existing biometrics & system
  setSlider("height-slider", "height-val", s.USER_HEIGHT_CM, v => `${v} cm`);
  setSlider("weight-slider", "weight-val", s.USER_WEIGHT_KG, v => `${v} kg`);
  setToggle("voice-toggle", s.VOICE_ON);
  setToggle("ar-toggle",    s.AR_MODE);

  // Camera & Capture
  const camSel = document.getElementById("camera-select");
  if (camSel) camSel.value = s.CAMERA_INDEX ?? 0;
  setToggle("mirror-toggle", s.MIRROR_VIDEO !== false);

  // Notifications & Session
  setToggle("pain-prompt-toggle", s.PAIN_PROMPT_ENABLED !== false);
  ui.painPromptEnabled = s.PAIN_PROMPT_ENABLED !== false;
  setSlider("session-timeout-slider", "session-timeout-val", s.SESSION_TIMEOUT_MINS ?? 0,
    v => parseInt(v) === 0 ? "Off" : `${v} min`);
  setSlider("rep-target-slider", "rep-target-val", s.DEFAULT_REP_TARGET ?? 0,
    v => parseInt(v) === 0 ? "None" : String(parseInt(v)));

  // Advanced thresholds
  setSlider("squat-depth-slider", "squat-depth-val", s.PARAM_SQUAT_DEPTH ?? 140, v => `${v}°`);
  setSlider("lean-warn-slider",   "lean-warn-val",   s.PARAM_LEAN_WARN   ?? 40,  v => `${v}°`);
  setSlider("lean-crit-slider",   "lean-crit-val",   s.PARAM_LEAN_CRIT   ?? 55,  v => `${v}°`);
  setSlider("rounding-slider",    "rounding-val",    s.PARAM_ROUNDING    ?? 18,  v => `${v}°`);
  setSlider("mp-detect-slider",   "mp-detect-val",   s.MP_DETECTION_CONFIDENCE ?? 0.5,
    v => parseFloat(v).toFixed(2));
  setSlider("mp-track-slider",    "mp-track-val",    s.MP_TRACKING_CONFIDENCE  ?? 0.5,
    v => parseFloat(v).toFixed(2));

  // Restore client-side preferences from localStorage
  const savedTheme  = localStorage.getItem("pv-theme");
  const savedAccent = localStorage.getItem("pv-accent");
  const savedSize   = localStorage.getItem("pv-font-size");
  if (savedTheme === "dark") applyTheme(true, false);
  if (savedAccent) applyAccent(savedAccent, false);
  if (savedSize)   applyFontSize(parseFloat(savedSize), false);
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
// Appearance helpers (client-side, persisted in localStorage)
// ---------------------------------------------------------------------------
function applyTheme(isDark, save = true) {
  document.body.classList.toggle("dark-mode", isDark);
  const toggle = document.getElementById("dark-mode-toggle");
  if (toggle) toggle.checked = isDark;
  if (save) localStorage.setItem("pv-theme", isDark ? "dark" : "light");
}

function applyAccent(color, save = true) {
  document.documentElement.style.setProperty("--accent-blue", color);
  document.querySelectorAll(".accent-swatch").forEach(s => {
    s.classList.toggle("active", s.dataset.accent === color);
  });
  if (save) localStorage.setItem("pv-accent", color);
}

function applyFontSize(size, save = true) {
  document.documentElement.style.fontSize = size + "px";
  const slider = document.getElementById("font-size-slider");
  const val    = document.getElementById("font-size-val");
  if (slider) slider.value = size;
  if (val)    val.textContent = `${size} px`;
  if (save) localStorage.setItem("pv-font-size", size);
}

// ---------------------------------------------------------------------------
// Analysis difficulty presets
// ---------------------------------------------------------------------------
const DIFFICULTY_PRESETS = {
  easy:   { PARAM_SQUAT_DEPTH: 150, PARAM_LEAN_WARN: 50, PARAM_LEAN_CRIT: 65, PARAM_ROUNDING: 25 },
  normal: { PARAM_SQUAT_DEPTH: 140, PARAM_LEAN_WARN: 40, PARAM_LEAN_CRIT: 55, PARAM_ROUNDING: 18 },
  strict: { PARAM_SQUAT_DEPTH: 120, PARAM_LEAN_WARN: 28, PARAM_LEAN_CRIT: 40, PARAM_ROUNDING: 12 },
};

function setDifficulty(preset) {
  const p = DIFFICULTY_PRESETS[preset];
  if (!p) return;
  document.querySelectorAll(".difficulty-pill").forEach(pill => {
    pill.classList.toggle("active", pill.id === `diff-${preset}`);
  });
  Object.entries(p).forEach(([key, val]) => {
    if (backend) backend.update_setting(key, JSON.stringify(val));
  });
  setSlider("squat-depth-slider", "squat-depth-val", p.PARAM_SQUAT_DEPTH, v => `${v}°`);
  setSlider("lean-warn-slider",   "lean-warn-val",   p.PARAM_LEAN_WARN,   v => `${v}°`);
  setSlider("lean-crit-slider",   "lean-crit-val",   p.PARAM_LEAN_CRIT,   v => `${v}°`);
  setSlider("rounding-slider",    "rounding-val",    p.PARAM_ROUNDING,    v => `${v}°`);
  toast(`Difficulty set to ${preset[0].toUpperCase() + preset.slice(1)}.`, "success");
}

// ---------------------------------------------------------------------------
// Profile
// ---------------------------------------------------------------------------
function changePassword() {
  toast("To change your password, please visit the Physio-Vision web portal.", "info");
}

// ---------------------------------------------------------------------------
// Data & Privacy
// ---------------------------------------------------------------------------
function exportData() {
  if (!_cachedRecords.length) { toast("No session data to export.", "warning"); return; }
  if (!backend) { toast("Backend not connected.", "error"); return; }
  backend.export_history(JSON.stringify(_cachedRecords, null, 2), (resultStr) => {
    const result = JSON.parse(resultStr);
    if (result.cancelled) return;
    if (result.ok) toast("Data exported successfully.", "success");
    else toast("Export failed.", "error");
  });
}

function clearHistory() {
  document.getElementById("clear-history-btn")?.classList.add("hidden");
  document.getElementById("clear-history-confirm")?.classList.remove("hidden");
}

function cancelClearHistory() {
  document.getElementById("clear-history-btn")?.classList.remove("hidden");
  document.getElementById("clear-history-confirm")?.classList.add("hidden");
}

function confirmClearHistory() {
  cancelClearHistory();
  _cachedRecords = [];
  populateRecords([]);
  updateKpis([]);
  renderAnalyticsChart([], "all");
  renderAnalyticsBreakdown([]);
  renderAnalyticsKpis([]);
  toast("Session history cleared.", "success");
  if (backend) backend.clear_history();
}

// ---------------------------------------------------------------------------
// Settings — two-way bridge
// ---------------------------------------------------------------------------
function onSettingSelect(key, selectId) {
  const el = document.getElementById(selectId);
  if (!el || !backend) return;
  backend.update_setting(key, JSON.stringify(parseInt(el.value)));
}

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
  if (ui.painPromptEnabled) {
    openPainDialog(report);
  } else {
    _autoSubmitSession(report);
  }
}

function _autoSubmitSession(report) {
  const newRecord = {
    date:       new Date().toLocaleDateString("en-GB", { day:"2-digit", month:"short", year:"numeric" }),
    exercise:   ui.currentExercise || "squat",
    reps:       report.reps || 0,
    score:      report.avg_score || 0,
    pain_level: 0,
    details:    report.details || [],
  };
  _cachedRecords = [newRecord, ..._cachedRecords];
  populateRecords(_cachedRecords);
  updateKpis(_cachedRecords);
  renderAnalyticsChart(_cachedRecords, "all");
  renderAnalyticsBreakdown(_cachedRecords);
  renderAnalyticsKpis(_cachedRecords);
  if (backend) {
    backend.submit_pain_score("0", ui.currentExercise || "", report.reps || 0,
      report.avg_score || 0, JSON.stringify(report.details || []));
    setTimeout(() => { backend.fetch_goals(); backend.fetch_achievements(); }, 2000);
  }
}

function onHistoryLoaded(jsonStr) {
  const records = JSON.parse(jsonStr);

  // FIX 1: Parse the SQLite stringified JSON details back into actual JS Arrays
  records.forEach(r => {
    if (typeof r.details === 'string') {
      try {
        r.details = JSON.parse(r.details);
      } catch (e) {
        r.details = [];
      }
    }
  });

  _cachedRecords = records;
  populateRecords(records);
  updateKpis(records);
  renderAnalyticsChart(records, "all");
  renderAnalyticsBreakdown(records);
  renderAnalyticsKpis(records);
}

// ---------------------------------------------------------------------------
// Goals — signal handler
// ---------------------------------------------------------------------------
function onGoalsLoaded(jsonStr) {
  const goals = JSON.parse(jsonStr);
  renderGoals(goals);
}

// ---------------------------------------------------------------------------
// Achievements — signal handler
// ---------------------------------------------------------------------------
function onAchievementsLoaded(jsonStr) {
  const achievements = JSON.parse(jsonStr);
  renderAchievements(achievements);
}

// ---------------------------------------------------------------------------
// Render active goals with progress bars
// ---------------------------------------------------------------------------
const GOAL_TYPE_LABELS = { reps: "Total Reps", score: "Avg Score", sessions: "Sessions" };
const EXERCISE_ICONS   = {
  any: "🏃", squat: "🦵", sts: "🪑", pushup: "💪",
  curl: "🏋️", lateral_raise: "🙆"
};

function renderGoals(goals) {
  const container = document.getElementById("goals-list");
  if (!container) return;

  if (!goals || goals.length === 0) {
    container.innerHTML = `
      <div class="goals-empty">
        No active goals yet. Hit <strong>+ Add Goal</strong> to create one.
      </div>`;
    return;
  }

  container.innerHTML = goals.map(g => {
    const icon      = EXERCISE_ICONS[g.exercise] || "🏃";
    const typeLabel = GOAL_TYPE_LABELS[g.goal_type] || g.goal_type;
    const pct       = g.progress_pct ?? 0;
    const fillClass = pct >= 80 ? "" : pct >= 40 ? "warn" : "crit";
    const deadline  = g.deadline
      ? `  ·  Due ${new Date(g.deadline).toLocaleDateString("en-GB",
          { day: "2-digit", month: "short", year: "numeric" })}`
      : "";

    let currentLabel, targetLabel;
    if (g.goal_type === "score") {
      currentLabel = `${g.current_value}/100 avg`;
      targetLabel  = `Target: ${g.target_value}/100`;
    } else {
      currentLabel = `${g.current_value} / ${g.target_value}`;
      targetLabel  = `${pct}%`;
    }

    return `
      <div class="goal-item">
        <div class="goal-item-header">
          <div>
            <div class="goal-item-label">${icon} ${typeLabel}${deadline}</div>
            <div class="goal-item-meta">${
              g.exercise === "any" ? "Any Exercise" : (g.exercise || "").replace(/_/g, " ")
            }</div>
          </div>
          <button class="goal-delete-btn" title="Remove goal"
                  onclick="deleteGoal(${g.id})">✕</button>
        </div>
        <div class="goal-progress-track">
          <div class="goal-progress-fill ${fillClass}"
               style="width:${pct}%"></div>
        </div>
        <div class="goal-progress-label">
          <span>${currentLabel}</span>
          <span>${targetLabel}</span>
        </div>
      </div>`;
  }).join("");
}

// ---------------------------------------------------------------------------
// Render achievement badges (locked + unlocked)
// ---------------------------------------------------------------------------
const TIER_COLORS = { gold: "tier-gold", silver: "tier-silver", bronze: "" };

function renderAchievements(achievements) {
  const grid = document.getElementById("achievements-grid");
  if (!grid || !achievements) return;

  // Show unlocked first, then locked
  const sorted = [
    ...achievements.filter(a => a.unlocked),
    ...achievements.filter(a => !a.unlocked),
  ];

  grid.innerHTML = sorted.map(a => {
    const lockedClass   = a.unlocked ? `unlocked ${TIER_COLORS[a.tier] || ""}` : "locked";
    const dateStr       = a.unlocked && a.unlocked_at
      ? new Date(a.unlocked_at).toLocaleDateString("en-GB",
          { day: "2-digit", month: "short", year: "numeric" })
      : null;
    const desc     = ACHIEVEMENT_DESCS[a.key] || a.desc || "";
    const lockIcon = a.unlocked ? "" : `<div class="achievement-icon">🔒</div>`;
    const realIcon      = a.unlocked ? `<div class="achievement-icon">${a.icon}</div>` : "";

    return `
      <div class="achievement-badge ${lockedClass}" title="${a.desc}">
        ${a.unlocked ? realIcon : lockIcon}
        <div class="achievement-title">${a.title}</div>
        ${a.unlocked && dateStr
          ? `<div class="achievement-date">${dateStr}</div>`
          : `<div class="achievement-desc">${desc}</div>`
        }
      </div>`;
  }).join("");
}

// ---------------------------------------------------------------------------
// Goal creation modal
// ---------------------------------------------------------------------------
const GOAL_HINTS = {
  reps:     "Accumulate this many reps from today.",
  sessions: "Complete this many sessions from today.",
  score:    "Reach this average form score (0 – 100).",
};

function openGoalDialog() {
  const overlay = document.getElementById("goal-modal-overlay");
  if (overlay) overlay.classList.add("open");
  updateGoalHint();
}

function closeGoalDialog() {
  const overlay = document.getElementById("goal-modal-overlay");
  if (overlay) overlay.classList.remove("open");
}

function updateGoalHint() {
  const type  = document.getElementById("gm-type")?.value || "reps";
  const hint  = document.getElementById("gm-hint");
  if (hint) hint.textContent = GOAL_HINTS[type] || "";

  // Cap score target at 100
  const targetInput = document.getElementById("gm-target");
  if (targetInput) {
    if (type === "score") {
      targetInput.max         = "100";
      targetInput.placeholder = "e.g. 85";
      if (parseInt(targetInput.value) > 100) targetInput.value = "85";
    } else {
      targetInput.max         = "10000";
      targetInput.placeholder = "e.g. 50";
    }
  }
}

function submitGoal() {
  if (!backend) return;

  const exercise = document.getElementById("gm-exercise")?.value || "any";
  const goalType = document.getElementById("gm-type")?.value     || "reps";
  const target   = parseFloat(document.getElementById("gm-target")?.value || "0");
  const deadline = document.getElementById("gm-deadline")?.value  || null;

  if (!target || target <= 0) {
    toast("Please enter a valid target.", "error");
    return;
  }
  if (goalType === "score" && target > 100) {
    toast("Score target cannot exceed 100.", "error");
    return;
  }

  closeGoalDialog();

  backend.create_goal(
    exercise,
    goalType,
    JSON.stringify(target),
    JSON.stringify(deadline)   // "null" or "\"2026-12-31\""
  );
  toast("Goal created!", "success");
}

function deleteGoal(goalId) {
  if (!backend) return;
  backend.delete_goal(goalId);
  toast("Goal removed.", "info");
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

  // Render newest-first, but preserve the original index into _cachedRecords
  // so exportSessionToPDF() can do a stable lookup regardless of render order.
  [...records].reverse().forEach((rec, i) => {
    // i=0 is the newest record, which lives at records[records.length-1-i]
    const origIdx  = records.length - 1 - i;
    const ex       = EXERCISES.find(e => e.key === (rec.exercise || "squat")) || EXERCISES[0];
    const scoreNum = rec.score || rec.avg_score || 0;
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
          <button class="btn-export-pdf"
                  onclick="event.stopPropagation(); exportSessionToPDF(${origIdx})">
            ↓ Export PDF
          </button>
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

// ---------------------------------------------------------------------------
// PDF Export
// ---------------------------------------------------------------------------
function exportSessionToPDF(origIdx) {
  // ── 1. Look up the record from the stable cache ──────────────────────────
  const rec = _cachedRecords[origIdx];
  if (!rec) {
    toast("Record not found — please refresh and try again.", "error");
    return;
  }

  const ex       = EXERCISES.find(e => e.key === (rec.exercise || "squat")) || EXERCISES[0];
  const scoreNum = rec.score ?? rec.avg_score ?? 0;
  const details  = Array.isArray(rec.details) ? rec.details : [];

  // ── 2. Derive computed fields ─────────────────────────────────────────────
  const scoreClass = scoreNum >= 80 ? "good" : scoreNum >= 60 ? "mid" : "low";
  const ratingText = scoreNum >= 80 ? "Excellent"
                   : scoreNum >= 60 ? "Acceptable"
                   :                  "Needs Work";
  const painLabel  = PAIN_LABELS[rec.pain_level] || String(rec.pain_level ?? "—");
  const now        = new Date();
  const generatedStr = now.toLocaleDateString("en-GB", {
    day: "2-digit", month: "long", year: "numeric",
  }) + "  " + now.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });

  // Filename: e.g. "Physio_Report_Squat_2024-06-01.pdf"
  const dateSlug = (rec.date || "").replace(/\s/g, "-").replace(/[^a-zA-Z0-9\-]/g, "") || "Session";
  const filename = `Physio_Report_${ex.title.replace(/\s+/g, "_")}_${dateSlug}.pdf`;

  // ── 3. Populate the hidden template ──────────────────────────────────────
  const tmpl = document.getElementById("clinical-report-template");

  // Header / meta
  document.getElementById("cr-generated-date").textContent = generatedStr;
  document.getElementById("cr-session-date").textContent   = rec.date || "—";
  document.getElementById("cr-exercise-name").textContent  = ex.title;

  // Patient name — read from the live sidebar element
  const nameEl = document.querySelector(".sidebar-user .user-name");
  document.getElementById("cr-patient-name").textContent = nameEl ? nameEl.textContent : "Patient";

  // Summary cards
  document.getElementById("cr-total-reps").textContent = rec.reps ?? "—";

  const scoreEl = document.getElementById("cr-form-score");
  scoreEl.textContent = scoreNum;
  scoreEl.className   = `cr-summary-value cr-score-${scoreClass}`;

  document.getElementById("cr-pain-level").textContent = rec.pain_level ?? "—";

  const ratingEl = document.getElementById("cr-form-rating");
  ratingEl.textContent = ratingText;
  ratingEl.className   = `cr-summary-value cr-score-${scoreClass}`;

  // Rep breakdown table body
  const tbody = document.getElementById("cr-rep-tbody");
  if (details.length) {
    tbody.innerHTML = details.map(d => {
      const s   = d.score ?? 0;
      const cls = s >= 80 ? "good" : s >= 60 ? "mid" : "low";
      const rat = s >= 80 ? "Excellent" : s >= 60 ? "Acceptable" : "Needs Work";
      return `
        <tr>
          <td>${d.rep_num ?? "—"}</td>
          <td class="cr-td-score-${cls}">${s}</td>
          <td class="cr-td-rating-${cls}">${rat}</td>
          <td>${d.issue || "Excellent Form"}</td>
        </tr>`;
    }).join("");
  } else {
    tbody.innerHTML = `
      <tr class="cr-no-data">
        <td colspan="4">No repetition-level data was recorded for this session.</td>
      </tr>`;
  }

  // ── 4. Briefly make the template visible off-screen for html2pdf ──────────
  // We move it off-screen rather than removing display:none so the browser
  // fully lays it out (required for html2canvas to measure dimensions).
  tmpl.style.display    = "block";
  tmpl.style.position   = "absolute";
  tmpl.style.left       = "-9999px";
  tmpl.style.top        = "0";
  tmpl.style.visibility = "hidden";

  const pageEl = tmpl.querySelector(".cr-page");

  // ── 5. Generate and save the PDF ─────────────────────────────────────────
  html2pdf()
    .set({
      margin:   [15, 15, 15, 15],   // top, right, bottom, left in mm
      filename: filename,
      image:    { type: "jpeg", quality: 0.98 },
      html2canvas: {
        scale: 2,                   // 2× for crisp text at high DPI
        useCORS: true,
        backgroundColor: "#FFFFFF",
      },
      jsPDF: {
        unit:        "mm",
        format:      "a4",
        orientation: "portrait",
      },
    })
    .from(pageEl)
    .save()
    .then(() => {
      // Restore the template to its hidden state
      tmpl.style.display    = "none";
      tmpl.style.position   = "";
      tmpl.style.left       = "";
      tmpl.style.top        = "";
      tmpl.style.visibility = "";
      toast(`PDF saved: ${filename}`, "success");
    })
    .catch((err) => {
      tmpl.style.display = "none";
      console.error("[PDF Export]", err);
      toast("PDF export failed — see console for details.", "error");
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

  // 1. Force undeniable geometry (Bypasses the CSS 'inset' bug)
  overlay.style.position = "fixed";
  overlay.style.top = "0";
  overlay.style.left = "0";
  overlay.style.width = "100vw";
  overlay.style.height = "100vh";
  overlay.style.backgroundColor = "rgba(15, 23, 42, 0.95)";
  overlay.style.zIndex = "9999";

  // 2. Force it to display
  overlay.style.display = "flex";
  overlay.classList.add("open");

  // Reset slider
  const slider = document.getElementById("pain_slider");
  if (slider) { slider.value = 0; updatePainUi(0); }
}

function closePainDialog() {
  const overlay = document.getElementById("pain-overlay");
  if (overlay) {
    overlay.style.display = "none";
    overlay.classList.remove("open");
  }
}

function updatePainUi(val) {
  val = parseInt(val, 10);
  const display = document.getElementById("pain-level-display");
  const label   = document.getElementById("pain-level-label");
  const img     = document.getElementById("pain_image");

  if (display) display.textContent = val;
  if (label)   label.textContent   = PAIN_LABELS[val] || "";

  // ── Range-based Image Selector ──
  if (img) {
    let imgName;
    if      (val <= 1)  imgName = "01";
    else if (val <= 3)  imgName = "23";
    else if (val <= 5)  imgName = "45";
    else if (val <= 7)  imgName = "67";
    else if (val <= 9)  imgName = "89";
    else                imgName = "10";

    img.src = `pain_imgs/Squat/${imgName}.PNG`;
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

  // FIX 2: Add the new record properly to the cache and re-render everything
  const newRecord = {
    date:       new Date().toLocaleDateString("en-GB", { day:"2-digit", month:"short", year:"numeric" }),
    exercise:   ui.currentExercise || "squat",
    reps:       report.reps || 0,
    score:      report.avg_score || 0,
    pain_level: pain,
    details:    report.details || [],
  };

  // Add the new record to the very top of the master cache
  _cachedRecords = [newRecord, ..._cachedRecords];

  // Refresh all UI elements so the new record appears instantly without deleting history
  populateRecords(_cachedRecords);
  updateKpis(_cachedRecords);
  renderAnalyticsChart(_cachedRecords, "all");
  renderAnalyticsBreakdown(_cachedRecords);
  renderAnalyticsKpis(_cachedRecords);

  backend.submit_pain_score(
    String(pain),
    ui.currentExercise || "",
    report.reps || 0,
    report.avg_score || 0,
    JSON.stringify(report.details || [])
  );
  // Re-fetch goals and achievements to reflect new session progress
  if (backend) {
    setTimeout(() => {
      backend.fetch_goals();
      backend.fetch_achievements();
    }, 2000);   // slight delay so the server has time to commit before we query
  }
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

  // Re-fetch goals & achievements when opening those pages
  document.querySelectorAll(".nav-item[data-page='goals'], .nav-item[data-page='achievements']")
    .forEach(btn => {
      btn.addEventListener("click", () => {
        if (!backend) return;
        backend.fetch_goals();
        backend.fetch_achievements();
      });
    });

  // Goal modal — close on overlay background click
  document.getElementById("goal-modal-overlay")?.addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeGoalDialog();
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

  // Appearance
  document.getElementById("dark-mode-toggle")?.addEventListener("change", e =>
    applyTheme(e.target.checked));
  document.querySelectorAll(".accent-swatch").forEach(s =>
    s.addEventListener("click", () => applyAccent(s.dataset.accent)));
  document.getElementById("font-size-slider")?.addEventListener("input", e =>
    applyFontSize(parseFloat(e.target.value)));

  // Camera & Capture
  document.getElementById("camera-select")?.addEventListener("change", () =>
    onSettingSelect("CAMERA_INDEX", "camera-select"));
  document.getElementById("mirror-toggle")?.addEventListener("change", () =>
    onSettingToggle("MIRROR_VIDEO", "mirror-toggle"));

  // Notifications & Session
  document.getElementById("pain-prompt-toggle")?.addEventListener("change", () => {
    onSettingToggle("PAIN_PROMPT_ENABLED", "pain-prompt-toggle");
    const el = document.getElementById("pain-prompt-toggle");
    if (el) ui.painPromptEnabled = el.checked;
  });
  document.getElementById("session-timeout-slider")?.addEventListener("input", () =>
    onSettingSlider("SESSION_TIMEOUT_MINS", "session-timeout-slider", "session-timeout-val",
      v => parseInt(v) === 0 ? "Off" : `${v} min`));
  document.getElementById("rep-target-slider")?.addEventListener("input", () =>
    onSettingSlider("DEFAULT_REP_TARGET", "rep-target-slider", "rep-target-val",
      v => parseInt(v) === 0 ? "None" : String(parseInt(v))));

  // Advanced thresholds
  document.getElementById("squat-depth-slider")?.addEventListener("input", () =>
    onSettingSlider("PARAM_SQUAT_DEPTH", "squat-depth-slider", "squat-depth-val", v => `${v}°`));
  document.getElementById("lean-warn-slider")?.addEventListener("input", () =>
    onSettingSlider("PARAM_LEAN_WARN", "lean-warn-slider", "lean-warn-val", v => `${v}°`));
  document.getElementById("lean-crit-slider")?.addEventListener("input", () =>
    onSettingSlider("PARAM_LEAN_CRIT", "lean-crit-slider", "lean-crit-val", v => `${v}°`));
  document.getElementById("rounding-slider")?.addEventListener("input", () =>
    onSettingSlider("PARAM_ROUNDING", "rounding-slider", "rounding-val", v => `${v}°`));
  document.getElementById("mp-detect-slider")?.addEventListener("input", () =>
    onSettingSlider("MP_DETECTION_CONFIDENCE", "mp-detect-slider", "mp-detect-val",
      v => parseFloat(v).toFixed(2)));
  document.getElementById("mp-track-slider")?.addEventListener("input", () =>
    onSettingSlider("MP_TRACKING_CONFIDENCE", "mp-track-slider", "mp-track-val",
      v => parseFloat(v).toFixed(2)));

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