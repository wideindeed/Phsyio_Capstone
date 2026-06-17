/* =============================================================================
   Physio-Vision  ·  app.js
   JS is a dumb terminal.  All API calls, session logic, and state mutation
   happen in Python (Bridge).  JS only renders what Python tells it.
   ============================================================================= */

"use strict";

// ---------------------------------------------------------------------------
// SVG Icon System — MIT-licensed Lucide-style line icons (no external deps)
// All icons: 24×24 viewBox, stroke-based, inherit currentColor.
// ---------------------------------------------------------------------------
const _i = d => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;

const ICONS = {
  // ── Exercise icons ─────────────────────────────────────────────
  squat:              _i('<path d="M15 4l-5 8 5 8"/>'),
  sts:                _i('<path d="M12 19V5"/><path d="M5 12l7-7 7 7"/>'),
  pushup:             _i('<path d="M7 12h10"/><rect x="3" y="9" width="4" height="6" rx="1"/><rect x="17" y="9" width="4" height="6" rx="1"/>'),
  curl:               _i('<path d="M7 12h10"/><rect x="1" y="9" width="4" height="6" rx="1"/><rect x="5" y="7.5" width="2" height="9" rx=".5"/><rect x="17" y="7.5" width="2" height="9" rx=".5"/><rect x="19" y="9" width="4" height="6" rx="1"/>'),
  lateral_raise:      _i('<circle cx="12" cy="5" r="2"/><path d="M12 7v9"/><path d="M4 11h16"/>'),
  knee_extension:     _i('<path d="M6 12h6l6-6"/>'),
  wall_pushup:        _i('<path d="M20 3v18"/><path d="M6 12h10"/><rect x="2" y="9" width="4" height="6" rx="1"/>'),
  hip_march:          _i('<path d="M2 12h4l3-8 6 16 3-8h4"/>'),
  shoulder_extension: _i('<circle cx="12" cy="5" r="2"/><path d="M12 7v5"/><path d="M6 16l6-4 6 4"/>'),
  shoulder_scaption:  _i('<circle cx="12" cy="8" r="2"/><path d="M12 10v6"/><path d="M7 4l5 6 5-6"/>'),

  // ── UI icons ───────────────────────────────────────────────────
  target:   _i('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>'),
  trophy:   _i('<path d="M6 9a6 6 0 0 0 12 0V4H6v5z"/><path d="M6 7H3v1a3 3 0 0 0 3 3"/><path d="M18 7h3v1a3 3 0 0 1-3 3"/><path d="M10 16h4"/><path d="M12 13v3"/><path d="M8 21h8"/>'),
  lock:     _i('<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>'),
  camera:   _i('<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>'),
  chart:    _i('<path d="M18 20V10M12 20V4M6 20v-6"/>'),
  medical:  _i('<path d="M12 4v16M4 12h16"/>'),
  star:     `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="1em" height="1em" fill="currentColor" stroke="currentColor" stroke-width="1"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>`,
  shield:   _i('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'),
  flag:     _i('<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><path d="M4 22v-7"/>'),
  flame:    _i('<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>'),
  refresh:  _i('<path d="M21 2v6h-6"/><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M3 22v-6h6"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/>'),
  medal:    _i('<circle cx="12" cy="8" r="6"/><path d="M15.477 12.89L17 22l-5-3-5 3 1.523-9.11"/>'),
  trending: _i('<path d="M23 6l-9.5 9.5-5-5L1 18"/><path d="M17 6h6v6"/>'),
  person:   _i('<circle cx="12" cy="5" r="2"/><path d="M12 7v5"/><path d="M8 21l4-9 4 9"/>'),
  clock:    _i('<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>'),
  check:    _i('<path d="M20 6L9 17l-5-5"/>'),
  play:     _i('<polygon points="5 3 19 12 5 21 5 3" fill="currentColor" stroke="none"/>'),
  stop:     _i('<rect x="4" y="4" width="16" height="16" rx="2" fill="currentColor" stroke="none"/>'),
};

// Map exercise keys to their SVG icon
function exerciseIcon(key) {
  return ICONS[key] || ICONS.hip_march;
}

// Map achievement keys to SVG icons (frontend owns icon rendering)
const ACHIEVEMENT_ICONS = {
  first_rep:         ICONS.flag,
  ten_sessions:      ICONS.trending,
  fifty_sessions:    ICONS.medal,
  hundred_sessions:  ICONS.trophy,
  perfect_score:     ICONS.star,
  high_scorer:       ICONS.trending,
  all_rounder:       ICONS.target,
  pain_warrior:      ICONS.shield,
  hundred_reps:      ICONS.medal,
  five_hundred_reps: ICONS.trophy,
  comeback_kid:      ICONS.refresh,
  streak_7:          ICONS.flame,
};

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
  backend.course_progress_loaded.connect(onCourseProgressLoaded);
  backend.groq_status_changed.connect(onGroqStatusChanged);
  backend.groq_key_info.connect(onGroqKeyInfo);

  // ── Seed the UI with real backend state ─────────────────────────────────
  backend.get_initial_state((stateJson) => {
    const s = JSON.parse(stateJson);
    initUi(s);
  });

  // ── Load history for Records page ───────────────────────────────────────
  backend.fetch_history();
  backend.fetch_goals();
  backend.fetch_achievements();
  backend.fetch_course_progress();
  backend.check_groq_status();
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
  activeCourse: null,   // { courseId, stepIndex } when launched from a course
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
  { key: "squat",           title: "Deep Squat",    desc: "Knee & hip mobility analysis",                    icon: ICONS.squat },
  { key: "sts",             title: "Sit to Stand",  desc: "Geriatric fall-risk assessment",                  icon: ICONS.sts },
  { key: "pushup",          title: "Push-up",       desc: "Upper body & core stabilisation",                 icon: ICONS.pushup },
  { key: "curl",            title: "Bicep Curl",    desc: "Elbow ROM & cheat classification",                icon: ICONS.curl },
  { key: "lateral_raise",   title: "Lateral Raise", desc: "Shoulder abduction & symmetry analysis",          icon: ICONS.lateral_raise },
  { key: "knee_extension",  title: "Knee Extension", desc: "Seated straight-leg raise & ROM analysis",       icon: ICONS.knee_extension },
  { key: "wall_pushup",     title: "Wall Push-Up",   desc: "Upper body & shoulder mobility analysis",        icon: ICONS.wall_pushup },
  { key: "hip_march",          title: "Hip March",          desc: "Hip flexor mobility & symmetry analysis",         icon: ICONS.hip_march },
  { key: "shoulder_extension", title: "Shoulder Extension", desc: "Standing shoulder extension & posture analysis",   icon: ICONS.shoulder_extension },
  { key: "shoulder_scaption",  title: "Shoulder Scaption",  desc: "Diagonal Y-raise & rotator cuff analysis",         icon: ICONS.shoulder_scaption },
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
  if (pageId === "analysis" && !ui.currentExercise) {
    toast("Select an exercise from the Hub first.", "warning");
    pageId = "hub";
  }
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
  document.getElementById("analysis-exercise-icon").innerHTML =
    EXERCISES.find(e => e.key === key)?.icon || ICONS.hip_march;

  // Reset metrics
  resetAnalysisMetrics();
  navigate("analysis");
}

// ---------------------------------------------------------------------------
// Initialise UI from Python state
// ---------------------------------------------------------------------------
function initUi(s) {
  // Avatar / greeting
  const rawName = s.username || "User";
  // Capitalize the first letter of each word so a raw db string ("test") shows as "Test"
  const displayName = rawName.replace(/\b\w/g, c => c.toUpperCase());
  const initial = displayName[0].toUpperCase();
  document.querySelectorAll(".user-avatar").forEach(el => el.textContent = initial);
  document.querySelectorAll(".user-name").forEach(el => el.textContent = displayName);

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
    btn.innerHTML = `${ICONS.stop}  Stop Session`;
    btn.className = "btn btn-stop btn-full";
  } else {
    btn.innerHTML = `${ICONS.play}  Start Session`;
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

  if (data.reps !== undefined && data.score !== undefined) {
    appendRepHistory(data.reps, data.score, data.feedback || "");
  }
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
    _logCourseStepIfActive(report.reps || 0, report.avg_score || 0);
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
// Courses — catalogue (client-side definitions; only progress goes to server)
// ---------------------------------------------------------------------------
const COURSES = [
  {
    id: "knee_rehab_beginner",
    title: "Knee Rehabilitation",
    icon: ICONS.knee_extension,
    category: "Knee",
    difficulty: "beginner",
    age_group: "all",
    duration_mins: 15,
    description: "Gentle knee rehabilitation for post-injury recovery. Rebuilds range of motion and quad strength progressively with low joint stress.",
    exercises: [
      { key: "knee_extension", label: "Knee Extension", reps: 8,  sets: 2 },
      { key: "hip_march",      label: "Hip March",      reps: 10, sets: 2 },
      { key: "sts",            label: "Sit to Stand",   reps: 5,  sets: 2 },
    ]
  },
  {
    id: "shoulder_recovery",
    title: "Shoulder Recovery",
    icon: ICONS.lateral_raise,
    category: "Shoulder",
    difficulty: "beginner",
    age_group: "45plus",
    duration_mins: 20,
    description: "Rotator cuff and scapular stabilization for shoulder impingement or post-surgery recovery. Low resistance, high focus on control.",
    exercises: [
      { key: "shoulder_scaption",  label: "Shoulder Scaption",  reps: 8,  sets: 2 },
      { key: "shoulder_extension", label: "Shoulder Extension", reps: 8,  sets: 2 },
      { key: "wall_pushup",        label: "Wall Push-Up",       reps: 10, sets: 2 },
      { key: "lateral_raise",      label: "Lateral Raise",      reps: 8,  sets: 2 },
    ]
  },
  {
    id: "senior_mobility",
    title: "Senior Mobility",
    icon: ICONS.person,
    category: "Senior",
    difficulty: "beginner",
    age_group: "60plus",
    duration_mins: 12,
    description: "Fall prevention and functional mobility for older adults. Targets lower limb strength and dynamic balance in a safe seated-to-standing progression.",
    exercises: [
      { key: "sts",            label: "Sit to Stand",   reps: 5, sets: 3 },
      { key: "hip_march",      label: "Hip March",      reps: 8, sets: 2 },
      { key: "knee_extension", label: "Knee Extension", reps: 6, sets: 2 },
    ]
  },
  {
    id: "post_surgery_recovery",
    title: "Post-Surgery Recovery",
    icon: ICONS.medical,
    category: "Knee",
    difficulty: "beginner",
    age_group: "all",
    duration_mins: 10,
    description: "Low-impact controlled movement protocol for patients recovering from lower extremity surgery. Prioritizes precision form over intensity.",
    exercises: [
      { key: "knee_extension", label: "Knee Extension", reps: 5, sets: 2 },
      { key: "hip_march",      label: "Hip March",      reps: 6, sets: 2 },
      { key: "sts",            label: "Sit to Stand",   reps: 4, sets: 2 },
    ]
  },
  {
    id: "upper_body_strength",
    title: "Upper Body Strength",
    icon: ICONS.pushup,
    category: "Upper Body",
    difficulty: "intermediate",
    age_group: "adult",
    duration_mins: 25,
    description: "Build chest, shoulder, and arm strength through progressive push and pull patterns. Demands consistent form across higher rep ranges.",
    exercises: [
      { key: "pushup",        label: "Push-up",       reps: 12, sets: 3 },
      { key: "curl",          label: "Bicep Curl",    reps: 10, sets: 3 },
      { key: "lateral_raise", label: "Lateral Raise", reps: 12, sets: 3 },
      { key: "wall_pushup",   label: "Wall Push-Up",  reps: 15, sets: 2 },
    ]
  },
  {
    id: "shoulder_upper_back",
    title: "Shoulder & Upper Back",
    icon: ICONS.shoulder_extension,
    category: "Shoulder",
    difficulty: "intermediate",
    age_group: "all",
    duration_mins: 20,
    description: "Targets posterior shoulder and scapular muscles. Counteracts desk posture and reduces chronic upper back tension through controlled eccentric work.",
    exercises: [
      { key: "shoulder_extension", label: "Shoulder Extension", reps: 10, sets: 3 },
      { key: "shoulder_scaption",  label: "Shoulder Scaption",  reps: 10, sets: 3 },
      { key: "wall_pushup",        label: "Wall Push-Up",       reps: 12, sets: 2 },
      { key: "lateral_raise",      label: "Lateral Raise",      reps: 10, sets: 2 },
    ]
  },
  {
    id: "full_body_conditioning",
    title: "Full Body Conditioning",
    icon: ICONS.curl,
    category: "Full Body",
    difficulty: "intermediate",
    age_group: "adult",
    duration_mins: 35,
    description: "Complete functional conditioning circuit targeting all major muscle groups. Great for general fitness maintenance and building exercise consistency.",
    exercises: [
      { key: "squat",         label: "Deep Squat",    reps: 10, sets: 3 },
      { key: "pushup",        label: "Push-up",       reps: 10, sets: 3 },
      { key: "curl",          label: "Bicep Curl",    reps: 12, sets: 2 },
      { key: "lateral_raise", label: "Lateral Raise", reps: 10, sets: 2 },
      { key: "hip_march",     label: "Hip March",     reps: 15, sets: 2 },
    ]
  },
  {
    id: "advanced_athletic",
    title: "Athletic Performance",
    icon: ICONS.trophy,
    category: "Full Body",
    difficulty: "advanced",
    age_group: "adult",
    duration_mins: 40,
    description: "High-intensity functional training demanding precision form at elevated rep counts. Built for athletes focused on movement quality under fatigue.",
    exercises: [
      { key: "squat",         label: "Deep Squat",    reps: 15, sets: 4 },
      { key: "pushup",        label: "Push-up",       reps: 15, sets: 4 },
      { key: "curl",          label: "Bicep Curl",    reps: 15, sets: 3 },
      { key: "lateral_raise", label: "Lateral Raise", reps: 15, sets: 3 },
    ]
  },
];

// { courseId: { completed_steps: Set<number>, step_scores: {index: score} } }
let _courseProgress = {};
let _activeCourseModal = null;

// ---------------------------------------------------------------------------
// Courses — server signal handler
// ---------------------------------------------------------------------------
function onCourseProgressLoaded(jsonStr) {
  const rows = JSON.parse(jsonStr);
  _courseProgress = {};
  if (!Array.isArray(rows)) return;
  rows.forEach(r => {
    if (!_courseProgress[r.course_id]) {
      _courseProgress[r.course_id] = { completed_steps: new Set(), step_scores: {} };
    }
    _courseProgress[r.course_id].completed_steps.add(r.step_index);
    _courseProgress[r.course_id].step_scores[r.step_index] = r.score;
  });
  if (ui.currentPage === "courses") renderCourses();
}

// ---------------------------------------------------------------------------
// Courses — render card grid
// ---------------------------------------------------------------------------
function renderCourses() {
  const diffFilter = document.querySelector("#course-diff-filters .courses-filter-pill.active")?.dataset.diff || "all";
  const ageFilter  = document.querySelector("#course-age-filters .courses-filter-pill.active")?.dataset.age  || "all";
  const grid = document.getElementById("courses-grid");
  if (!grid) return;

  const AGE_MAP = { all: "All Ages", adult: "18–60", "45plus": "45+", "60plus": "60+" };

  grid.innerHTML = "";

  const visible = COURSES.filter(c => {
    const diffOk = diffFilter === "all" || c.difficulty === diffFilter;
    const ageOk  = ageFilter  === "all" || c.age_group  === ageFilter || c.age_group === "all";
    return diffOk && ageOk;
  });

  if (!visible.length) {
    grid.innerHTML = `<div style="grid-column:1/-1;text-align:center;padding:48px 0;color:var(--text-muted);font-size:13px;">No courses match the selected filters.</div>`;
    return;
  }

  visible.forEach((course, i) => {
    const prog           = _courseProgress[course.id];
    const completedCount = prog ? prog.completed_steps.size : 0;
    const totalSteps     = course.exercises.length;
    const pct            = totalSteps > 0 ? Math.round((completedCount / totalSteps) * 100) : 0;
    const isComplete     = completedCount >= totalSteps && totalSteps > 0;
    const isStarted      = completedCount > 0 && !isComplete;

    const diffLabel = { beginner: "Beginner", intermediate: "Intermediate", advanced: "Advanced" }[course.difficulty];
    const ageLabel  = AGE_MAP[course.age_group] || "All Ages";

    let btnLabel = "View Course";
    let btnClass = "";
    if (isComplete)  { btnLabel = `${ICONS.check} Completed`; btnClass = "complete"; }
    else if (isStarted) { btnLabel = "Continue &rarr;"; btnClass = "primary"; }

    const card = document.createElement("div");
    card.className = "course-card";
    card.style.animationDelay = `${i * 50}ms`;
    card.innerHTML = `
      <div class="course-card-accent ${course.difficulty}"></div>
      <div class="course-card-body">
        <div class="course-card-top">
          <div class="course-card-emoji ${course.difficulty}">${course.icon}</div>
          <div class="course-card-title-wrap">
            <div class="course-card-title">${course.title}</div>
            <div class="course-card-category">${course.category}</div>
          </div>
        </div>
        <div class="course-card-desc">${course.description}</div>
        <div class="course-card-tags">
          <span class="course-tag ${course.difficulty}">${diffLabel}</span>
          <span class="course-tag">${ageLabel}</span>
          <span class="course-tag">${ICONS.clock} ${course.duration_mins} min</span>
          <span class="course-tag">${totalSteps} exercises</span>
        </div>
      </div>
      <div class="course-card-footer">
        <div class="course-card-progress-wrap">
          <div class="course-card-progress-label">${completedCount} / ${totalSteps} steps</div>
          <div class="course-card-progress-track">
            <div class="course-card-progress-fill" style="width:${pct}%"></div>
          </div>
        </div>
        <button class="course-card-btn ${btnClass}">${btnLabel}</button>
      </div>
    `;
    card.addEventListener("click", () => openCourseModal(course.id));
    grid.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Courses — detail modal
// ---------------------------------------------------------------------------
function openCourseModal(courseId) {
  const course = COURSES.find(c => c.id === courseId);
  if (!course) return;
  _activeCourseModal = courseId;

  const prog           = _courseProgress[courseId];
  const completedSteps = prog ? prog.completed_steps : new Set();
  const stepScores     = prog ? prog.step_scores     : {};
  const completedCount = completedSteps.size;
  const totalSteps     = course.exercises.length;
  const pct            = totalSteps > 0 ? Math.round((completedCount / totalSteps) * 100) : 0;

  const AGE_MAP   = { all: "All Ages", adult: "18–60", "45plus": "45+", "60plus": "60+" };
  const DIFF_MAP  = { beginner: "Beginner", intermediate: "Intermediate", advanced: "Advanced" };

  document.getElementById("cm-title").innerHTML = `<span class="course-modal-title-icon">${course.icon}</span>  ${course.title}`;
  document.getElementById("cm-tags").innerHTML = `
    <span class="course-tag ${course.difficulty}">${DIFF_MAP[course.difficulty]}</span>
    <span class="course-tag">${AGE_MAP[course.age_group] || "All Ages"}</span>
    <span class="course-tag">${ICONS.clock} ${course.duration_mins} min</span>
    <span class="course-tag">${totalSteps} exercises</span>
  `;
  document.getElementById("cm-desc").textContent = course.description;

  const stepList = document.getElementById("cm-steps");
  stepList.innerHTML = "";
  course.exercises.forEach((ex, idx) => {
    const done   = completedSteps.has(idx);
    const score  = stepScores[idx];
    // A step is unlocked if it's done OR if the previous step is done (or it's the first)
    const unlocked = done || idx === 0 || completedSteps.has(idx - 1);

    const stepIcon = exerciseIcon(ex.key);
    const item = document.createElement("div");
    item.className = `course-step-item${done ? " done" : ""}`;
    item.innerHTML = `
      <div class="course-step-num ${done ? "done" : ""}">${done ? ICONS.check : idx + 1}</div>
      <div class="course-step-icon">${stepIcon}</div>
      <div class="course-step-info">
        <div class="course-step-label">${ex.label}</div>
        <div class="course-step-meta">${ex.sets} × ${ex.reps} reps</div>
      </div>
      ${done
        ? `<div class="course-step-score">${score !== undefined ? score + "/100" : ICONS.check}</div>
           <button class="course-step-btn done-btn" disabled>Done</button>`
        : `<button class="course-step-btn" ${!unlocked ? 'disabled' : ''} onclick="startCourseStep('${courseId}',${idx})">Start</button>`
      }
    `;
    stepList.appendChild(item);
  });

  document.getElementById("cm-progress-label").textContent = `${completedCount} / ${totalSteps} steps completed`;
  document.getElementById("cm-progress-fill").style.width = pct + "%";

  const resetBtn = document.getElementById("cm-reset-btn");
  if (resetBtn) resetBtn.style.display = completedCount > 0 ? "" : "none";

  document.getElementById("course-modal-overlay").classList.add("open");
}

function closeCourseModal() {
  document.getElementById("course-modal-overlay")?.classList.remove("open");
  _activeCourseModal = null;
}

// ---------------------------------------------------------------------------
// Courses — start a step
// ---------------------------------------------------------------------------
function startCourseStep(courseId, stepIndex) {
  const course = COURSES.find(c => c.id === courseId);
  if (!course) return;
  const step = course.exercises[stepIndex];
  if (!step) return;

  ui.activeCourse = { courseId, stepIndex };
  closeCourseModal();
  launchExercise(step.key);
  toast(`Step ${stepIndex + 1}/${course.exercises.length}: ${step.label} — target ${step.reps} reps`, "info");
}

// ---------------------------------------------------------------------------
// Courses — reset progress
// ---------------------------------------------------------------------------
function resetActiveCourse() {
  if (!_activeCourseModal) return;
  const courseId = _activeCourseModal;
  delete _courseProgress[courseId];
  if (backend) backend.reset_course(courseId);
  openCourseModal(courseId);  // re-render modal with cleared state
  renderCourses();
  toast("Course progress reset.", "info");
}

// ---------------------------------------------------------------------------
// Courses — log step when session finishes
// ---------------------------------------------------------------------------
function _logCourseStepIfActive(reps, score) {
  if (!ui.activeCourse || !backend) return;
  const { courseId, stepIndex } = ui.activeCourse;
  ui.activeCourse = null;

  if (!_courseProgress[courseId]) {
    _courseProgress[courseId] = { completed_steps: new Set(), step_scores: {} };
  }
  _courseProgress[courseId].completed_steps.add(stepIndex);
  _courseProgress[courseId].step_scores[stepIndex] = score;
  renderCourses();

  backend.log_course_step(courseId, stepIndex, JSON.stringify(reps), JSON.stringify(score));

  const course = COURSES.find(c => c.id === courseId);
  if (course) {
    const allDone = _courseProgress[courseId].completed_steps.size >= course.exercises.length;
    if (allDone) {
      setTimeout(() => toast(`Course "${course.title}" complete!`, "success"), 600);
    } else {
      const next = stepIndex + 1;
      if (next < course.exercises.length) {
        setTimeout(() => toast(`Step ${stepIndex + 1} done! Next: ${course.exercises[next].label}`, "success"), 400);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Render active goals with progress bars
// ---------------------------------------------------------------------------
const GOAL_TYPE_LABELS = { reps: "Total Reps", score: "Avg Score", sessions: "Sessions" };
const EXERCISE_ICONS   = {
  any: ICONS.hip_march, squat: ICONS.squat, sts: ICONS.sts, pushup: ICONS.pushup,
  curl: ICONS.curl, lateral_raise: ICONS.lateral_raise,
  knee_extension: ICONS.knee_extension, wall_pushup: ICONS.wall_pushup,
  hip_march: ICONS.hip_march, shoulder_extension: ICONS.shoulder_extension,
  shoulder_scaption: ICONS.shoulder_scaption,
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
    const icon      = EXERCISE_ICONS[g.exercise] || ICONS.hip_march;
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
    const lockIcon = a.unlocked ? "" : `<div class="achievement-icon">${ICONS.lock}</div>`;
    const svgIcon       = ACHIEVEMENT_ICONS[a.key] || ICONS.star;
    const realIcon      = a.unlocked ? `<div class="achievement-icon">${svgIcon}</div>` : "";

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
// Groq AI Feedback Settings
// ---------------------------------------------------------------------------
function onGroqStatusChanged(status) {
  const dot  = document.getElementById("groq-status-dot");
  const text = document.getElementById("groq-status-text");
  if (!dot || !text) return;
  const map = {
    active:       { cls: "status-dot--active",  label: "Active" },
    rate_limited: { cls: "status-dot--limited", label: "Rate Limited — please use your own key" },
    invalid_key:  { cls: "status-dot--error",   label: "Invalid Key" },
    error:        { cls: "status-dot--error",   label: "Unreachable" },
  };
  const info = map[status] || { cls: "status-dot--error", label: status };
  dot.className    = "status-dot " + info.cls;
  text.textContent = info.label;
}

function onGroqKeyInfo(usingUserKey) {
  const src      = document.getElementById("groq-key-source");
  const clearBtn = document.getElementById("groq-clear-btn");
  if (src)      src.textContent    = usingUserKey ? "Your personal key" : "Project default";
  if (clearBtn) clearBtn.style.display = usingUserKey ? "inline-flex" : "none";
}

function checkGroqStatus() {
  if (backend) backend.check_groq_status();
}

function saveGroqKey() {
  const input = document.getElementById("groq-key-input");
  if (!input || !input.value.trim()) return;
  if (!input.value.trim().startsWith("gsk_")) {
    toast("Key must start with gsk_", "error");
    return;
  }
  if (backend) {
    backend.save_groq_key(input.value.trim());
    input.value = "";
    toast("API key saved locally.", "success");
  }
}

function clearGroqKey() {
  if (backend) {
    backend.clear_groq_key();
    toast("Reverted to project default key.", "info");
  }
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
        <div class="empty-icon">${ICONS.chart}</div>
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
  ["metric-reps", "metric-score"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = "—";
  });
  updateScoreRing(0);
  const fb = document.getElementById("feedback-chip");
  if (fb) { fb.textContent = "Awaiting session…"; fb.className = "feedback-chip"; }
  updateStatusPill("OFFLINE", "");
  clearRepHistory();
}

// ---------------------------------------------------------------------------
// Live Rep History (analysis sidebar)
// ---------------------------------------------------------------------------
let _lastRepCount = 0;

function appendRepHistory(repNum, score, feedback) {
  if (repNum <= _lastRepCount) return;
  _lastRepCount = repNum;

  const list = document.getElementById("rep-history-list");
  if (!list) return;

  const empty = list.querySelector(".rep-history-empty");
  if (empty) empty.remove();

  const cls = score >= 80 ? "good" : score >= 60 ? "mid" : "low";
  const label = feedback && feedback !== "Excellent Form" ? feedback : "Good Form";

  const row = document.createElement("div");
  row.className = "rep-history-row";
  row.innerHTML = `
    <span class="rep-history-num">#${repNum}</span>
    <span class="rep-history-score ${cls}">${score}</span>
    <span class="rep-history-feedback">${label}</span>
  `;
  list.prepend(row);
}

function clearRepHistory() {
  _lastRepCount = 0;
  const list = document.getElementById("rep-history-list");
  if (!list) return;
  list.innerHTML = `<div class="rep-history-empty">Reps will appear here as you go.</div>`;
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

  function _restoreTemplate() {
    tmpl.style.display    = "none";
    tmpl.style.position   = "";
    tmpl.style.left       = "";
    tmpl.style.top        = "";
    tmpl.style.visibility = "";
  }

  // ── 5. Render PDF as blob, then hand off to Python for the save dialog ───
  html2pdf()
    .set({
      margin:   [15, 15, 15, 15],
      filename: filename,
      image:    { type: "jpeg", quality: 0.98 },
      html2canvas: { scale: 2, useCORS: true, backgroundColor: "#FFFFFF" },
      jsPDF: { unit: "mm", format: "a4", orientation: "portrait" },
    })
    .from(pageEl)
    .output("blob")
    .then((blob) => {
      _restoreTemplate();
      if (!backend) { toast("Backend not connected.", "error"); return; }
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(",")[1];
        backend.save_pdf(base64, filename, (resultStr) => {
          const result = JSON.parse(resultStr);
          if (result.cancelled) return;
          if (result.ok) toast("PDF saved successfully.", "success");
          else toast("PDF save failed.", "error");
        });
      };
      reader.readAsDataURL(blob);
    })
    .catch((err) => {
      _restoreTemplate();
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
  _logCourseStepIfActive(report.reps || 0, report.avg_score || 0);
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
  // Sidebar collapse toggle — persists across sessions via localStorage
  const sidebar = document.getElementById("sidebar");
  const toggleBtn = document.getElementById("sidebar-toggle");
  if (sidebar && toggleBtn) {
    if (localStorage.getItem("sidebarCollapsed") === "1") sidebar.classList.add("collapsed");
    toggleBtn.addEventListener("click", () => {
      sidebar.classList.toggle("collapsed");
      localStorage.setItem("sidebarCollapsed", sidebar.classList.contains("collapsed") ? "1" : "0");
    });
  }

  // Navigation buttons
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => {
      navigate(btn.dataset.page);
      if (btn.dataset.page === "settings" && backend) {
        backend.check_groq_status();
      }
    });
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

  // Render courses grid when opening courses page
  document.querySelector(".nav-item[data-page='courses']")
    ?.addEventListener("click", () => renderCourses());

  // Course filter pills
  document.querySelectorAll("#course-diff-filters .courses-filter-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll("#course-diff-filters .courses-filter-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      renderCourses();
    });
  });
  document.querySelectorAll("#course-age-filters .courses-filter-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll("#course-age-filters .courses-filter-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      renderCourses();
    });
  });

  // Course modal — close on overlay background click
  document.getElementById("course-modal-overlay")?.addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeCourseModal();
  });

  // Goal modal — close on overlay background click
  document.getElementById("goal-modal-overlay")?.addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeGoalDialog();
  });


  // Exercise cards — populate SVG icons and attach click handlers
  document.querySelectorAll(".exercise-card[data-key]").forEach(card => {
    const ex = EXERCISES.find(e => e.key === card.dataset.key);
    if (ex) {
      const iconEl = card.querySelector(".exercise-icon");
      if (iconEl) iconEl.innerHTML = ex.icon;
    }
    card.addEventListener("click", () => launchExercise(card.dataset.key));
  });

  // Populate initial session button icon
  const sessionBtn = document.getElementById("session-btn");
  if (sessionBtn) sessionBtn.innerHTML = `${ICONS.play}  Start Session`;

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