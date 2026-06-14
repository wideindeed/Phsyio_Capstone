"""
goals.py — Goal & Achievement Engine for Physio-Vision
=======================================================

Standalone client-side module.  Zero changes to engine.py.

Integration into dashboard.py (Bridge) is minimal:
  1. `from goals import GoalTracker` at the top of dashboard.py
  2. One line in Bridge.__init__ to instantiate
  3. Two new Signal declarations on Bridge
  4. Four new @Slot methods on Bridge

All network calls run on daemon threads and callback into the Qt
main thread via Bridge._emit_safe() — no Qt imports needed here.
"""

import json
import threading
import requests
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Achievement catalogue — single source of truth on the client side.
# The server holds an identical copy; these must stay in sync.
# ---------------------------------------------------------------------------
ACHIEVEMENT_DEFINITIONS: dict[str, dict] = {
    "first_rep": {
        "title": "First Rep",
        "desc":  "Completed your very first session.",
        "icon":  "🏁",
        "tier":  "bronze",
    },
    "ten_sessions": {
        "title": "Consistent",
        "desc":  "Completed 10 sessions.",
        "icon":  "🔟",
        "tier":  "bronze",
    },
    "fifty_sessions": {
        "title": "Dedicated",
        "desc":  "Completed 50 sessions.",
        "icon":  "🏅",
        "tier":  "silver",
    },
    "hundred_sessions": {
        "title": "Centurion",
        "desc":  "Completed 100 sessions.",
        "icon":  "💯",
        "tier":  "gold",
    },
    "perfect_score": {
        "title": "Flawless",
        "desc":  "Achieved a perfect form score of 100 on a rep.",
        "icon":  "⭐",
        "tier":  "gold",
    },
    "high_scorer": {
        "title": "High Performer",
        "desc":  "Averaged 90+ form score across 10 or more sessions.",
        "icon":  "📈",
        "tier":  "gold",
    },
    "all_rounder": {
        "title": "All-Rounder",
        "desc":  "Tried all 5 exercises at least once.",
        "icon":  "🎯",
        "tier":  "silver",
    },
    "pain_warrior": {
        "title": "Pain Warrior",
        "desc":  "Completed a session and reported pain level 7 or above.",
        "icon":  "🛡️",
        "tier":  "bronze",
    },
    "hundred_reps": {
        "title": "The Century",
        "desc":  "Accumulated 100 total reps across all sessions.",
        "icon":  "💪",
        "tier":  "silver",
    },
    "five_hundred_reps": {
        "title": "Iron Will",
        "desc":  "Accumulated 500 total reps across all sessions.",
        "icon":  "🦾",
        "tier":  "gold",
    },
    "comeback_kid": {
        "title": "Comeback Kid",
        "desc":  "Returned to training after a 7+ day absence.",
        "icon":  "🔁",
        "tier":  "bronze",
    },
    "streak_7": {
        "title": "7-Day Streak",
        "desc":  "Exercised on 7 consecutive calendar days.",
        "icon":  "🔥",
        "tier":  "silver",
    },
}

# Human-readable labels for the UI
GOAL_TYPE_LABELS = {
    "reps":     "Total Reps",
    "score":    "Avg Score",
    "sessions": "Sessions",
}

EXERCISE_LABELS = {
    "any":          "Any Exercise",
    "squat":        "Deep Squat",
    "sts":          "Sit to Stand",
    "pushup":       "Push-up",
    "curl":         "Bicep Curl",
    "lateral_raise":"Lateral Raise",
}


# ---------------------------------------------------------------------------
# GoalTracker — wraps all API calls related to goals and achievements.
# ---------------------------------------------------------------------------
class GoalTracker:
    """
    Manages cloud-persisted goals and achievements via the Physio-Vision API.

    All network operations run on daemon threads to keep the Qt main thread
    responsive.  Callbacks are invoked from those background threads; the
    caller (Bridge) is responsible for trampolining back to the main thread
    via _emit_safe() before touching any Qt signals or widgets.
    """

    def __init__(self, api_url: str, token: str) -> None:
        self._api_url = api_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "X-Desktop-Key": "my_secret_desktop_key_2026",
            "Content-Type":  "application/json",
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get(self, endpoint: str, callback: Callable) -> None:
        def _worker():
            try:
                resp = requests.get(
                    f"{self._api_url}{endpoint}",
                    headers=self._headers, timeout=10
                )
                callback(resp.json() if resp.ok else None)
            except Exception as e:
                print(f"[GoalTracker] GET {endpoint} failed: {e}")
                callback(None)
        threading.Thread(target=_worker, daemon=True).start()

    def _post(self, endpoint: str, payload: dict,
              callback: Optional[Callable] = None) -> None:
        def _worker():
            try:
                resp = requests.post(
                    f"{self._api_url}{endpoint}",
                    data=json.dumps(payload),
                    headers=self._headers, timeout=10
                )
                if callback:
                    callback(resp.json() if resp.ok else None)
            except Exception as e:
                print(f"[GoalTracker] POST {endpoint} failed: {e}")
                if callback:
                    callback(None)
        threading.Thread(target=_worker, daemon=True).start()

    def _delete(self, endpoint: str,
                callback: Optional[Callable] = None) -> None:
        def _worker():
            try:
                resp = requests.delete(
                    f"{self._api_url}{endpoint}",
                    headers=self._headers, timeout=10
                )
                if callback:
                    callback(resp.ok)
            except Exception as e:
                print(f"[GoalTracker] DELETE {endpoint} failed: {e}")
                if callback:
                    callback(False)
        threading.Thread(target=_worker, daemon=True).start()

    # ── Public API ────────────────────────────────────────────────────────

    def fetch_goals(self, callback: Callable) -> None:
        """Fetch active goals with computed progress.
        Calls callback(list[dict] | None)."""
        self._get("/get_goals",
                  lambda d: callback(d.get("goals") if d else None))

    def fetch_achievements(self, callback: Callable) -> None:
        """Fetch all achievements with locked/unlocked status.
        Calls callback(list[dict] | None)."""
        self._get("/get_achievements",
                  lambda d: callback(d.get("achievements") if d else None))

    def create_goal(self, exercise: str, goal_type: str,
                    target_value: float, deadline: Optional[str],
                    callback: Optional[Callable] = None) -> None:
        """
        Create a new goal on the server.

        Args:
            exercise:     'any' | 'squat' | 'sts' | 'pushup' | 'curl' | 'lateral_raise'
            goal_type:    'reps' | 'score' | 'sessions'
            target_value: numeric target (reps count, score out of 100, session count)
            deadline:     ISO-8601 date string (e.g. '2026-12-31') or None
            callback:     optional callable(dict | None)
        """
        self._post("/set_goal", {
            "exercise":     exercise,
            "goal_type":    goal_type,
            "target_value": float(target_value),
            "deadline":     deadline,
        }, callback)

    def delete_goal(self, goal_id: int,
                    callback: Optional[Callable] = None) -> None:
        """Delete a goal by ID.  Only the owning user can delete their goals."""
        self._delete(f"/delete_goal/{goal_id}", callback)
