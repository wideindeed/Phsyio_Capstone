# =============================================================================
#  llm_coach.py — LLM Coaching Layer for Physio-Vision
#
#  Translates raw kinematic data into dynamic, spoken coaching cues
#  using the Groq API (free tier, llama-3.1-8b-instant).
#
#  If GROQ_API_KEY is not set or the API is unreachable, the module
#  silently disables itself and the app falls back to rule-based cues.
#
#  Depends on: groq (pip install groq)
# =============================================================================

import os
import time
import threading

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    print("[LLM Coach] groq package not installed. LLM coaching disabled.")


# =============================================================================
#  EXERCISE PROFILES  (add new exercises here to scale LLM coaching)
# =============================================================================

EXERCISE_PROFILES = {
    "deep_squat": {
        "name": "Deep Squat",
        "joints_monitored": "hip, knee, ankle (sagittal plane)",
        "common_faults": [
            "forward trunk lean",
            "back rounding / thoracic kyphosis",
            "knee valgus",
            "insufficient depth",
            "heel rise",
        ],
        "ideal_depth_angle": "below 120° knee flexion",
    },
    # Future exercises can be added here:
    # "pushup": { ... },
    # "bicep_curl": { ... },
}


# =============================================================================
#  SYSTEM PROMPT  (Coach persona + strict output rules)
# =============================================================================

SYSTEM_PROMPT = """\
You are Coach Vance, an elite sports physiotherapist coaching a live exercise session.

RULES — follow these exactly:
1. Output ONE coaching cue only. No greetings, no questions, no lists.
2. Maximum 18 words. Be punchy and commanding.
3. Reference the actual numbers you are given (angles, scores, thresholds).
4. If form is perfect (score >= 90, no faults), give brief positive reinforcement.
5. If you detect a declining trend in session_trend scores, call out fatigue explicitly.
6. Never say "try to" or "maybe". Command directly: "Drive", "Lock", "Brace", etc.
7. Speak as if the athlete can hear you RIGHT NOW mid-set. Present tense only.

GOOD examples:
- "Chest up — you're leaning 48 degrees past vertical, brace that core."
- "Three reps straight declining — fatigue setting in, focus on depth."
- "Clean rep, 94% form. Lock it in."

BAD examples (never do these):
- "Great job! Keep it up!" (too generic, no data)
- "You should try to keep your back straight." (weak language, no numbers)
- "Here are some tips for your squat: 1. ..." (list format, not a live cue)
"""


# =============================================================================
#  SNAPSHOT BUILDER  (translates kinematics → LLM-readable dict)
# =============================================================================

def build_rep_snapshot(
    exercise_type: str,
    rep_number: int,
    joint_angle_at_peak: float,
    form_score_percent: int,
    faults_detected: list,
    raw_metrics: dict,
    session_log: list,
    user_height_cm: float,
) -> dict:
    """Build the structured data packet sent to the LLM after each rep.

    Args:
        exercise_type:       Key into EXERCISE_PROFILES (e.g. "deep_squat").
        rep_number:          Which rep this is (1-indexed).
        joint_angle_at_peak: Joint angle at the deepest/peak point of the rep.
        form_score_percent:  0-100 form quality score.
        faults_detected:     List of fault strings (empty if perfect).
        raw_metrics:         Dict of lean/rounding angles and thresholds.
        session_log:         Full session history (list of rep dicts).
        user_height_cm:      User's height for biomechanical context.

    Returns:
        A dict ready to be sent as the LLM user prompt.
    """
    # Extract last 3 scores for fatigue trend detection
    recent_scores = [entry["score"] for entry in session_log[-3:]]

    profile = EXERCISE_PROFILES.get(exercise_type, {})

    return {
        "exercise": profile.get("name", exercise_type),
        "joints_monitored": profile.get("joints_monitored", "unknown"),
        "common_faults_reference": profile.get("common_faults", []),
        "rep_number": rep_number,
        "joint_angle_at_peak_degrees": round(joint_angle_at_peak, 1),
        "form_score_percent": form_score_percent,
        "faults_detected": faults_detected if faults_detected else ["none"],
        "raw_metrics": raw_metrics,
        "session_trend_last_3_scores": recent_scores,
        "user_height_cm": user_height_cm,
    }


# =============================================================================
#  LLM COACH CLASS
# =============================================================================

class LLMCoach:
    """Async LLM coaching interface with cooldown and graceful fallback.

    Usage:
        coach = LLMCoach()            # reads GROQ_API_KEY from env
        if coach.enabled:
            snapshot = build_rep_snapshot(...)
            coach.cue_async(snapshot, callback=speak_async)
    """

    def __init__(self):
        self.enabled: bool = False
        self.model: str = "llama-3.1-8b-instant"
        self._last_call_time: float = 0.0
        self._cooldown_seconds: float = 8.0   # Min gap between LLM calls
        self._client = None

        if not _GROQ_AVAILABLE:
            return

        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            print("[LLM Coach] GROQ_API_KEY not set. LLM coaching disabled.")
            return

        try:
            self._client = Groq(api_key=api_key)
            self.enabled = True
            print(f"[LLM Coach] Enabled — model: {self.model}")
        except Exception as e:
            print(f"[LLM Coach] Failed to initialise Groq client: {e}")

    def cue_async(self, snapshot: dict, callback=None) -> None:
        """Fire-and-forget: calls the LLM in a background thread.

        This method returns immediately and never blocks the caller.
        The actual HTTP request to Groq runs on a daemon thread.

        Args:
            snapshot: The rep data dict from build_rep_snapshot().
            callback: A callable(str) invoked with the coaching cue text
                      when the LLM responds.  Typically ``speak_async``
                      so the cue is spoken aloud via TTS.
        """
        if not self.enabled:
            return

        now = time.time()
        if now - self._last_call_time < self._cooldown_seconds:
            return   # Still in cooldown — skip this rep's LLM call
        self._last_call_time = now

        def _worker():
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": str(snapshot)},
                    ],
                    temperature=0.7,
                    max_tokens=60,
                )
                cue = response.choices[0].message.content.strip()
                if cue and callback:
                    callback(cue)
            except Exception as e:
                print(f"[LLM Coach] API error (non-fatal): {e}")

        threading.Thread(target=_worker, daemon=True).start()
