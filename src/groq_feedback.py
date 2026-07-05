import threading
import groq
from config import GROQ_API_KEY
from feedback_knowledge_base import get_grounding_context

# ── Key management ─────────────────────────────────────────────────
# _user_key: set at runtime from Settings page (takes priority)
# _default_key: baked in from config.py (project-wide fallback)

_default_key: str        = GROQ_API_KEY
_user_key:    str | None = None
_client:      groq.Groq  = groq.Groq(api_key=_default_key)


def set_user_key(key: str | None) -> None:
    global _user_key, _client
    _user_key = key.strip() if key and key.strip() else None
    _client   = groq.Groq(api_key=_user_key if _user_key else _default_key)


def using_user_key() -> bool:
    return bool(_user_key)


def check_api_status(callback) -> None:
    def _worker():
        try:
            _client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=3,
            )
            callback("active")
        except groq.RateLimitError:
            callback("rate_limited")
        except groq.AuthenticationError:
            callback("invalid_key")
        except Exception:
            callback("error")
    threading.Thread(target=_worker, daemon=True).start()


def get_rep_feedback(exercise: str, rep_num: int,
                     score: int, issues: list,
                     callback) -> None:
    def _worker():
        try:
            issues_str = ', '.join(issues) if issues else 'none detected'
            grounding = get_grounding_context(exercise, issues)
            grounding_block = (
                f"Clinical grounding for the detected issue(s) (base your "
                f"correction cue on this, don't quote it verbatim):\n{grounding}\n\n"
                if grounding else ""
            )
            prompt = (
                f"You are a physiotherapy assistant giving live spoken feedback "
                f"after one rep of an exercise. Be brief, direct, and encouraging.\n"
                f"Exercise: {exercise}\n"
                f"Rep number: {rep_num}\n"
                f"Form score: {score}/100\n"
                f"Detected issues: {issues_str}\n\n"
                f"{grounding_block}"
                f"Rules:\n"
                f"- Exactly ONE sentence, maximum 20 words\n"
                f"- No markdown, no bullet points\n"
                f"- Speak directly to the patient\n"
                f"- If score >= 80, be positive\n"
                f"- If score < 80, name the issue and give one correction cue"
            )
            resp = _client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.7,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Groq] fallback — {e}")
            if score >= 80:
                text = "Excellent form, keep it up."
            elif score >= 60:
                text = "Good effort. Focus on your form on the next rep."
            else:
                text = "Compensatory motion detected. Slow down and reset."
        callback(text)

    threading.Thread(target=_worker, daemon=True).start()
