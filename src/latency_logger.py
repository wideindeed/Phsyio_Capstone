"""
Stage-level latency instrumentation for the measured-latency task.

Wraps the four pipeline stages that matter for an end-to-end latency budget:
MediaPipe pose extraction, BiLSTM inference, the Groq feedback round-trip
(network + LLM), and TTS synthesis. Every call appends one row to
latency_log.csv at the repo root so a real session produces real numbers
instead of estimates.
"""

import csv
import os
import time
import threading

from groq_feedback import get_rep_feedback

_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "latency_log.csv")
_lock = threading.Lock()


def log_latency(stage: str, ms: float) -> None:
    with _lock:
        write_header = not os.path.exists(_LOG_PATH)
        with open(_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "stage", "ms"])
            w.writerow([time.time(), stage, f"{ms:.3f}"])


class timed:
    """Context manager: `with timed('mediapipe_pose'): ...` logs elapsed ms."""

    def __init__(self, stage: str):
        self.stage = stage

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        log_latency(self.stage, (time.perf_counter() - self._t0) * 1000)
        return False


def timed_get_rep_feedback(exercise: str, rep_num: int, score: int,
                            issues: list, callback) -> None:
    """Drop-in wrapper around get_rep_feedback() that logs the Groq
    round-trip latency (call -> callback fired), without changing behavior."""
    t0 = time.perf_counter()

    def _wrapped_callback(text):
        log_latency(f"groq_feedback::{exercise}", (time.perf_counter() - t0) * 1000)
        callback(text)

    get_rep_feedback(exercise=exercise, rep_num=rep_num, score=score,
                      issues=issues, callback=_wrapped_callback)
