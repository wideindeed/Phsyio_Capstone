import numpy as np
import cv2
from latency_logger import timed, timed_get_rep_feedback as get_rep_feedback

# ── Constants ──────────────────────────────────────────────────────────────────
TIME_STEPS   = 69
MODEL_GLOBAL = None     # Set by engine.py after loading

# Raw sigmoid range from training — calibrate after testing:
RAW_MIN = 0.55
RAW_MAX = 0.95

HOLD_SECONDS   = 0.6    # seconds to hold start position before recording
TIMEOUT_FRAMES = 200    # max frames per rep before auto-reset

# Hip flexion angle thresholds for standing hip march
# Angle measured at hip (shoulder-hip-knee)
DOWN_THRESHOLD = 155.0   # legs relaxed in standing position
UP_THRESHOLD   = 120.0   # knee lifted (hip flexed)


def set_model(m):
    """Called once by engine.py after load_model() succeeds."""
    global MODEL_GLOBAL
    MODEL_GLOBAL = m


def normalize_live(frames_list: list) -> np.ndarray:
    """
    Variable-length buffer → (1, TIME_STEPS, 66) model input.
    Pelvis Anchor Normalization — matches train.py exactly.
    """
    raw    = np.array(frames_list, dtype=np.float32)
    warped = cv2.resize(raw, (66, TIME_STEPS), interpolation=cv2.INTER_LINEAR)
    data   = warped.reshape(1, TIME_STEPS, 22, 3)

    root        = data[:, :, 0:1, :]
    data        = data - root
    left_hip    = data[:, :, 18:19, :]
    right_hip   = data[:, :, 14:15, :]
    pelvis_w    = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    data        = data / np.maximum(pelvis_w, 0.0001)

    return data.reshape(1, TIME_STEPS, 66)


def raw_to_score(pred: float) -> int:
    return int(max(0, min(100, ((pred - RAW_MIN) / (RAW_MAX - RAW_MIN)) * 100)))


class HipMarchAnalyzer:
    """
    State machine for Hip March (standing alternating knee lifts).
    Instantiated once by VisionWorker. reset() is called on each new session.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.stage       = "WAITING"
        self.buffer      = []
        self.frame_count = 0
        self.hit_top     = False
        self.hold_start  = 0.0

    def process(self, landmarks_3d, worker) -> None:
        import time
        from engine import (calculate_angle_3d, extract_prmd_features,
                            speak_async)

        # Hip flexion angle on each side (shoulder-hip-knee)
        left_hip_angle  = calculate_angle_3d(landmarks_3d[11], landmarks_3d[23], landmarks_3d[25])
        right_hip_angle = calculate_angle_3d(landmarks_3d[12], landmarks_3d[24], landmarks_3d[26])
        avg_angle       = (left_hip_angle + right_hip_angle) / 2.0
        min_angle       = min(left_hip_angle, right_hip_angle)  # captures the lifted knee

        is_down   = avg_angle > DOWN_THRESHOLD   # both legs relaxed, standing
        is_raised = min_angle < UP_THRESHOLD     # at least one knee is lifted

        if self.stage == "WAITING":
            if is_down:
                self.stage      = "HOLDING"
                self.hold_start = time.time()
                worker.system_status.emit("HOLD STANDING POSITION...", "#ffaa00")

        elif self.stage == "HOLDING":
            if not is_down:
                self.stage = "WAITING"
                worker.system_status.emit("RETURN TO STANDING POSITION", "#ffaa00")
            elif time.time() - self.hold_start > HOLD_SECONDS:
                self.stage       = "RECORDING"
                self.buffer      = []
                self.frame_count = 0
                self.hit_top     = False
                worker.system_status.emit("● RECORDING (MARCH)", "#ff4444")
                speak_async("Begin.")

        elif self.stage == "RECORDING":
            self.buffer.append(extract_prmd_features(landmarks_3d, mirror_x=True))
            self.frame_count += 1

            if is_raised:
                self.hit_top = True

            if self.hit_top and is_down:
                self.stage = "INFERENCE"
                worker.system_status.emit("ANALYZING...", "#0099ff")

            if self.frame_count > TIMEOUT_FRAMES:
                self.stage       = "WAITING"
                self.frame_count = 0
                worker.system_status.emit("TIMEOUT — RETURN TO START", "#ffaa00")

        elif self.stage == "INFERENCE":
            try:
                if len(self.buffer) >= 5 and MODEL_GLOBAL:
                    normalized  = normalize_live(self.buffer)
                    with timed("bilstm::Hip March"):
                        pred    = float(MODEL_GLOBAL.predict(normalized, verbose=0)[0][0])
                    score       = raw_to_score(pred)
                    feedback    = "Excellent Form" if score >= 80 else "Compensatory Motion Detected"

                    worker.reps += 1
                    log_entry    = {"rep_num": worker.reps, "score": score, "issue": feedback}
                    worker.session_log.append(log_entry)
                    worker.stats_update.emit({"reps": worker.reps, "score": score, "feedback": feedback})

                    get_rep_feedback(
                        exercise="Hip March",
                        rep_num=worker.reps,
                        score=score,
                        issues=[log_entry["issue"]] if log_entry.get("issue") and log_entry["issue"] != "Excellent Form" else [],
                        callback=lambda text: speak_async(text),
                    )

            except Exception as e:
                print(f"[HipMarch] Inference error: {e}")

            self.stage = "WAITING"
            worker.system_status.emit("RETURN TO STANDING POSITION", "#ffaa00")
