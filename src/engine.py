import sys
import cv2
import time
import threading
import numpy as np
import pyttsx3
import math
import os
from datetime import datetime
from keras.models import load_model


def resource_path(relative: str) -> str:
    """Resolve a path that works both in PyCharm and in the .exe."""
    if getattr(sys, 'frozen', False):
        base = os.path.join(sys._MEIPASS, 'src')
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative)


_MODEL_DIR = (
    os.path.join(sys._MEIPASS, "models")
    if getattr(sys, "frozen", False)
    else os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    )
)
# ─────────────────────────────────────────────────────────────────────────────


# --- Environment flags must be set BEFORE any AI/GPU imports ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

from PyQt5.QtCore import QThread, pyqtSignal as Signal
from PyQt5.QtGui import QImage

from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

from knee_extension_analyzer      import KneeExtensionAnalyzer,      set_model as set_knee_model
from wall_pushup_analyzer         import WallPushupAnalyzer,         set_model as set_wall_model
from hip_march_analyzer           import HipMarchAnalyzer,           set_model as set_hip_model
from shoulder_extension_analyzer  import ShoulderExtensionAnalyzer,  set_model as set_shoulder_ext_model
from shoulder_scaption_analyzer   import ShoulderScaptionAnalyzer,   set_model as set_shoulder_sca_model

from groq_feedback import get_rep_feedback
import groq_feedback as _groq_fb
from latency_logger import timed, timed_get_rep_feedback as get_rep_feedback


# =============================================================================
#  GLOBAL STATE & TUNING PARAMETERS
# =============================================================================

class AppState:
    """Single source of truth for all runtime parameters.
    Both the GUI and the engine read/write this shared object."""

    # --- User Biometrics ---
    USER_HEIGHT_CM: float = 174.0
    USER_WEIGHT_KG: float = 65.0

    # --- System Toggles ---
    VOICE_ON: bool = True
    AR_MODE: bool = False

    # --- File Paths ---
    GUIDE_PATH: str = resource_path(
        os.path.join('..', 'assets', 'Video_Generation_Person_Squatting.mp4')
    )

    # --- Squat Analysis Thresholds ---
    PARAM_SQUAT_DEPTH: float = 140.0  # Knee angle that counts as "down"
    PARAM_UP_THRESHOLD: float = 160.0  # Knee angle that counts as "standing"
    PARAM_LEAN_WARN: float = 40.0  # Trunk lean degrees → "Chest Up" warning
    PARAM_LEAN_CRIT: float = 55.0  # Trunk lean degrees → critical alert
    PARAM_ROUNDING: float = 18.0  # Max back curvature degrees allowed
    PARAM_PUSHUP_UP_ANGLE: float = 145.0  # Elbow angle that counts as "up"
    PARAM_PUSHUP_DOWN_ANGLE: float = 105.0  # Elbow angle that counts as "down"
    PARAM_PUSHUP_TIMEOUT_FRAMES: int = 300  # Max frames allowed for one rep attempt
    PARAM_PUSHUP_HIP_DEV_METERS: float = 0.12  # Max hip deviation from body line before warning
    PARAM_PUSHUP_HIP_DEV_RATIO: float = 0.20  # Max relative hip deviation vs body length
    PARAM_HEAD_ANGLE: float = 65.0  # Max head-to-torso angle before "Head Down" warning
    PARAM_CURL_DOWN_ANGLE: float = 150.0  # Arm fully extended
    PARAM_CURL_UP_ANGLE: float = 75.0  # Arm fully curled

    # --- Lateral Raise Analysis Thresholds ---
    PARAM_LATERAL_RAISE_DOWN_ANGLE: float = 30.0  # Shoulder angle when arms rest at sides
    PARAM_LATERAL_RAISE_ASYMMETRY_TOL: float = 18.0  # Max L/R peak-angle difference (degrees)
    PARAM_LATERAL_RAISE_SHRUG_RATIO: float = 0.65  # Shrug if ear-shoulder dist drops below this × baseline

    # --- Camera & Capture ---
    CAMERA_INDEX: int = 0
    MP_DETECTION_CONFIDENCE: float = 0.5
    MP_TRACKING_CONFIDENCE: float = 0.5
    MIRROR_VIDEO: bool = True

    # --- Session & Notifications ---
    PAIN_PROMPT_ENABLED: bool = True
    SESSION_TIMEOUT_MINS: int = 0
    DEFAULT_REP_TARGET: int = 0

    # --- Session History (in-memory, not persisted) ---
    HISTORY: list = []


# Singleton instance shared across the entire application
state = AppState()


# =============================================================================
#  PART 1: HOLOGRAPHIC AR ENGINE
# =============================================================================

class HologramProjector:
    """Draws the animated floor-target AR overlay onto a camera frame.
    Call draw() each frame; it returns True when the subject is in-zone."""

    def __init__(self):
        self.spin_angle_1: float = 0
        self.spin_angle_2: float = 0
        self.pulse_val: float = 0
        self.pulse_dir: int = 1
        self.lock_anim: float = 0.0

    def draw(self, frame, landmarks, width: int, height: int) -> bool:
        # Advance animations
        self.spin_angle_1 = (self.spin_angle_1 + 2) % 360
        self.spin_angle_2 = (self.spin_angle_2 - 3) % 360
        self.pulse_val += 0.05 * self.pulse_dir
        if self.pulse_val > 1.0 or self.pulse_val < 0.0:
            self.pulse_dir *= -1

        cx, cy = int(width // 2), int(height * 0.85)
        base_w, base_h = 140, 50

        overlay = frame.copy()
        status = "SEARCHING..."
        COLOR_BASE = (255, 200, 0)
        COLOR_LOCK = (0, 215, 255)
        COLOR_WARN = (0, 0, 255)
        active_color = COLOR_BASE
        in_zone = False

        if landmarks:
            l_ankle = landmarks[27]
            r_ankle = landmarks[28]
            lx, ly = int(l_ankle.x * width), int(l_ankle.y * height)
            rx, ry = int(r_ankle.x * width), int(r_ankle.y * height)
            fx, fy = (lx + rx) // 2, (ly + ry) // 2

            dist = math.hypot(cx - fx, cy - fy)
            in_zone = dist < 70

            if in_zone:
                self.lock_anim = min(1.0, self.lock_anim + 0.1)
                active_color = COLOR_LOCK
                status = "TARGET LOCKED"
            else:
                self.lock_anim = max(0.0, self.lock_anim - 0.1)
                if fy < (cy - 60):
                    active_color = COLOR_WARN
                    status = "MOVE BACK"
                elif fy > (cy + 60):
                    active_color = COLOR_WARN
                    status = "MOVE FWD"

            cv2.line(overlay, (lx, ly), (cx, cy), active_color, 1)
            cv2.line(overlay, (rx, ry), (cx, cy), active_color, 1)
            cv2.circle(overlay, (lx, ly), 4, active_color, -1)
            cv2.circle(overlay, (rx, ry), 4, active_color, -1)

        # Reactor core pulse
        core_size = int(10 + 5 * self.pulse_val)
        cv2.ellipse(overlay, (cx, cy), (core_size * 2, core_size), 0, 0, 360, active_color, -1)

        # Spinning rings
        for i in range(3):
            s = self.spin_angle_2 + (i * 120)
            cv2.ellipse(overlay, (cx, cy), (base_w - 20, base_h - 10), 0, s, s + 80, active_color, 2)

        if self.lock_anim > 0.8:
            cv2.ellipse(overlay, (cx, cy), (base_w, base_h), 0, 0, 360, active_color, 3)
        else:
            for i in range(4):
                s = self.spin_angle_1 + (i * 90)
                cv2.ellipse(overlay, (cx, cy), (base_w, base_h), 0, s, s + 40, active_color, 2)

        # Radial grid lines
        grid_overlay = overlay.copy()
        for i in range(0, 180, 20):
            rad = math.radians(i)
            x_off = int(math.cos(rad) * (base_w + 50))
            y_off = int(math.sin(rad) * (base_h + 30))
            cv2.line(grid_overlay, (cx, cy), (cx + x_off, cy + y_off), active_color, 1)
        cv2.addWeighted(grid_overlay, 0.3, overlay, 0.7, 0, overlay)

        cv2.putText(overlay, f"STATUS: {status}", (cx - 80, cy + base_h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, active_color, 1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        return in_zone


# Module-level singleton — imported by the worker
hologram = HologramProjector()


# =============================================================================
#  PART 2: POSE MATH ENGINE
# =============================================================================

def calculate_angle_3d(a, b, c) -> float:
    """Calculate the joint angle at point b, using 3D world coordinates."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def is_profile_view(landmarks) -> bool:
    """Returns True when the subject has turned sideways (shoulder X-spread < threshold)."""
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    return abs(l_sh.x - r_sh.x) < 0.20


def normalize_skeleton_sts_live(frames_list):
    """Formats the 88 captured frames exactly how the Keras model expects it."""
    data = np.array(frames_list).reshape(1, 88, 22, 3)  # Batch 1, 88 frames, 22 joints, 3 dims
    root = data[:, :, 0:1, :]
    data = data - root
    left_hip, right_hip = data[:, :, 18:19, :], data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    data = data / np.maximum(pelvis_width, 0.0001)
    return data.reshape(1, 88, 66)


def extract_prmd_features(lm, mirror_x=False):
    """Translates MediaPipe's 33 landmarks into UI-PRMD's 22 specific joints."""

    # If mirror_x is True, we negate X to undo the webcam mirror effect
    def pt(i):
        return [-lm[i].x if mirror_x else lm[i].x, -lm[i].y, lm[i].z]

    def avg(i, j):
        x_i = -lm[i].x if mirror_x else lm[i].x
        x_j = -lm[j].x if mirror_x else lm[j].x
        return [(x_i + x_j) / 2, -(lm[i].y + lm[j].y) / 2, (lm[i].z + lm[j].z) / 2]

    prmd = [
        avg(23, 24), pt(23), avg(11, 12), avg(11, 12), pt(0), pt(0),
        pt(11), pt(13), pt(15), pt(15), pt(12), pt(14), pt(16), pt(16),
        pt(24), pt(26), pt(28), pt(32), pt(23), pt(25), pt(27), pt(31)
    ]
    return [coord for joint in prmd for coord in joint]


def analyze_form_mechanics_3d(world_landmarks, stage: str, knee_angle: float):
    """Analyse squat form from 3-D world landmarks.
    Returns (penalty: float, feedback: list[str])."""
    penalty = 0.0
    feedback = []

    def ext(idx):
        return np.array([world_landmarks[idx].x, world_landmarks[idx].y, world_landmarks[idx].z])

    def unit_vector(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    l_sh, r_sh = ext(11), ext(12)
    l_hip, r_hip = ext(23), ext(24)
    mid_sh = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2

    # --- 1. Trunk lean check (height-adjusted tolerance) ---
    lean_tolerance = state.PARAM_LEAN_WARN + (state.USER_HEIGHT_CM - 170) * 0.1
    spine_vec = mid_sh - mid_hip
    vertical_vec = np.array([0, 1, 0])
    lean_angle = float(np.degrees(np.arccos(np.clip(np.dot(unit_vector(spine_vec), vertical_vec), -1.0, 1.0))))
    lean_from_vertical = abs(180 - lean_angle)

    if stage == "DOWN" or knee_angle < state.PARAM_SQUAT_DEPTH:
        if lean_from_vertical > state.PARAM_LEAN_CRIT:
            penalty += 0.30
            feedback.append("CRITICAL LEAN")
        elif lean_from_vertical > lean_tolerance:
            penalty += 0.10
            feedback.append("Chest Up")

    # --- 2. Back rounding check ---
    collarbone_vec = r_sh - l_sh
    dot_round = np.abs(np.dot(unit_vector(spine_vec), unit_vector(collarbone_vec)))
    rounding_angle = float(np.degrees(np.arcsin(np.clip(dot_round, 0.0, 1.0))))

    if stage == "DOWN" and rounding_angle > state.PARAM_ROUNDING:
        penalty += 0.20
        feedback.insert(0, "BACK ROUNDING")

    return penalty, feedback


def analyze_pushup_form_3d(world_landmarks, elbow_angle: float):
    """Check pushup-specific form: hip sag, elbow flare, head drop."""
    penalty = 0.0
    feedback = []

    def ext(idx):
        return np.array([world_landmarks[idx].x,
                         world_landmarks[idx].y,
                         world_landmarks[idx].z])

    l_sh, r_sh = ext(11), ext(12)
    l_hip, r_hip = ext(23), ext(24)
    l_ank, r_ank = ext(27), ext(28)
    nose = ext(0)

    mid_sh = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2
    mid_ank = (l_ank + r_ank) / 2

    # --- 1. Hip Sag / Pike check ---
    # Ideal: shoulders, hips, ankles form a straight line (small deviation)
    body_vec = mid_ank - mid_sh
    hip_offset = mid_hip - mid_sh
    if np.linalg.norm(body_vec) > 0:
        t = np.dot(hip_offset, body_vec) / np.dot(body_vec, body_vec)
        t = np.clip(t, 0.0, 1.0)
        closest = mid_sh + t * body_vec
        sag_dist = np.linalg.norm(mid_hip - closest)
        body_len = np.linalg.norm(body_vec)
        sag_ratio = sag_dist / max(body_len, 1e-6)
        # Evaluate sag mostly under load (mid/lower pushup) to reduce top-position noise.
        if elbow_angle < 140 and (
                sag_dist > state.PARAM_PUSHUP_HIP_DEV_METERS or
                sag_ratio > state.PARAM_PUSHUP_HIP_DEV_RATIO
        ):
            penalty += 0.25
            direction = "Hip Sag" if mid_hip[1] > closest[1] else "Hip Pike"
            feedback.append(direction)

    # --- 2. Head / Neck alignment ---
    # Compare neck direction to torso "up" to avoid false positives from opposite vectors.
    neck_vec = nose - mid_sh
    torso_up_vec = mid_sh - mid_hip
    if np.linalg.norm(torso_up_vec) > 0:
        head_angle = float(np.degrees(np.arccos(np.clip(
            np.dot(neck_vec / (np.linalg.norm(neck_vec) + 1e-6),
                   torso_up_vec / (np.linalg.norm(torso_up_vec) + 1e-6)), -1, 1))))
        if elbow_angle < 130 and head_angle > state.PARAM_HEAD_ANGLE:
            penalty += 0.08
            feedback.append("Head Down")

    # --- 3. Elbow flare (check at bottom of rep) ---
    if elbow_angle < 100:
        l_elb = ext(13)
        r_elb = ext(14)
        l_wr = ext(15)
        r_wr = ext(16)
        # Elbow should track roughly over wrist, not splayed wide
        l_flare = abs((l_elb - l_sh)[0]) - abs((l_wr - l_sh)[0])
        r_flare = abs((r_elb - r_sh)[0]) - abs((r_wr - r_sh)[0])
        if l_flare > 0.07 or r_flare > 0.07:
            penalty += 0.15
            feedback.append("Elbow Flare")

    return penalty, feedback


# =============================================================================
#  PART 3: AUDIO (NON-BLOCKING)
# =============================================================================

def speak_async(text: str) -> None:
    """Fire-and-forget TTS call on a daemon thread. Silently ignored if voice is off."""
    if not state.VOICE_ON:
        return

    def _speak():
        try:
            with timed("tts_synthesis"):
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
        except Exception:
            pass

    threading.Thread(target=_speak, daemon=True).start()


# =============================================================================
#  BACKGROUND MODEL LOADER
#  All 11 Keras models load in parallel daemon threads so the UI appears
#  instantly.  Every inference site already guards with `if MODEL:` and
#  falls back to a hardcoded score, so a session started before loading
#  finishes still works — it just won't have AI scoring until the model
#  arrives.
# =============================================================================

import tensorflow as tf

SQUAT_MODEL         = None
PUSHUP_MODEL        = None
STS_MODEL           = None
CURL_MODEL          = None
LATERAL_RAISE_MODEL = None

def _load_model_safe(path, use_tf=False):
    """Returns a loaded model or None. Never raises."""
    try:
        if not os.path.exists(path):
            return None
        if use_tf:
            return tf.keras.models.load_model(path)
        return load_model(path)
    except Exception:
        return None

def _bg_load_all_models():
    """Load every model on background threads, set globals when done."""
    import concurrent.futures

    global SQUAT_MODEL, PUSHUP_MODEL, STS_MODEL, CURL_MODEL, LATERAL_RAISE_MODEL

    jobs = {
        "squat":      (os.path.join(_MODEL_DIR, "deep_squat_robust.keras"),      False),
        "pushup":     (os.path.join(_MODEL_DIR, "pushup_robust.keras"),          False),
        "sts":        (os.path.join(_MODEL_DIR, "sit_to_stand_robust.keras"),    False),
        "curl":       (os.path.join(_MODEL_DIR, "bicep_curl_robust.keras"),      True),
        "lateral":    (os.path.join(_MODEL_DIR, "w_raise_robust.keras"),         True),
        "knee_ext":   (os.path.join(_MODEL_DIR, "knee_extension_robust.keras"),  True),
        "wall_push":  (os.path.join(_MODEL_DIR, "wall_pushup_robust.keras"),     True),
        "hip_march":  (os.path.join(_MODEL_DIR, "hip_march_robust.keras"),       True),
        "sh_ext":     (os.path.join(_MODEL_DIR, "shoulder_extension_robust.keras"), True),
        "sh_scap":    (os.path.join(_MODEL_DIR, "shoulder_scaption_robust.keras"),  True),
    }

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for key, (path, use_tf) in jobs.items():
            futures[key] = pool.submit(_load_model_safe, path, use_tf)

        for key, future in futures.items():
            try:
                model = future.result(timeout=120)
            except Exception:
                model = None

            if key == "squat":        SQUAT_MODEL         = model
            elif key == "pushup":     PUSHUP_MODEL        = model
            elif key == "sts":        STS_MODEL           = model
            elif key == "curl":       CURL_MODEL          = model
            elif key == "lateral":    LATERAL_RAISE_MODEL = model
            elif key == "knee_ext":   set_knee_model(model) if model else None
            elif key == "wall_push":  set_wall_model(model) if model else None
            elif key == "hip_march":  set_hip_model(model) if model else None
            elif key == "sh_ext":     set_shoulder_ext_model(model) if model else None
            elif key == "sh_scap":    set_shoulder_sca_model(model) if model else None

threading.Thread(target=_bg_load_all_models, daemon=True).start()


# --- 2. NORMALIZATION FUNCTIONS ---
def normalize_skeleton_squat_live(frames_list):
    """Squat Normalizer: 81 frames, scales by Spine Length"""
    import numpy as np
    data = np.array(frames_list).reshape(1, 81, 22, 3)
    root = data[:, :, 0:1, :]
    data = data - root
    spine_top = data[:, :, 2:3, :]
    spine_len = np.linalg.norm(spine_top, axis=3, keepdims=True)
    data = data / np.maximum(spine_len, 0.0001)
    return data.reshape(1, 81, 66)


def normalize_skeleton_sts_live(frames_list):
    """STS Normalizer: 88 frames, scales by Pelvis Width"""
    import numpy as np
    data = np.array(frames_list).reshape(1, 88, 22, 3)
    root = data[:, :, 0:1, :]
    data = data - root
    left_hip, right_hip = data[:, :, 18:19, :], data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    data = data / np.maximum(pelvis_width, 0.0001)
    return data.reshape(1, 88, 66)


def normalize_skeleton_pushup_live(frames_list):
    """Pushup Normalizer: 60 frames, scales by Shoulder Width"""
    data = np.array(frames_list).reshape(1, 60, 22, 3)
    root = data[:, :, 0:1, :]
    data = data - root
    l_sh = data[:, :, 6:7, :]  # left shoulder in PRMD mapping
    r_sh = data[:, :, 10:11, :]  # right shoulder
    shoulder_width = np.linalg.norm(l_sh - r_sh, axis=3, keepdims=True)
    data = data / np.maximum(shoulder_width, 0.0001)
    return data.reshape(1, 60, 66)


def normalize_skeleton_curl_live(frames_list):
    """
    Zero-centers the skeleton at the hips and scales by torso length.

    Matches the exact preprocessing pipeline used during LSTM training:
      - cv2.resize interpolates the variable-length sequence to exactly 40 frames
      - Hip centering: subtracts the mid-hip position so hips sit at the origin
      - Torso scaling: divides by the hip-to-shoulder-midpoint distance
    """
    import numpy as np
    import cv2

    # (N_frames, 99) → interpolate time axis to exactly 40 frames, keep 99 features
    raw = np.array(frames_list, dtype=np.float32)
    warped = cv2.resize(raw, (99, 40), interpolation=cv2.INTER_LINEAR)  # → (40, 99)

    # Reshape to (Frames, 33 joints, 3 axes)
    data = warped.reshape(-1, 33, 3)  # → (40, 33, 3)

    # ── Step 1: Center at hips ──
    mid_hip = (data[:, 23:24, :] + data[:, 24:25, :]) / 2.0  # (40, 1, 3)
    data = data - mid_hip  # hip is now at origin

    # ── Step 2: Scale by torso length ──
    # mid_sh is computed AFTER centering, so the hip origin is [0,0,0].
    # Torso length = distance from origin to shoulder midpoint = norm(mid_sh).
    # DO NOT subtract mid_hip again here — it is no longer the hip position.
    mid_sh = (data[:, 11:12, :] + data[:, 12:13, :]) / 2.0  # (40, 1, 3)
    torso_length = np.linalg.norm(mid_sh, axis=2, keepdims=True)  # ← FIXED (was: mid_sh - mid_hip)
    data = data / np.maximum(torso_length, 0.0001)

    return data.reshape(1, 40, 99)


def normalize_skeleton_lateral_raise_live(frames_list):
    """
    Matches the exact UIPRMD_REG_SSA training pipeline:
    74 frames, 66 features, Pelvis Anchor Normalization.
    """
    import numpy as np
    import cv2

    raw = np.array(frames_list, dtype=np.float32)
    # Resize to exactly 74 frames and 66 features
    warped = cv2.resize(raw, (66, 74), interpolation=cv2.INTER_LINEAR)
    data = warped.reshape(1, 74, 22, 3)

    # Center at Mid-Hip
    root = data[:, :, 0:1, :]
    data = data - root

    # Scale by Pelvis Width
    left_hip, right_hip = data[:, :, 18:19, :], data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    data = data / np.maximum(pelvis_width, 0.0001)

    return data.reshape(1, 74, 66)



def apply_mirror_matrix(landmarks):
    """Swaps left/right body parts and inverts X to undo the webcam mirror effect."""
    swap_map = {
        11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15,
        23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27,
        29: 30, 30: 29, 31: 32, 32: 31,
        1: 4, 4: 1, 2: 5, 5: 2, 3: 6, 6: 3, 7: 8, 8: 7
    }
    mirrored = []
    for i in range(33):
        target_idx = swap_map.get(i, i)
        lm = landmarks[target_idx]
        mirrored.extend([-lm.x, lm.y, lm.z])
    return mirrored


# =============================================================================
#  PART 4: VISION WORKER THREAD
# =============================================================================



class VisionWorker(QThread):
    """Runs the camera loop + MediaPipe pose on a background thread.
    Communicates back to the GUI exclusively via Qt signals."""

    frame_processed = Signal(QImage)  # Rendered camera frame
    stats_update = Signal(dict)  # Rep count / score / feedback
    system_status = Signal(str, str)  # (label_text, hex_color)
    session_finished = Signal(dict)  # Full session report dict

    # Internal FSM states
    STATE_CALIB = 0
    STATE_WARMUP = 1
    STATE_SESSION = 2

    def __init__(self):
        super().__init__()
        self.running = False
        self.exercise_mode = "squat"  # Default, UI will change this
        self.pose = mp_pose.Pose(
            min_detection_confidence=state.MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=state.MP_TRACKING_CONFIDENCE
        )
        self._knee_analyzer          = KneeExtensionAnalyzer()
        self._wall_analyzer          = WallPushupAnalyzer()
        self._hip_analyzer           = HipMarchAnalyzer()
        self._shoulder_ext_analyzer  = ShoulderExtensionAnalyzer()
        self._shoulder_sca_analyzer  = ShoulderScaptionAnalyzer()
        self.current_state = self.STATE_CALIB
        self.reset_session()

    def reset_session(self) -> None:
        self.reps = 0
        self.stage = "UP"
        self.max_rep_penalty = 0.0
        self.last_speech_time = 0.0
        self.calib_data = []
        self.session_log = []
        self.start_time = None
        self.ar_locked = False

        # --- NEW STS TRACKERS ---
        self.sts_stage = "WAITING"
        self.sts_timer = 0.0
        self.sts_buffer = []

        # --- NEW Lateral Raise TRACKERS ---
        self.lr_peak_left = 0.0
        self.lr_peak_right = 0.0
        self.lr_baseline_shrug = 1.0
        self.lr_min_shrug = 1.0
        self._lr_frame_count = 0

        self._knee_analyzer.reset()
        self._wall_analyzer.reset()
        self._hip_analyzer.reset()
        self._shoulder_ext_analyzer.reset()
        self._shoulder_sca_analyzer.reset()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        # Use CAP_DSHOW on Windows to fix instant-crash / black screen issues
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(state.CAMERA_INDEX, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(state.CAMERA_INDEX)

        # ── THE BUG FIX: Check if camera actually opened ──
        if not self.cap.isOpened():
            self.system_status.emit("CAMERA ERROR: Not Found or Blocked", "#C93535")
            self.session_finished.emit({"error": True})  # Tell UI to reset, but skip pain dialog
            return

        # Test if we can actually read a frame
        ret, frame = self.cap.read()
        if not ret:
            self.system_status.emit("CAMERA ERROR: In use by another app", "#C93535")
            self.cap.release()
            self.session_finished.emit({"error": True})  # Tell UI to reset, but skip pain dialog
            return

        self.running = True
        self.current_state = self.STATE_CALIB
        self.reset_session()
        self.start_time = datetime.now()
        self.system_status.emit("INITIALIZING...", "#ffaa00")

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if state.SESSION_TIMEOUT_MINS > 0 and self.start_time:
                if (datetime.now() - self.start_time).total_seconds() / 60 >= state.SESSION_TIMEOUT_MINS:
                    break
            if state.MIRROR_VIDEO:
                frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            with timed("mediapipe_pose"):
                results = self.pose.process(rgb_frame)

            # AR overlay
            if state.AR_MODE:
                lm_list = results.pose_landmarks.landmark if results.pose_landmarks else None
                self.ar_locked = hologram.draw(rgb_frame, lm_list, w, h)
                if not self.ar_locked and self.current_state == self.STATE_SESSION:
                    cv2.putText(rgb_frame, "ALIGN WITH TARGET", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                self.ar_locked = True

            # Pose processing
            if results.pose_landmarks and self.ar_locked:
                mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                self.process_logic(results)

            qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_processed.emit(qt_img)
            self.msleep(30)

        self.cap.release()

        # Build and emit the final session report
        report = {
            "date": self.start_time.strftime("%Y-%m-%d %H:%M") if self.start_time else "Unknown",
            "reps": self.reps,
            "avg_score": self._calculate_avg_score(),
            "details": self.session_log,
        }
        self.session_finished.emit(report)

    # ------------------------------------------------------------------
    # Finite state machine
    # ------------------------------------------------------------------
    def process_logic(self, results) -> None:
        landmarks_2d = results.pose_landmarks.landmark

        # ==========================================
        # STATE 0: PROFILE CALIBRATION
        # ==========================================
        if self.current_state == self.STATE_CALIB:
            if self.exercise_mode in ("pushup", "lateral_raise",
                                      "knee_extension", "wall_pushup", "hip_march",
                                      "shoulder_extension", "shoulder_scaption"):
                self.current_state = self.STATE_SESSION
                speak_async("System Ready.")
            elif not is_profile_view(landmarks_2d):
                self.system_status.emit("Turn Sideways", "#ff4444")
                self.calib_data = []
            else:
                self.system_status.emit("CALIBRATING...", "#0099ff")
                self.calib_data.append(1)
                if len(self.calib_data) > 30:
                    self.current_state = self.STATE_WARMUP
                    speak_async("System Ready.")

        # ==========================================
        # STATE 1: WARMUP (Transition)
        # ==========================================
        elif self.current_state == self.STATE_WARMUP:
            self.system_status.emit("GET IN POSITION", "#00cc66")
            time.sleep(0.5)
            self.current_state = self.STATE_SESSION

        # ==========================================
        # STATE 2: ACTIVE SESSION (Hybrid Engine)
        # ==========================================
        elif self.current_state == self.STATE_SESSION:
            landmarks_3d = results.pose_world_landmarks.landmark
            knee_angle = calculate_angle_3d(landmarks_3d[23], landmarks_3d[25], landmarks_3d[27])

            if self.exercise_mode == "pushup":
                left_elbow_angle = calculate_angle_3d(
                    landmarks_3d[11], landmarks_3d[13], landmarks_3d[15]
                )
                right_elbow_angle = calculate_angle_3d(
                    landmarks_3d[12], landmarks_3d[14], landmarks_3d[16]
                )

                # Use the smaller angle so one occluded/extended arm does not suppress valid reps.
                elbow_angle = min(left_elbow_angle, right_elbow_angle)
                is_up = elbow_angle > state.PARAM_PUSHUP_UP_ANGLE
                is_down = elbow_angle < state.PARAM_PUSHUP_DOWN_ANGLE

                if self.sts_stage == "WAITING":
                    if is_up:
                        self.sts_stage = "HOLDING"
                        self.sts_timer = time.time()
                        self.system_status.emit("HOLD PLANK POSITION...", "#ffaa00")

                elif self.sts_stage == "HOLDING":
                    if not is_up:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("GET IN PLANK POSITION", "#ffaa00")
                    elif time.time() - self.sts_timer > 0.5:
                        self.sts_stage = "RECORDING"
                        self.sts_buffer = []
                        self.hit_bottom = False
                        self.system_status.emit("● RECORDING (LOWER DOWN)", "#ff4444")
                        speak_async("Begin.")

                elif self.sts_stage == "RECORDING":
                    self.sts_buffer.append(extract_prmd_features(landmarks_3d))

                    penalty, issues = analyze_pushup_form_3d(landmarks_3d, elbow_angle)
                    if issues and not hasattr(self, 'current_rep_issues'):
                        self.current_rep_issues = issues[0]

                    if is_down:
                        self.hit_bottom = True
                    if getattr(self, 'hit_bottom', False) and is_up:
                        self.sts_stage = "INFERENCE"
                        self.system_status.emit("ANALYZING AI...", "#0099ff")

                    if len(self.sts_buffer) > state.PARAM_PUSHUP_TIMEOUT_FRAMES:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("TIMEOUT. RESETTING.", "#ffaa00")

                elif self.sts_stage == "INFERENCE":
                    try:
                        import cv2 as cv2
                        raw_frames = np.array(self.sts_buffer, dtype=np.float32)
                        warped_frames = cv2.resize(raw_frames, (66, 60),
                                                   interpolation=cv2.INTER_LINEAR)
                        if PUSHUP_MODEL:
                            normalized = normalize_skeleton_pushup_live(warped_frames)
                            with timed("bilstm::Push-Up"):
                                prediction = PUSHUP_MODEL.predict(normalized, verbose=0)[0][0]
                        else:
                            prediction = 0.85

                        self.reps += 1
                        raw_min, raw_max = 0.55, 0.95
                        score = int(max(0, min(100,
                                               ((prediction - raw_min) / (raw_max - raw_min)) * 100)))

                        feedback = "Excellent Form"
                        if hasattr(self, 'current_rep_issues'):
                            feedback = self.current_rep_issues
                            del self.current_rep_issues
                        elif score < 80:
                            feedback = "Compensatory Motion Detected"

                        log_entry = {"rep_num": self.reps, "score": score, "issue": feedback}
                        self.session_log.append(log_entry)
                        self.stats_update.emit({"reps": self.reps, "score": score, "feedback": feedback})

                        get_rep_feedback(
                            exercise="Push-Up",
                            rep_num=self.reps,
                            score=score,
                            issues=[log_entry["issue"]] if log_entry.get("issue") and log_entry["issue"] != "Excellent Form" else [],
                            callback=lambda text: speak_async(text),
                        )

                    except Exception as e:
                        print(f"Pushup AI Inference Error: {e}")

                    self.sts_stage = "WAITING"
                    self.system_status.emit("RESETTING...", "#ffaa00")

                return  # Don't fall through to squat/STS logic

            if self.exercise_mode == "curl":
                # Compare Z-depth of shoulders to see how the MIRRORED UI sees you
                right_shoulder_z = landmarks_3d[12].z
                left_shoulder_z = landmarks_3d[11].z

                # Scenario A: You physically face RIGHT (Mirrored screen shows facing LEFT)
                # This ironically perfectly matches the Kaggle dataset layout! No fix needed.
                facing_right = left_shoulder_z < right_shoulder_z

                if right_shoulder_z < left_shoulder_z:
                    active_sh, active_elb, active_wr = landmarks_3d[12], landmarks_3d[14], landmarks_3d[16]
                    needs_matrix_fix = False
                # Scenario B: You physically face LEFT (Mirrored screen shows facing RIGHT)
                # This is inverted from Kaggle. We MUST apply the mirror matrix.
                else:
                    active_sh, active_elb, active_wr = landmarks_3d[11], landmarks_3d[13], landmarks_3d[15]
                    needs_matrix_fix = True

                elbow_angle = calculate_angle_3d(active_sh, active_elb, active_wr)

                is_extended = elbow_angle > state.PARAM_CURL_DOWN_ANGLE
                is_curled = elbow_angle < state.PARAM_CURL_UP_ANGLE

                if self.sts_stage == "WAITING":
                    if is_extended:
                        self.sts_stage = "HOLDING"
                        self.sts_timer = time.time()
                        self.system_status.emit("HOLD ARM STRAIGHT...", "#ffaa00")

                elif self.sts_stage == "HOLDING":
                    if not is_extended:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("EXTEND ARM DOWN", "#ffaa00")
                    elif time.time() - self.sts_timer > 0.6:
                        self.sts_stage = "RECORDING"
                        self.sts_buffer = []
                        self.hit_top = False
                        self.system_status.emit("● RECORDING (CURL UP)", "#ff4444")
                        speak_async("Begin.")

                elif self.sts_stage == "RECORDING":
                    # ── Coordinate Extraction with Flip Compensation ──
                    #
                    # cv2.flip(frame, 1) is applied to the frame BEFORE MediaPipe processes it.
                    # This horizontally mirrors the image, which causes MediaPipe to output
                    # world-coordinate X-values with an INVERTED sign relative to training data.
                    #
                    # Training data was recorded raw (no flip), user facing LEFT, right arm in
                    # foreground. To feed the LSTM correctly we must restore the original X sign:
                    #
                    #   facing_right=True  → mirror_skeleton() swaps L/R joints AND negates X.
                    #                         The double-negation (flip inverted X, mirror restores it)
                    #                         plus the joint swap correctly simulates the training pose.
                    #
                    #   facing_right=False → User faces left (matches training orientation).
                    #                         We only need to negate X to undo the flip.
                    #                         No joint swap needed — the right arm is already foreground.
                    #
                    if facing_right:
                        frame_data = apply_mirror_matrix(landmarks_3d)  # swap L/R + negate X

                    else:
                        frame_data = []
                        for lm in landmarks_3d:
                            frame_data.extend([-lm.x, lm.y, lm.z])  # ← FIXED: negate X to undo cv2.flip

                    self.sts_buffer.append(frame_data)

                    if is_curled:
                        self.hit_top = True

                    if getattr(self, 'hit_top', False) and is_extended:
                        self.sts_stage = "INFERENCE"
                        self.system_status.emit("ANALYZING AI...", "#0099ff")

                    if len(self.sts_buffer) > 200:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("TIMEOUT. RESETTING.", "#ffaa00")

                elif self.sts_stage == "INFERENCE":
                    try:
                        if CURL_MODEL:
                            model_input = normalize_skeleton_curl_live(self.sts_buffer)
                            with timed("bilstm::Bicep Curl"):
                                prediction = CURL_MODEL.predict(model_input, verbose=0)[0]

                            class_idx = np.argmax(prediction)
                            confidence = prediction[class_idx]

                            # Heave bias dampener
                            if class_idx == 2 and confidence < 0.85:
                                class_idx = 3
                                confidence = 0.80

                            feedback_map = {
                                0: "Drag Cheat (Elbow Shift)",
                                1: "Half Rep (Incomplete ROM)",
                                2: "Heave Cheat (Back Momentum)",
                                3: "Excellent Form",
                                4: "Swing Cheat (Shoulder Leverage)"
                            }
                            feedback = feedback_map.get(class_idx, "Unknown Pattern")

                            # Fixed the 10% math lock
                            if class_idx == 3:
                                score = int(confidence * 100)
                            else:
                                score = int((1.0 - confidence) * 60) + 40  # Maps to a 40-100 range instead of 10
                        else:
                            feedback = "Excellent Form"
                            score = 90
                            if not getattr(self, 'hit_top', False):
                                feedback = "Half Rep (Incomplete ROM)"
                                score = 55

                        self.reps += 1
                        score = max(0, min(100, score))

                        log_entry = {"rep_num": self.reps, "score": score, "issue": feedback}
                        self.session_log.append(log_entry)
                        self.stats_update.emit({"reps": self.reps, "score": score, "feedback": feedback})

                        get_rep_feedback(
                            exercise="Bicep Curl",
                            rep_num=self.reps,
                            score=score,
                            issues=[log_entry["issue"]] if log_entry.get("issue") and log_entry["issue"] != "Excellent Form" else [],
                            callback=lambda text: speak_async(text),
                        )
                    except Exception as e:
                        print(f"Curl AI Inference Error: {e}")

                    self.sts_stage = "WAITING"
                    self.system_status.emit("RESETTING...", "#ffaa00")
                return

            if self.exercise_mode == "lateral_raise":
                # Shoulder abduction angle: angle at the shoulder between the
                # hip (torso reference) and the elbow.
                #   ~0°  → arm hanging at the side
                #   ~90° → arm raised out to shoulder height
                left_angle = calculate_angle_3d(landmarks_3d[23], landmarks_3d[11], landmarks_3d[13])
                right_angle = calculate_angle_3d(landmarks_3d[24], landmarks_3d[12], landmarks_3d[14])
                avg_angle = (left_angle + right_angle) / 2

                is_down = avg_angle < state.PARAM_LATERAL_RAISE_DOWN_ANGLE
                is_raised = avg_angle > state.PARAM_LATERAL_RAISE_DOWN_ANGLE + 20

                def _shrug_ratio():
                    """Avg ear-to-shoulder 3D distance, normalised by torso length."""
                    def dist3d(p1, p2):
                        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

                    l_sh, r_sh = landmarks_3d[11], landmarks_3d[12]
                    l_ear, r_ear = landmarks_3d[7], landmarks_3d[8]
                    l_hip, r_hip = landmarks_3d[23], landmarks_3d[24]

                    # Use true 3D Torso Length (Mid-Shoulder to Mid-Hip)
                    # This does NOT compress when the user turns sideways!
                    mid_sh_x, mid_sh_y, mid_sh_z = (l_sh.x + r_sh.x)/2, (l_sh.y + r_sh.y)/2, (l_sh.z + r_sh.z)/2
                    mid_hip_x, mid_hip_y, mid_hip_z = (l_hip.x + r_hip.x)/2, (l_hip.y + r_hip.y)/2, (l_hip.z + r_hip.z)/2
                    torso_len = math.sqrt((mid_sh_x - mid_hip_x)**2 + (mid_sh_y - mid_hip_y)**2 + (mid_sh_z - mid_hip_z)**2) + 1e-6

                    l_dist = dist3d(l_ear, l_sh)
                    r_dist = dist3d(r_ear, r_sh)

                    return ((l_dist + r_dist) / 2) / torso_len

                if self.sts_stage == "WAITING":
                    if is_down:
                        self.sts_stage = "HOLDING"
                        self.sts_timer = time.time()
                        self.system_status.emit("HOLD ARMS DOWN...", "#ffaa00")

                elif self.sts_stage == "HOLDING":
                    if not is_down:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("LOWER ARMS TO YOUR SIDES", "#ffaa00")
                    elif time.time() - self.sts_timer > 0.6:
                        self.sts_stage = "RECORDING"
                        self.sts_buffer = []
                        self.lr_peak_left = 0.0
                        self.lr_peak_right = 0.0
                        self.lr_baseline_shrug = _shrug_ratio()
                        self.lr_min_shrug = self.lr_baseline_shrug
                        self.hit_top = False
                        self.system_status.emit("● RECORDING (RAISE ARMS)", "#ff4444")
                        speak_async("Begin.")



                elif self.sts_stage == "RECORDING":

                    # mirror_x=True restores the true 3D spatial geometry so arms go OUT, not IN

                    self.sts_buffer.append(extract_prmd_features(landmarks_3d, mirror_x=True))

                    self.lr_peak_left = max(self.lr_peak_left, left_angle)

                    self.lr_peak_right = max(self.lr_peak_right, right_angle)

                    self.lr_min_shrug = min(self.lr_min_shrug, _shrug_ratio())

                    if is_raised:
                        self.hit_top = True

                    if getattr(self, 'hit_top', False) and is_down:
                        self.sts_stage = "INFERENCE"

                        self.system_status.emit("ANALYZING...", "#0099ff")

                    self._lr_frame_count += 1

                    if self._lr_frame_count > 200:
                        self.sts_stage = "WAITING"

                        self._lr_frame_count = 0

                        self.system_status.emit("TIMEOUT. RESETTING.", "#ffaa00")


                elif self.sts_stage == "INFERENCE":

                    try:

                        self._lr_frame_count = 0

                        peak_avg = (self.lr_peak_left + self.lr_peak_right) / 2

                        asymmetry = abs(self.lr_peak_left - self.lr_peak_right)

                        shrug_detected = self.lr_min_shrug < (

                                self.lr_baseline_shrug * state.PARAM_LATERAL_RAISE_SHRUG_RATIO)

                        # ---------------------------------------------------------

                        # 1. CALCULATE AI SCORE (The "Vibe" Check)

                        # ---------------------------------------------------------

                        ai_score = 90

                        if LATERAL_RAISE_MODEL:
                            normalized = normalize_skeleton_lateral_raise_live(self.sts_buffer)

                            with timed("bilstm::Lateral Raise"):
                                prediction = LATERAL_RAISE_MODEL.predict(normalized, verbose=0)[0]

                            # Using the Regression logic

                            raw_min, raw_max = 0.55, 0.95

                            pred = float(prediction[0])

                            ai_score = int(max(0, min(100, ((pred - raw_min) / (raw_max - raw_min)) * 100)))

                        # ---------------------------------------------------------

                        # 2. CALCULATE PURE MATH SCORE (The Geometry Check)

                        # ---------------------------------------------------------

                        math_score = 100

                        math_feedback = "Excellent Form"

                        rep_frames = len(self.sts_buffer)

                        # Penalty A: Momentum / Tube-Man Flailing

                        if rep_frames < 35:

                            math_score -= 40

                            math_feedback = "Momentum Cheat (Swinging / Too Fast)"


                        # Penalty B: Incomplete Range of Motion

                        elif peak_avg < 65:

                            math_score -= 35

                            math_feedback = "Half Rep (Incomplete ROM)"

                        elif peak_avg < 75:

                            math_score -= 15  # Minor deduction for being slightly low

                            math_feedback = "Raise Arms Slightly Higher"


                        # Penalty C: Asymmetry

                        elif asymmetry > state.PARAM_LATERAL_RAISE_ASYMMETRY_TOL:

                            # Dynamic deduction: worse asymmetry = heavier penalty

                            math_score -= min(35, int(asymmetry * 1.5))

                            math_feedback = "Asymmetric Raise (Uneven Arms)"


                        # Penalty D: Shrugging

                        elif shrug_detected:

                            math_score -= 30

                            math_feedback = "Shrugging (Trapezius Compensation)"

                        math_score = max(0, min(100, math_score))

                        # ---------------------------------------------------------

                        # 3. SENSOR FUSION (The Smart Blend)

                        # ---------------------------------------------------------

                        # If the math caught a blatant cheat, it takes 80% authority.

                        if math_score < 75:

                            final_score = int((0.20 * ai_score) + (0.80 * math_score))

                            feedback = math_feedback

                        else:

                            # If math looks okay, blend 50/50 for a smooth, fair grade.

                            final_score = int((0.50 * ai_score) + (0.50 * math_score))

                            # Decide on the text label

                            if ai_score < 80 and math_score >= 80:

                                feedback = "Compensatory Motion Detected"

                            else:

                                feedback = math_feedback

                        score = max(0, min(100, final_score))

                        self.reps += 1

                        log_entry = {"rep_num": self.reps, "score": score, "issue": feedback}

                        self.session_log.append(log_entry)

                        self.stats_update.emit({"reps": self.reps, "score": score, "feedback": feedback})

                        get_rep_feedback(
                            exercise="Lateral Raise",
                            rep_num=self.reps,
                            score=score,
                            issues=[log_entry["issue"]] if log_entry.get("issue") and log_entry["issue"] != "Excellent Form" else [],
                            callback=lambda text: speak_async(text),
                        )


                    except Exception as e:

                        print(f"Lateral Raise Inference Error: {e}")

                    self.sts_stage = "WAITING"

                    self.system_status.emit("RESETTING...", "#ffaa00")

                return

            if self.exercise_mode == "knee_extension":
                self._knee_analyzer.process(landmarks_3d, self)
                return

            if self.exercise_mode == "wall_pushup":
                self._wall_analyzer.process(landmarks_3d, self)
                return

            if self.exercise_mode == "hip_march":
                self._hip_analyzer.process(landmarks_3d, self)
                return

            if self.exercise_mode == "shoulder_extension":
                self._shoulder_ext_analyzer.process(landmarks_3d, self)
                return

            if self.exercise_mode == "shoulder_scaption":
                self._shoulder_sca_analyzer.process(landmarks_3d, self)
                return

            # Simple heuristic triggers
            is_standing = knee_angle > 150
            is_sitting = knee_angle < 110

            # --- PHASE 1: WAITING FOR STARTING POSE ---
            if self.sts_stage == "WAITING":
                if self.exercise_mode == "squat" and is_standing:
                    self.sts_stage = "HOLDING"
                    self.sts_timer = time.time()
                    self.system_status.emit("STAND STILL...", "#ffaa00")
                elif self.exercise_mode == "sts" and is_sitting:
                    self.sts_stage = "HOLDING"
                    self.sts_timer = time.time()
                    self.system_status.emit("HOLD STILL...", "#ffaa00")

            # --- PHASE 2: CONFIRMING POSE ---
            elif self.sts_stage == "HOLDING":
                # If they break pose early, reset
                if (self.exercise_mode == "squat" and not is_standing) or (
                        self.exercise_mode == "sts" and not is_sitting):
                    self.sts_stage = "WAITING"
                    self.system_status.emit("GET IN POSITION", "#ffaa00")

                # If they hold perfectly for 2 seconds, begin recording!
                elif time.time() - self.sts_timer > 2.0:
                    self.sts_stage = "RECORDING"
                    self.sts_buffer = []
                    self.hit_bottom = False  # Tracker for the squat depth

                    action_text = "SQUAT DOWN" if self.exercise_mode == "squat" else "STAND UP"
                    self.system_status.emit(f"● RECORDING ({action_text})", "#ff4444")
                    speak_async("Begin.")

            # --- PHASE 3: DYNAMIC RECORDING & DIAGNOSTICS ---
            elif self.sts_stage == "RECORDING":
                # 1. Save inverted frame to buffer
                self.sts_buffer.append(extract_prmd_features(landmarks_3d))

                # 2. Math Engine Diagnostics (Squat Only)
                if self.exercise_mode == "squat":
                    penalty, issues = analyze_form_mechanics_3d(landmarks_3d, "DOWN", knee_angle)
                    if issues and not hasattr(self, 'current_rep_issues'):
                        self.current_rep_issues = issues[0]

                    # Stop Condition: They went deep enough, and are now back to standing
                    if knee_angle < state.PARAM_SQUAT_DEPTH:
                        self.hit_bottom = True
                    if getattr(self, 'hit_bottom', False) and is_standing:
                        self.sts_stage = "INFERENCE"
                        self.system_status.emit("ANALYZING AI...", "#0099ff")

                # 3. Math Engine Diagnostics (STS Only)
                elif self.exercise_mode == "sts":
                    # Stop Condition: The rep is finished the moment they are fully standing
                    if is_standing:
                        self.sts_stage = "INFERENCE"
                        self.system_status.emit("ANALYZING AI...", "#0099ff")

                # 4. Failsafe Timeout
                if len(self.sts_buffer) > 200:
                    self.sts_stage = "WAITING"
                    self.system_status.emit("TIMEOUT. RESETTING.", "#ffaa00")

            # --- PHASE 4: TIME WARPING & AI GRADING ---
            elif self.sts_stage == "INFERENCE":
                try:
                    # 1. DYNAMIC TIME WARPING
                    import cv2
                    raw_frames = np.array(self.sts_buffer, dtype=np.float32)
                    target_length = 81 if self.exercise_mode == "squat" else 88

                    # Stretch or squash the movement to exact AI requirements
                    warped_frames = cv2.resize(raw_frames, (66, target_length), interpolation=cv2.INTER_LINEAR)

                    # 2. KERAS PREDICTION
                    if self.exercise_mode == "squat" and SQUAT_MODEL:
                        normalized = normalize_skeleton_squat_live(warped_frames)
                        with timed("bilstm::Deep Squat"):
                            prediction = SQUAT_MODEL.predict(normalized, verbose=0)[0][0]
                    elif self.exercise_mode == "sts" and STS_MODEL:
                        normalized = normalize_skeleton_sts_live(warped_frames)
                        with timed("bilstm::Sit to Stand"):
                            prediction = STS_MODEL.predict(normalized, verbose=0)[0][0]
                    else:
                        prediction = 0.85  # Fallback

                    self.reps += 1
                    # --- THE MAGICAL MATH: MIN-MAX SCALING ---
                    # The absolute minimum and maximum scores from the UI-PRMD dataset
                    raw_min = 0.60
                    raw_max = 0.96

                    # Stretch the prediction to a 0-100 scale
                    mapped_score = ((prediction - raw_min) / (raw_max - raw_min)) * 100

                    # "Clamp" the score to guarantee it stays between 0 and 100
                    score = int(max(0, min(100, mapped_score)))

                    # 3. COMBINE FEEDBACK
                    feedback = "Excellent Form"
                    if self.exercise_mode == "squat" and hasattr(self, 'current_rep_issues'):
                        feedback = self.current_rep_issues
                        del self.current_rep_issues
                    elif score < 80:
                        feedback = "Compensatory Motion Detected"

                    # 4. Log and update UI
                    log_entry = {"rep_num": self.reps, "score": score, "issue": feedback}
                    self.session_log.append(log_entry)
                    self.stats_update.emit({"reps": self.reps, "score": score, "feedback": feedback})

                    # 5. Speak the results
                    _ex_name = "Deep Squat" if self.exercise_mode == "squat" else "Sit to Stand"
                    get_rep_feedback(
                        exercise=_ex_name,
                        rep_num=self.reps,
                        score=score,
                        issues=[log_entry["issue"]] if log_entry.get("issue") and log_entry["issue"] != "Excellent Form" else [],
                        callback=lambda text: speak_async(text),
                    )

                except Exception as e:
                    print(f"AI Inference Error: {e}")

                self.sts_stage = "WAITING"
                self.system_status.emit("RESETTING...", "#ffaa00")

    def stop(self) -> None:
        self.running = False

    def _calculate_avg_score(self) -> int:
        if not self.session_log:
            return 0
        return int(sum(x["score"] for x in self.session_log) / len(self.session_log))