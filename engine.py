import cv2
import time
import threading
import numpy as np
import pyttsx3
import math
import os
from datetime import datetime
from keras.models import load_model as _keras_load_model


def load_model_safe(path):
    """Version-safe model loader that handles Keras 2 vs 3 kwarg mismatches.

    Models trained on Keras 3.x may contain config keys (quantization_config,
    time_major) that Keras 2.x doesn't understand, and vice versa. This wrapper
    tries a clean load first, then falls back to compile=False + manual recompile
    which bypasses most deserialization issues.
    """
    try:
        return _keras_load_model(path)
    except TypeError as e:
        # Keras 2/3 kwarg mismatch — load weights only, skip optimizer state
        print(f"[Physio-Vision] Keras version mismatch for {os.path.basename(path)}, "
              f"retrying with compile=False: {e}")
        try:
            model = _keras_load_model(path, compile=False)
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
            return model
        except Exception as e2:
            print(f"[Physio-Vision] Fallback also failed: {e2}")
            raise

# --- Environment flags must be set BEFORE any AI/GPU imports ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

from PyQt5.QtCore import QThread, pyqtSignal as Signal
from PyQt5.QtGui import QImage

from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing


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
    GUIDE_PATH: str = "Video_Generation_Person_Squatting.mp4"

    # --- Squat Analysis Thresholds ---
    PARAM_SQUAT_DEPTH: float = 140.0   # Knee angle that counts as "down"
    PARAM_UP_THRESHOLD: float = 160.0  # Knee angle that counts as "standing"
    PARAM_LEAN_WARN: float = 40.0      # Trunk lean degrees → "Chest Up" warning
    PARAM_LEAN_CRIT: float = 55.0      # Trunk lean degrees → critical alert
    PARAM_ROUNDING: float = 18.0       # Max back curvature degrees allowed
    PARAM_PUSHUP_UP_ANGLE: float = 145.0      # Elbow angle that counts as "up"
    PARAM_PUSHUP_DOWN_ANGLE: float = 105.0    # Elbow angle that counts as "down"
    PARAM_PUSHUP_TIMEOUT_FRAMES: int = 300    # Max frames allowed for one rep attempt
    PARAM_PUSHUP_HIP_DEV_METERS: float = 0.12 # Max hip deviation from body line before warning
    PARAM_PUSHUP_HIP_DEV_RATIO: float = 0.20  # Max relative hip deviation vs body length
    PARAM_HEAD_ANGLE: float = 65.0            # Max head-to-torso angle before "Head Down" warning
    PARAM_CURL_DOWN_ANGLE: float = 150.0  # Arm fully extended
    PARAM_CURL_UP_ANGLE: float = 75.0  # Arm fully curled

    # --- Knee Extension Thresholds ---
    PARAM_KNEE_EXT_UP_ANGLE: float = 170.0     # Leg fully extended (kick out)
    PARAM_KNEE_EXT_DOWN_ANGLE: float = 95.0    # Leg bent at ~90° in chair

    # --- Wall Push-Up Thresholds ---
    PARAM_WALL_PUSHUP_UP_ANGLE: float = 160.0  # Arms fully extended
    PARAM_WALL_PUSHUP_DOWN_ANGLE: float = 100.0 # Arms flexed against wall

    # --- Calf Raise Thresholds ---
    PARAM_CALF_RAISE_UP_DISP: float = 0.03     # Y-displacement threshold for peak
    PARAM_CALF_RAISE_DOWN_DISP: float = 0.01   # Y-displacement baseline

    # --- Hip March Thresholds ---
    PARAM_HIP_MARCH_UP_ANGLE: float = 65.0     # Hip flexion peak (knee pulled up)
    PARAM_HIP_MARCH_DOWN_ANGLE: float = 95.0   # Relaxed seated angle

    # --- W Raise Thresholds ---
    PARAM_W_RAISE_UP: float = 0.15             # Hand above head threshold (relative)
    PARAM_W_RAISE_DOWN: float = 0.05           # Hands at shoulder level (relative)

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


try:
    STS_MODEL = load_model_safe("sit_to_stand_robust.keras")
except:
    STS_MODEL = None
    print("[Physio-Vision] WARNING: sit_to_stand_robust.keras not found.")

def normalize_skeleton_sts_live(frames_list):
    """Formats the 88 captured frames exactly how the Keras model expects it."""
    data = np.array(frames_list).reshape(1, 88, 22, 3) # Batch 1, 88 frames, 22 joints, 3 dims
    root = data[:, :, 0:1, :]
    data = data - root
    left_hip, right_hip = data[:, :, 18:19, :], data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    data = data / np.maximum(pelvis_width, 0.0001)
    return data.reshape(1, 88, 66)


def extract_prmd_features(lm):
    """Translates MediaPipe's 33 landmarks into UI-PRMD's 22 specific joints."""

    # FIX: INVERT THE Y-AXIS (MediaPipe is positive-down, PRMD is positive-up)
    def pt(i): return [lm[i].x, -lm[i].y, lm[i].z]

    def avg(i, j): return [(lm[i].x + lm[j].x) / 2, -(lm[i].y + lm[j].y) / 2, (lm[i].z + lm[j].z) / 2]

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

    l_sh,  r_sh  = ext(11), ext(12)
    l_hip, r_hip = ext(23), ext(24)
    mid_sh  = (l_sh  + r_sh)  / 2
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

    l_sh,  r_sh  = ext(11), ext(12)
    l_hip, r_hip = ext(23), ext(24)
    l_ank, r_ank = ext(27), ext(28)
    nose         = ext(0)

    mid_sh  = (l_sh  + r_sh)  / 2
    mid_hip = (l_hip + r_hip) / 2
    mid_ank = (l_ank + r_ank) / 2

    # --- 1. Hip Sag / Pike check ---
    # Ideal: shoulders, hips, ankles form a straight line (small deviation)
    body_vec   = mid_ank - mid_sh
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
        l_wr  = ext(15)
        r_wr  = ext(16)
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
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    threading.Thread(target=_speak, daemon=True).start()

try:
    SQUAT_MODEL = load_model_safe("deep_squat_robust.keras")
    print("[Physio-Vision] SUCCESS: Squat AI Model Loaded.")
except:
    SQUAT_MODEL = None
    print("[Physio-Vision] WARNING: deep_squat_robust.keras not found.")

try:
    PUSHUP_MODEL = load_model_safe("pushup_robust.keras")
    print("[Physio-Vision] SUCCESS: Pushup AI Model Loaded.")
except:
    PUSHUP_MODEL = None
    print("[Physio-Vision] WARNING: pushup_robust.keras not found.")

# NOTE: STS_MODEL is already loaded above (line ~227). No duplicate needed.

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_model_path = os.path.join(current_dir, "bicep_curl_robust.keras")
    if os.path.exists(target_model_path):
        CURL_MODEL = load_model_safe(target_model_path)
        print("[Physio-Vision] SUCCESS: Bicep Curl AI Model Loaded.")
    else:
        CURL_MODEL = None
        print(f"[Physio-Vision] WARNING: Model file not found at {target_model_path}")
except Exception as e:
    CURL_MODEL = None
    print(f"[Physio-Vision] ERROR loading Bicep Curl model: {e}")

# --- NEW EXERCISE MODELS ---
# Uses load_model_safe() to handle Keras 2/3 version mismatches gracefully.
# Calf Raise is intentionally omitted — it uses rule-based scoring (no ML model)
# because the UI-PRMD "STS" dataset is biomechanically different from calf raises.

try:
    KNEE_EXT_MODEL = load_model_safe(os.path.join(os.path.dirname(os.path.abspath(__file__)), "knee_extension_robust.keras"))
    print("[Physio-Vision] SUCCESS: Knee Extension AI Model Loaded.")
except Exception:
    KNEE_EXT_MODEL = None
    print("[Physio-Vision] WARNING: knee_extension_robust.keras not found.")

try:
    WALL_PUSHUP_MODEL = load_model_safe(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wall_pushup_robust.keras"))
    print("[Physio-Vision] SUCCESS: Wall Push-Up AI Model Loaded.")
except Exception:
    WALL_PUSHUP_MODEL = None
    print("[Physio-Vision] WARNING: wall_pushup_robust.keras not found.")

# NOTE: No CALF_RAISE_MODEL — uses rule-based scoring (see process_logic)
print("[Physio-Vision] INFO: Calf Raise uses rule-based scoring (no ML model needed).")

try:
    HIP_MARCH_MODEL = load_model_safe(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hip_march_robust.keras"))
    print("[Physio-Vision] SUCCESS: Hip March AI Model Loaded.")
except Exception:
    HIP_MARCH_MODEL = None
    print("[Physio-Vision] WARNING: hip_march_robust.keras not found.")

try:
    W_RAISE_MODEL = load_model_safe(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w_raise_robust.keras"))
    print("[Physio-Vision] SUCCESS: W Raise AI Model Loaded.")
except Exception:
    W_RAISE_MODEL = None
    print("[Physio-Vision] WARNING: w_raise_robust.keras not found.")


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
    l_sh = data[:, :, 6:7, :]   # left shoulder in PRMD mapping
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
    data = data - mid_hip                                      # hip is now at origin

    # ── Step 2: Scale by torso length ──
    # mid_sh is computed AFTER centering, so the hip origin is [0,0,0].
    # Torso length = distance from origin to shoulder midpoint = norm(mid_sh).
    # DO NOT subtract mid_hip again here — it is no longer the hip position.
    mid_sh = (data[:, 11:12, :] + data[:, 12:13, :]) / 2.0   # (40, 1, 3)
    torso_length = np.linalg.norm(mid_sh, axis=2, keepdims=True)  # ← FIXED (was: mid_sh - mid_hip)
    data = data / np.maximum(torso_length, 0.0001)

    return data.reshape(1, 40, 99)

def normalize_skeleton_knee_ext_live(frames_list):
    """Knee Extension Normalizer: 63 frames, scales by Pelvis Width"""
    data = np.array(frames_list).reshape(1, 63, 22, 3)
    root = data[:, :, 0:1, :]
    data = data - root
    left_hip, right_hip = data[:, :, 18:19, :], data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    data = data / np.maximum(pelvis_width, 0.0001)
    return data.reshape(1, 63, 66)

def normalize_skeleton_wall_pushup_live(frames_list):
    """Wall Push-Up Normalizer: 77 frames, scales by Pelvis Width"""
    data = np.array(frames_list).reshape(1, 77, 22, 3)
    root = data[:, :, 0:1, :]
    data = data - root
    left_hip, right_hip = data[:, :, 18:19, :], data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    data = data / np.maximum(pelvis_width, 0.0001)
    return data.reshape(1, 77, 66)

def normalize_skeleton_calf_raise_live(frames_list):
    """Calf Raise Normalizer: 88 frames, scales by Pelvis Width"""
    data = np.array(frames_list).reshape(1, 88, 22, 3)
    root = data[:, :, 0:1, :]
    data = data - root
    left_hip, right_hip = data[:, :, 18:19, :], data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    data = data / np.maximum(pelvis_width, 0.0001)
    return data.reshape(1, 88, 66)

def normalize_skeleton_hip_march_live(frames_list):
    """Hip March Normalizer: 69 frames, scales by Pelvis Width"""
    data = np.array(frames_list).reshape(1, 69, 22, 3)
    root = data[:, :, 0:1, :]
    data = data - root
    left_hip, right_hip = data[:, :, 18:19, :], data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    data = data / np.maximum(pelvis_width, 0.0001)
    return data.reshape(1, 69, 66)

def normalize_skeleton_w_raise_live(frames_list):
    """W Raise Normalizer: 74 frames, scales by Pelvis Width"""
    data = np.array(frames_list).reshape(1, 74, 22, 3)
    root = data[:, :, 0:1, :]
    data = data - root
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

    frame_processed  = Signal(QImage)       # Rendered camera frame
    stats_update     = Signal(dict)         # Rep count / score / feedback
    system_status    = Signal(str, str)     # (label_text, hex_color)
    session_finished = Signal(dict)         # Full session report dict

    # Internal FSM states
    STATE_CALIB   = 0
    STATE_WARMUP  = 1
    STATE_SESSION = 2

    def __init__(self):
        super().__init__()
        self.running = False
        self.exercise_mode = "squat"  # Default, UI will change this
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        # Use CAP_DSHOW on Windows to fix instant-crash / black screen issues
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0)

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

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
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
            if self.exercise_mode in ("pushup", "wall_pushup", "w_raise"):
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
                        raw_frames    = np.array(self.sts_buffer, dtype=np.float32)
                        warped_frames = cv2.resize(raw_frames, (66, 60),
                                                   interpolation=cv2.INTER_LINEAR)
                        if PUSHUP_MODEL:
                            normalized = normalize_skeleton_pushup_live(warped_frames)
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

                        speak_text = f"Rep {self.reps}."
                        if feedback != "Excellent Form":
                            speak_text += f" {feedback}."
                        speak_async(speak_text)

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
                        frame_data = apply_mirror_matrix(landmarks_3d)   # swap L/R + negate X
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
                                score = int((1.0 - confidence) * 60) + 40 # Maps to a 40-100 range instead of 10
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

                        speak_text = f"Rep {self.reps}."
                        if feedback != "Excellent Form":
                            speak_text += f" {feedback}."
                        speak_async(speak_text)
                    except Exception as e:
                        print(f"Curl AI Inference Error: {e}")

                    self.sts_stage = "WAITING"
                    self.system_status.emit("RESETTING...", "#ffaa00")
                return

            # ============================================================
            # NEW EXERCISES: Regression-Based (Pelvis-Anchored Pipeline)
            # ============================================================
            # These exercises share the same pipeline:
            #   1. Track a trigger angle/displacement
            #   2. FSM: WAITING → HOLDING → RECORDING → INFERENCE
            #   3. Time-warp to target frames
            #   4. Pelvis-anchor normalize
            #   5. Run regression model (0-1 score)
            # ============================================================

            if self.exercise_mode == "knee_ext":
                # Track knee angle (hip → knee → ankle)
                knee_angle_right = calculate_angle_3d(landmarks_3d[24], landmarks_3d[26], landmarks_3d[28])
                knee_angle_left = calculate_angle_3d(landmarks_3d[23], landmarks_3d[25], landmarks_3d[27])
                active_knee = min(knee_angle_right, knee_angle_left)  # Use the active leg

                is_extended = active_knee > state.PARAM_KNEE_EXT_UP_ANGLE
                is_bent = active_knee < state.PARAM_KNEE_EXT_DOWN_ANGLE

                if self.sts_stage == "WAITING":
                    if is_bent:
                        self.sts_stage = "HOLDING"
                        self.sts_timer = time.time()
                        self.system_status.emit("SIT STILL — KNEES BENT...", "#ffaa00")

                elif self.sts_stage == "HOLDING":
                    if not is_bent:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("SIT WITH KNEES BENT", "#ffaa00")
                    elif time.time() - self.sts_timer > 1.5:
                        self.sts_stage = "RECORDING"
                        self.sts_buffer = []
                        self.hit_bottom = False
                        self.system_status.emit("● RECORDING (EXTEND LEG)", "#ff4444")
                        speak_async("Extend your leg.")

                elif self.sts_stage == "RECORDING":
                    self.sts_buffer.append(extract_prmd_features(landmarks_3d))

                    # Symbolic fault: Slouching check
                    def ext(idx):
                        return np.array([landmarks_3d[idx].x, landmarks_3d[idx].y, landmarks_3d[idx].z])
                    mid_sh = (ext(11) + ext(12)) / 2
                    mid_hp = (ext(23) + ext(24)) / 2
                    spine_v = mid_sh - mid_hp
                    vert = np.array([0, 1, 0])
                    spine_ang = float(np.degrees(np.arccos(np.clip(
                        np.dot(spine_v / (np.linalg.norm(spine_v) + 1e-6), vert), -1, 1))))
                    if abs(180 - spine_ang) > 45 and not hasattr(self, 'current_rep_issues'):
                        self.current_rep_issues = "Slouching / Thoracic Collapse"

                    if is_extended:
                        self.hit_bottom = True
                    if getattr(self, 'hit_bottom', False) and is_bent:
                        self.sts_stage = "INFERENCE"
                        self.system_status.emit("ANALYZING AI...", "#0099ff")

                    if len(self.sts_buffer) > 200:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("TIMEOUT. RESETTING.", "#ffaa00")

                elif self.sts_stage == "INFERENCE":
                    try:
                        import cv2
                        raw_frames = np.array(self.sts_buffer, dtype=np.float32)
                        warped_frames = cv2.resize(raw_frames, (66, 63), interpolation=cv2.INTER_LINEAR)
                        if KNEE_EXT_MODEL:
                            normalized = normalize_skeleton_knee_ext_live(warped_frames)
                            prediction = KNEE_EXT_MODEL.predict(normalized, verbose=0)[0][0]
                        else:
                            prediction = 0.85
                        self.reps += 1
                        raw_min, raw_max = 0.60, 0.96
                        score = int(max(0, min(100, ((prediction - raw_min) / (raw_max - raw_min)) * 100)))
                        feedback = "Excellent Form"
                        if hasattr(self, 'current_rep_issues'):
                            feedback = self.current_rep_issues
                            del self.current_rep_issues
                        elif score < 80:
                            feedback = "Incomplete Extension"
                        log_entry = {"rep_num": self.reps, "score": score, "issue": feedback}
                        self.session_log.append(log_entry)
                        self.stats_update.emit({"reps": self.reps, "score": score, "feedback": feedback})
                        speak_text = f"Rep {self.reps}."
                        if feedback != "Excellent Form":
                            speak_text += f" {feedback}."
                        speak_async(speak_text)
                    except Exception as e:
                        print(f"Knee Extension AI Inference Error: {e}")
                    self.sts_stage = "WAITING"
                    self.system_status.emit("RESETTING...", "#ffaa00")
                return

            if self.exercise_mode == "wall_pushup":
                # Track elbow angle (shoulder → elbow → wrist)
                left_elbow_angle = calculate_angle_3d(landmarks_3d[11], landmarks_3d[13], landmarks_3d[15])
                right_elbow_angle = calculate_angle_3d(landmarks_3d[12], landmarks_3d[14], landmarks_3d[16])
                elbow_angle = min(left_elbow_angle, right_elbow_angle)

                is_up = elbow_angle > state.PARAM_WALL_PUSHUP_UP_ANGLE
                is_down = elbow_angle < state.PARAM_WALL_PUSHUP_DOWN_ANGLE

                if self.sts_stage == "WAITING":
                    if is_up:
                        self.sts_stage = "HOLDING"
                        self.sts_timer = time.time()
                        self.system_status.emit("ARMS EXTENDED — HOLD...", "#ffaa00")

                elif self.sts_stage == "HOLDING":
                    if not is_up:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("STAND FACING WALL, ARMS OUT", "#ffaa00")
                    elif time.time() - self.sts_timer > 1.0:
                        self.sts_stage = "RECORDING"
                        self.sts_buffer = []
                        self.hit_bottom = False
                        self.system_status.emit("● RECORDING (LEAN IN)", "#ff4444")
                        speak_async("Begin.")

                elif self.sts_stage == "RECORDING":
                    self.sts_buffer.append(extract_prmd_features(landmarks_3d))

                    # Symbolic fault: Hip sag (adapt from pushup)
                    penalty, issues = analyze_pushup_form_3d(landmarks_3d, elbow_angle)
                    if issues and not hasattr(self, 'current_rep_issues'):
                        self.current_rep_issues = issues[0]

                    if is_down:
                        self.hit_bottom = True
                    if getattr(self, 'hit_bottom', False) and is_up:
                        self.sts_stage = "INFERENCE"
                        self.system_status.emit("ANALYZING AI...", "#0099ff")

                    if len(self.sts_buffer) > 300:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("TIMEOUT. RESETTING.", "#ffaa00")

                elif self.sts_stage == "INFERENCE":
                    try:
                        import cv2
                        raw_frames = np.array(self.sts_buffer, dtype=np.float32)
                        warped_frames = cv2.resize(raw_frames, (66, 77), interpolation=cv2.INTER_LINEAR)
                        if WALL_PUSHUP_MODEL:
                            normalized = normalize_skeleton_wall_pushup_live(warped_frames)
                            prediction = WALL_PUSHUP_MODEL.predict(normalized, verbose=0)[0][0]
                        else:
                            prediction = 0.85
                        self.reps += 1
                        raw_min, raw_max = 0.60, 0.96
                        score = int(max(0, min(100, ((prediction - raw_min) / (raw_max - raw_min)) * 100)))
                        feedback = "Excellent Form"
                        if hasattr(self, 'current_rep_issues'):
                            feedback = self.current_rep_issues
                            del self.current_rep_issues
                        elif score < 80:
                            feedback = "Keep Body Straight"
                        log_entry = {"rep_num": self.reps, "score": score, "issue": feedback}
                        self.session_log.append(log_entry)
                        self.stats_update.emit({"reps": self.reps, "score": score, "feedback": feedback})
                        speak_text = f"Rep {self.reps}."
                        if feedback != "Excellent Form":
                            speak_text += f" {feedback}."
                        speak_async(speak_text)
                    except Exception as e:
                        print(f"Wall Push-Up AI Inference Error: {e}")
                    self.sts_stage = "WAITING"
                    self.system_status.emit("RESETTING...", "#ffaa00")
                return

            if self.exercise_mode == "calf_raise":
                # Track vertical displacement of mid_hip and mid_ankle
                def ext(idx):
                    return np.array([landmarks_3d[idx].x, landmarks_3d[idx].y, landmarks_3d[idx].z])
                mid_hip_pt = (ext(23) + ext(24)) / 2
                mid_ankle_pt = (ext(27) + ext(28)) / 2

                if not hasattr(self, 'calf_baseline_y'):
                    self.calf_baseline_y = mid_hip_pt[1]

                y_disp = abs(mid_hip_pt[1] - self.calf_baseline_y)
                # Also check knee angle for cheat detection
                knee_angle_check = calculate_angle_3d(landmarks_3d[23], landmarks_3d[25], landmarks_3d[27])

                is_up = y_disp > state.PARAM_CALF_RAISE_UP_DISP
                is_down = y_disp < state.PARAM_CALF_RAISE_DOWN_DISP

                if self.sts_stage == "WAITING":
                    self.calf_baseline_y = mid_hip_pt[1]  # Reset baseline
                    if is_down:  # Standing still
                        self.sts_stage = "HOLDING"
                        self.sts_timer = time.time()
                        self.system_status.emit("STAND STILL — HEELS DOWN...", "#ffaa00")

                elif self.sts_stage == "HOLDING":
                    if is_up:  # Moved too early
                        self.sts_stage = "WAITING"
                        self.system_status.emit("STAND STILL WITH FLAT FEET", "#ffaa00")
                    elif time.time() - self.sts_timer > 1.5:
                        self.sts_stage = "RECORDING"
                        self.sts_buffer = []
                        self.hit_bottom = False
                        self.system_status.emit("● RECORDING (RISE UP)", "#ff4444")
                        speak_async("Rise up on your toes.")

                elif self.sts_stage == "RECORDING":
                    self.sts_buffer.append(extract_prmd_features(landmarks_3d))

                    # Symbolic fault: Knee bending cheat
                    if knee_angle_check < 165 and not hasattr(self, 'current_rep_issues'):
                        self.current_rep_issues = "Keep Knees Straight"

                    if is_up:
                        self.hit_bottom = True
                    if getattr(self, 'hit_bottom', False) and is_down:
                        self.sts_stage = "INFERENCE"
                        self.system_status.emit("ANALYZING AI...", "#0099ff")

                    if len(self.sts_buffer) > 200:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("TIMEOUT. RESETTING.", "#ffaa00")

                elif self.sts_stage == "INFERENCE":
                    # ── RULE-BASED SCORING (no ML model) ──
                    # Calf raise form is scored directly from landmark geometry.
                    # This avoids the STS↔Calf Raise dataset mismatch entirely.
                    try:
                        score = 100
                        feedback = "Excellent Form"
                        penalties = []

                        # Analyze the recorded buffer frame-by-frame
                        frames = self.sts_buffer
                        n_frames = len(frames)

                        # Metric 1: Knee bend cheat (already caught live, but re-check avg)
                        if hasattr(self, 'current_rep_issues'):
                            feedback = self.current_rep_issues
                            del self.current_rep_issues
                            score -= 25

                        # Metric 2: Rep duration quality (too fast = sloppy)
                        if n_frames < 15:
                            penalties.append("Too Fast")
                            score -= 20

                        # Metric 3: Peak displacement (higher = stronger raise)
                        if n_frames > 0:
                            # Measure peak Y-displacement from first frame
                            first_frame = np.array(frames[0]).reshape(22, 3)
                            baseline_hip_y = (first_frame[23][1] + first_frame[24][1]) / 2
                            peak_disp = 0.0
                            for f in frames:
                                f_arr = np.array(f).reshape(22, 3)
                                hip_y = (f_arr[23][1] + f_arr[24][1]) / 2
                                disp = abs(hip_y - baseline_hip_y)
                                peak_disp = max(peak_disp, disp)

                            # Weak raise = small displacement
                            if peak_disp < 0.015:
                                penalties.append("Shallow Raise")
                                score -= 15

                        # Metric 4: Trunk lean check (shoulders should stay over hips)
                        if n_frames > 5:
                            mid_idx = n_frames // 2
                            mid_f = np.array(frames[mid_idx]).reshape(22, 3)
                            mid_sh = (mid_f[11] + mid_f[12]) / 2
                            mid_hp = (mid_f[23] + mid_f[24]) / 2
                            trunk_lean = abs(mid_sh[2] - mid_hp[2])  # Z-axis lean
                            if trunk_lean > 0.12:
                                penalties.append("Trunk Leaning")
                                score -= 15

                        # Final
                        self.reps += 1
                        score = max(0, min(100, score))
                        if penalties and feedback == "Excellent Form":
                            feedback = penalties[0]  # Show the first detected issue

                        log_entry = {"rep_num": self.reps, "score": score, "issue": feedback}
                        self.session_log.append(log_entry)
                        self.stats_update.emit({"reps": self.reps, "score": score, "feedback": feedback})
                        speak_text = f"Rep {self.reps}."
                        if feedback != "Excellent Form":
                            speak_text += f" {feedback}."
                        speak_async(speak_text)
                    except Exception as e:
                        print(f"Calf Raise Rule-Based Scoring Error: {e}")
                    self.sts_stage = "WAITING"
                    if hasattr(self, 'calf_baseline_y'):
                        del self.calf_baseline_y
                    self.system_status.emit("RESETTING...", "#ffaa00")
                return

            if self.exercise_mode == "hip_march":
                # Track hip flexion angle (shoulder → hip → knee)
                hip_angle_right = calculate_angle_3d(landmarks_3d[12], landmarks_3d[24], landmarks_3d[26])
                hip_angle_left = calculate_angle_3d(landmarks_3d[11], landmarks_3d[23], landmarks_3d[25])
                active_hip = min(hip_angle_right, hip_angle_left)

                is_marched = active_hip < state.PARAM_HIP_MARCH_UP_ANGLE
                is_resting = active_hip > state.PARAM_HIP_MARCH_DOWN_ANGLE

                if self.sts_stage == "WAITING":
                    if is_resting:
                        self.sts_stage = "HOLDING"
                        self.sts_timer = time.time()
                        self.system_status.emit("SIT STILL — FEET DOWN...", "#ffaa00")

                elif self.sts_stage == "HOLDING":
                    if not is_resting:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("SIT WITH FEET FLAT", "#ffaa00")
                    elif time.time() - self.sts_timer > 1.5:
                        self.sts_stage = "RECORDING"
                        self.sts_buffer = []
                        self.hit_bottom = False
                        self.system_status.emit("● RECORDING (LIFT KNEE)", "#ff4444")
                        speak_async("Lift your knee.")

                elif self.sts_stage == "RECORDING":
                    self.sts_buffer.append(extract_prmd_features(landmarks_3d))

                    # Symbolic fault: Trunk recline / lean back
                    def ext_hip(idx):
                        return np.array([landmarks_3d[idx].x, landmarks_3d[idx].y, landmarks_3d[idx].z])
                    mid_sh = (ext_hip(11) + ext_hip(12)) / 2
                    mid_hp = (ext_hip(23) + ext_hip(24)) / 2
                    spine_v = mid_sh - mid_hp
                    vert = np.array([0, 1, 0])
                    spine_ang = float(np.degrees(np.arccos(np.clip(
                        np.dot(spine_v / (np.linalg.norm(spine_v) + 1e-6), vert), -1, 1))))
                    if abs(180 - spine_ang) > 50 and not hasattr(self, 'current_rep_issues'):
                        self.current_rep_issues = "Don't Lean Back"

                    if is_marched:
                        self.hit_bottom = True
                    if getattr(self, 'hit_bottom', False) and is_resting:
                        self.sts_stage = "INFERENCE"
                        self.system_status.emit("ANALYZING AI...", "#0099ff")

                    if len(self.sts_buffer) > 200:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("TIMEOUT. RESETTING.", "#ffaa00")

                elif self.sts_stage == "INFERENCE":
                    try:
                        import cv2
                        raw_frames = np.array(self.sts_buffer, dtype=np.float32)
                        warped_frames = cv2.resize(raw_frames, (66, 69), interpolation=cv2.INTER_LINEAR)
                        if HIP_MARCH_MODEL:
                            normalized = normalize_skeleton_hip_march_live(warped_frames)
                            prediction = HIP_MARCH_MODEL.predict(normalized, verbose=0)[0][0]
                        else:
                            prediction = 0.85
                        self.reps += 1
                        raw_min, raw_max = 0.60, 0.96
                        score = int(max(0, min(100, ((prediction - raw_min) / (raw_max - raw_min)) * 100)))
                        feedback = "Excellent Form"
                        if hasattr(self, 'current_rep_issues'):
                            feedback = self.current_rep_issues
                            del self.current_rep_issues
                        elif score < 80:
                            feedback = "Shallow March (Weak Flexion)"
                        log_entry = {"rep_num": self.reps, "score": score, "issue": feedback}
                        self.session_log.append(log_entry)
                        self.stats_update.emit({"reps": self.reps, "score": score, "feedback": feedback})
                        speak_text = f"Rep {self.reps}."
                        if feedback != "Excellent Form":
                            speak_text += f" {feedback}."
                        speak_async(speak_text)
                    except Exception as e:
                        print(f"Hip March AI Inference Error: {e}")
                    self.sts_stage = "WAITING"
                    self.system_status.emit("RESETTING...", "#ffaa00")
                return

            if self.exercise_mode == "w_raise":
                # Track hand elevation relative to head
                def ext_wr(idx):
                    return np.array([landmarks_3d[idx].x, landmarks_3d[idx].y, landmarks_3d[idx].z])
                l_hand, r_hand = ext_wr(15), ext_wr(16)
                head = ext_wr(0)
                l_elbow, r_elbow = ext_wr(13), ext_wr(14)
                l_shoulder, r_shoulder = ext_wr(11), ext_wr(12)

                # "Up" = hands above head, "Down" = elbows at shoulder level
                l_elbow_angle = calculate_angle_3d(landmarks_3d[11], landmarks_3d[13], landmarks_3d[15])
                r_elbow_angle = calculate_angle_3d(landmarks_3d[12], landmarks_3d[14], landmarks_3d[16])

                # Start/Up: hands above head level (V shape)
                hands_up = (l_hand[1] < head[1]) and (r_hand[1] < head[1])  # MediaPipe Y is inverted in world
                # End/Down: elbows drop to shoulder level, ~90° angle
                elbows_down = l_elbow_angle < 110 and r_elbow_angle < 110

                if self.sts_stage == "WAITING":
                    if hands_up:
                        self.sts_stage = "HOLDING"
                        self.sts_timer = time.time()
                        self.system_status.emit("ARMS UP — HOLD V SHAPE...", "#ffaa00")

                elif self.sts_stage == "HOLDING":
                    if not hands_up:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("RAISE ARMS OVERHEAD", "#ffaa00")
                    elif time.time() - self.sts_timer > 1.0:
                        self.sts_stage = "RECORDING"
                        self.sts_buffer = []
                        self.hit_bottom = False
                        self.system_status.emit("● RECORDING (PULL DOWN TO W)", "#ff4444")
                        speak_async("Pull down to W shape.")

                elif self.sts_stage == "RECORDING":
                    self.sts_buffer.append(extract_prmd_features(landmarks_3d))

                    # Symbolic fault: Asymmetric pulling
                    l_elb_y = landmarks_3d[13].y
                    r_elb_y = landmarks_3d[14].y
                    if abs(l_elb_y - r_elb_y) > 0.05 and not hasattr(self, 'current_rep_issues'):
                        self.current_rep_issues = "Pull Evenly on Both Sides"

                    # Symbolic fault: Forward head
                    nose_pt = ext_wr(0)
                    mid_sh_pt = (ext_wr(11) + ext_wr(12)) / 2
                    if abs(nose_pt[2] - mid_sh_pt[2]) > 0.15 and not hasattr(self, 'current_rep_issues'):
                        self.current_rep_issues = "Forward Head Jutting"

                    if elbows_down:
                        self.hit_bottom = True
                    if getattr(self, 'hit_bottom', False) and hands_up:
                        self.sts_stage = "INFERENCE"
                        self.system_status.emit("ANALYZING AI...", "#0099ff")

                    if len(self.sts_buffer) > 200:
                        self.sts_stage = "WAITING"
                        self.system_status.emit("TIMEOUT. RESETTING.", "#ffaa00")

                elif self.sts_stage == "INFERENCE":
                    try:
                        import cv2
                        raw_frames = np.array(self.sts_buffer, dtype=np.float32)
                        warped_frames = cv2.resize(raw_frames, (66, 74), interpolation=cv2.INTER_LINEAR)
                        if W_RAISE_MODEL:
                            normalized = normalize_skeleton_w_raise_live(warped_frames)
                            prediction = W_RAISE_MODEL.predict(normalized, verbose=0)[0][0]
                        else:
                            prediction = 0.85
                        self.reps += 1
                        raw_min, raw_max = 0.60, 0.96
                        score = int(max(0, min(100, ((prediction - raw_min) / (raw_max - raw_min)) * 100)))
                        feedback = "Excellent Form"
                        if hasattr(self, 'current_rep_issues'):
                            feedback = self.current_rep_issues
                            del self.current_rep_issues
                        elif score < 80:
                            feedback = "Asymmetric Pulling"
                        log_entry = {"rep_num": self.reps, "score": score, "issue": feedback}
                        self.session_log.append(log_entry)
                        self.stats_update.emit({"reps": self.reps, "score": score, "feedback": feedback})
                        speak_text = f"Rep {self.reps}."
                        if feedback != "Excellent Form":
                            speak_text += f" {feedback}."
                        speak_async(speak_text)
                    except Exception as e:
                        print(f"W Raise AI Inference Error: {e}")
                    self.sts_stage = "WAITING"
                    self.system_status.emit("RESETTING...", "#ffaa00")
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
                        prediction = SQUAT_MODEL.predict(normalized, verbose=0)[0][0]
                    elif self.exercise_mode == "sts" and STS_MODEL:
                        normalized = normalize_skeleton_sts_live(warped_frames)
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
                    speak_text = f"Rep {self.reps}."
                    if feedback != "Excellent Form":
                        speak_text += f" {feedback}."
                    speak_async(speak_text)

                except Exception as e:
                    print(f"AI Inference Error: {e}")

                self.sts_stage = "WAITING"
                self.system_status.emit("RESETTING...", "#ffaa00")

    def stop(self) -> None:
        self.running = False
        self.wait()

    def _calculate_avg_score(self) -> int:
        if not self.session_log:
            return 0
        return int(sum(x["score"] for x in self.session_log) / len(self.session_log))