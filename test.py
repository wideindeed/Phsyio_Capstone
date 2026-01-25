import cv2
import numpy as np
import tensorflow as tf
import os

# --- USER PROFILE (CONSTANTS) ---
USER_HEIGHT_CM = 193.0
USER_WEIGHT_KG = 84.0
REAL_SPINE_LENGTH_M = (USER_HEIGHT_CM * 0.29) / 100.0

# --- IMPORTS & SETUP ---
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'deep_squat_robust.keras')

try:
    model = tf.keras.models.load_model(model_path)
    print("AI Model loaded.")
except:
    try:
        from keras.models import load_model

        model = load_model(model_path)
        print("AI Model loaded (Fallback).")
    except:
        print("Warning: Model not found. Running in Physics Mode.")
        model = None


# --- UPDATED: STRICTER FORM JUDGE ---
# --- UPDATED: MECHANICS JUDGE WITH STAGGER CHECK ---
# --- UPDATED: TUNED MECHANICS JUDGE (TALL USER FRIENDLY) ---
def analyze_form_mechanics_3d(world_landmarks, stage, knee_angle):
    """
    Adjusted for Biomechanics:
    1. Lean Threshold relaxed to 40 deg (accommodates long torso/femurs).
    2. Only enforces lean when deep (prevents warnings during transition).
    """
    penalty = 0.0
    feedback = []

    def ext(idx):
        return np.array([world_landmarks[idx].x, world_landmarks[idx].y, world_landmarks[idx].z])

    def unit_vector(v):
        return v / np.linalg.norm(v)

    # Get Key Landmarks
    l_sh, r_sh = ext(11), ext(12)
    l_hip, r_hip = ext(23), ext(24)
    mid_sh, mid_hip = (l_sh + r_sh) / 2, (l_hip + r_hip) / 2

    # --- CHECK 1: TRUNK LEAN (RELAXED) ---
    spine_vec = mid_sh - mid_hip
    vertical_vec = np.array([0, 1, 0])

    dot_prod = np.dot(unit_vector(spine_vec), vertical_vec)
    lean_angle_raw = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
    lean_from_vertical = abs(180 - lean_angle_raw)

    # RULE: Only judge lean if we are actually squatting (angle < 160)
    # AND we are in the 'DOWN' phase.
    if stage == "DOWN" or knee_angle < 140:
        # Threshold relaxed from 30 -> 40 degrees
        if lean_from_vertical > 55:
            penalty += 0.40
            feedback.append(f"EXTREME LEAN ({int(lean_from_vertical)}°)")
        elif lean_from_vertical > 40:
            penalty += 0.40  # Lowered penalty weight too
            feedback.append("Keep Chest Up")

    # --- CHECK 2: TORSO ROUNDING (UNCHANGED) ---
    collarbone_vec = r_sh - l_sh
    dot_prod_round = np.abs(np.dot(unit_vector(spine_vec), unit_vector(collarbone_vec)))
    rounding_angle = np.degrees(np.arcsin(np.clip(dot_prod_round, 0.0, 1.0)))

    if stage == "DOWN":
        if rounding_angle > 18:  # Slightly relaxed from 15 -> 18
            penalty += 0.40
            feedback.insert(0, "ROUNDING BACK!")

            # --- CHECK 3: STAGGERED FEET ---
    l_ankle, r_ankle = ext(27), ext(28)
    stagger_dist = abs(l_ankle[0] - r_ankle[0])
    shoulder_width = np.linalg.norm(l_sh - r_sh)

    if stagger_dist > (shoulder_width * 0.5):  # Relaxed from 0.4 -> 0.5
        penalty += 0.20
        feedback.append("ALIGN FEET")

    return penalty, feedback

# --- HELPER: 3D ANGLE CALCULATION ---
def calculate_angle_3d(a, b, c):
    """Calculates angle ABC using 3D coordinates"""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


# --- AI DATA MAPPING (KEEPS 2D FOR COMPATIBILITY) ---
def get_uiprmd_mapping_calibrated(landmarks, calibration_scale=None):
    def ext(idx):
        return [landmarks[idx].x, -landmarks[idx].y, landmarks[idx].z]

    # ... (Same logic as before to match training data) ...
    # We do NOT change this function because the AI was trained on this specific 2D/pseudo-3D format
    left_hip, right_hip = np.array(ext(23)), np.array(ext(24))
    mid_hip = (left_hip + right_hip) / 2
    left_shoulder, right_shoulder = np.array(ext(11)), np.array(ext(12))
    spine_shoulder = (left_shoulder + right_shoulder) / 2
    joint_data = [
        mid_hip, mid_hip, spine_shoulder,
        ext(0), ext(0), ext(11), ext(13), ext(15), ext(19),
        ext(12), ext(14), ext(16), ext(20),
        ext(23), ext(25), ext(27), ext(31),
        ext(24), ext(26), ext(28), ext(32),
        spine_shoulder
    ]
    root = joint_data[0]
    centered_data = [j - root for j in joint_data]
    if calibration_scale:
        spine_len = calibration_scale
    else:
        spine_len = np.linalg.norm(centered_data[2] - centered_data[0]) or 1.0
    scaled_data = [j / spine_len for j in centered_data]
    return np.concatenate(scaled_data)


# --- MAIN APP LOOP ---
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

STATE_CALIBRATING, STATE_WARMUP, STATE_SESSION = 0, 1, 2
current_state = STATE_CALIBRATING

# PHYSICS VARS
user_spine_px, pixels_per_meter = None, None
start_hip_y, lowest_hip_y = 0, 0
user_spine_norm = None

# SESSION VARS
calib_frames_px, calib_frames_norm = [], []
reps_completed, target_reps = 0, 5
stage = "UP"
rep_scores, session_report = [], []
sequence_buffer, physics_report = [], []
current_feedback = ""

max_rep_penalty = 0.0

print(f"--- 3D HYBRID MODE ACTIVE ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    cv2.rectangle(frame, (0, 0), (640, 90), (30, 30, 30), -1)

    if results.pose_landmarks and results.pose_world_landmarks:
        landmarks_2d = results.pose_landmarks.landmark
        landmarks_3d = results.pose_world_landmarks.landmark

        # --- PHASE 1: CALIBRATION ---
        if current_state == STATE_CALIBRATING:
            cv2.putText(frame, "CALIBRATION...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Stand Profile & Still", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


            def ext_y_px(idx):
                return landmarks_2d[idx].y * frame.shape[0]


            current_spine_px = abs(((ext_y_px(23) + ext_y_px(24)) / 2) - ((ext_y_px(11) + ext_y_px(12)) / 2))
            calib_frames_px.append(current_spine_px)


            def ext_norm(idx):
                return np.array([landmarks_2d[idx].x, -landmarks_2d[idx].y, landmarks_2d[idx].z])


            current_spine_norm = np.linalg.norm(
                ((ext_norm(11) + ext_norm(12)) / 2) - ((ext_norm(23) + ext_norm(24)) / 2))
            calib_frames_norm.append(current_spine_norm)

            cv2.rectangle(frame, (0, 85), (int(640 * (len(calib_frames_px) / 60)), 90), (0, 255, 0), -1)

            if len(calib_frames_px) >= 60:
                user_spine_px = np.median(calib_frames_px)
                user_spine_norm = np.median(calib_frames_norm)
                pixels_per_meter = user_spine_px / REAL_SPINE_LENGTH_M
                current_state = STATE_WARMUP

        # --- PHASE 2: WARMUP ---
        elif current_state == STATE_WARMUP:
            cv2.putText(frame, "BUFFERING AI...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            model_input = get_uiprmd_mapping_calibrated(landmarks_2d, user_spine_norm)
            sequence_buffer.append(model_input)
            sequence_buffer = sequence_buffer[-81:]
            if len(sequence_buffer) == 81:
                current_state = STATE_SESSION
                print("Session Started.")

        # --- PHASE 3: SESSION ---
        elif current_state == STATE_SESSION:
            # 1. Physics
            hip_y_px = ((landmarks_2d[23].y + landmarks_2d[24].y) / 2) * frame.shape[0]
            if stage == "UP":
                start_hip_y = hip_y_px
            elif stage == "DOWN":
                lowest_hip_y = max(lowest_hip_y, hip_y_px)

            # 2. AI Input
            model_input = get_uiprmd_mapping_calibrated(landmarks_2d, user_spine_norm)
            sequence_buffer.append(model_input)
            sequence_buffer = sequence_buffer[-81:]

            # 3. Geometry
            hip = landmarks_3d[23]
            knee = landmarks_3d[25]
            ankle = landmarks_3d[27]
            angle = calculate_angle_3d(hip, knee, ankle)

            # --- HYBRID JUDGE ---
            current_penalty, issues = analyze_form_mechanics_3d(landmarks_3d, stage, angle)

            if stage == "DOWN":
                if current_penalty > max_rep_penalty:
                    max_rep_penalty = current_penalty

            if issues:
                current_feedback = issues[0]
            else:
                current_feedback = ""

            # State Machine
            if angle < 120 and stage == "UP":
                stage = "DOWN"
                lowest_hip_y = hip_y_px
                max_rep_penalty = 0.0

            if angle > 160:
                if stage == "DOWN":
                    reps_completed += 1

                    # Grade Rep
                    final_grade = 0.0
                    if len(rep_scores) > 2:
                        rep_scores.sort(reverse=True)
                        top_scores = rep_scores[:max(1, int(len(rep_scores) * 0.3))]
                        final_grade = sum(top_scores) / len(top_scores)

                    # --- SCALING FIX (The 100% Patch) ---
                    # 1. Scale AI score up by 15% (turns 0.87 into 1.0)
                    scaled_ai_score = min(1.0, final_grade * 1.15)

                    # 2. Apply Penalty to the SCALED score
                    final_score = max(0.0, scaled_ai_score - max_rep_penalty)

                    session_report.append(final_score)

                    # Calc Work
                    travel_m = (lowest_hip_y - start_hip_y) / pixels_per_meter
                    work_joules = USER_WEIGHT_KG * 9.8 * travel_m
                    physics_report.append((travel_m, work_joules))

                    print(
                        f"Rep {reps_completed}: AI {int(final_grade * 100)}% -> Scaled {int(scaled_ai_score * 100)}% - Pen {int(max_rep_penalty * 100)}% = FINAL {int(final_score * 100)}%")

                    rep_scores = []
                    max_rep_penalty = 0.0

                    if reps_completed >= target_reps: print("Done! Press Q.")

                stage = "UP"

            # UI Updates
            cv2.putText(frame, f"REPS: {reps_completed}/{target_reps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            if current_feedback:
                cv2.putText(frame, f"WARN: {current_feedback}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # AI Inference
            if model and angle < 150 and len(sequence_buffer) == 81:
                input_tensor = np.expand_dims(sequence_buffer, axis=0)
                pred = model.predict(input_tensor, verbose=0)[0][0]

                # Curve the live bar too
                curved_pred = min(1.0, pred * 1.15)
                display_score = max(0, curved_pred - current_penalty)

                rep_scores.append(pred)  # Keep tracking raw for stats

                bar_width = int(display_score * 150)
                color = (0, 255, 0) if display_score > 0.8 else (0, 0, 255)
                cv2.rectangle(frame, (480, 20), (480 + bar_width, 50), color, -1)
                cv2.putText(frame, f"FORM: {int(display_score * 100)}%", (480, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Physio-Vision Hybrid 3D', frame)


    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# --- REPORT ---
print("\n" + "=" * 45)
print(f"   PHYSIO-VISION REPORT (User: {int(USER_HEIGHT_CM)}cm/{int(USER_WEIGHT_KG)}kg)   ")
print("=" * 45)
for i, (depth, work) in enumerate(physics_report):
    grade = session_report[i] if i < len(session_report) else 0.0
    status = "EXCELLENT" if grade > 0.8 else "GOOD" if grade > 0.6 else "IMPROVE"
    print(f"Rep {i + 1}: {int(grade * 100)}/100 [{status}]")
    print(f"  > Depth: {int(depth * 100)}cm | Energy: {int(work)}J")
print("=" * 45)