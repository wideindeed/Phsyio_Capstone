import numpy as np
import cv2

TIME_STEPS = 67
MODEL_GLOBAL = None
RAW_MIN = 0.55
RAW_MAX = 0.95
HOLD_SECONDS = 0.6
TIMEOUT_FRAMES = 200

def set_model(m):
    global MODEL_GLOBAL
    MODEL_GLOBAL = m

def normalize_live(frames_list):
    raw = np.array(frames_list, dtype=np.float32)
    warped = cv2.resize(raw, (66, TIME_STEPS), interpolation=cv2.INTER_LINEAR)
    data = warped.reshape(1, TIME_STEPS, 22, 3)
    root = data[:, :, 0:1, :]
    data = data - root
    left_hip = data[:, :, 18:19, :]
    right_hip = data[:, :, 14:15, :]
    pelvis_w = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    data = data / np.maximum(pelvis_w, 0.0001)
    return data.reshape(1, TIME_STEPS, 66)

def raw_to_score(pred):
    return int(max(0, min(100, ((pred - RAW_MIN) / (RAW_MAX - RAW_MIN)) * 100)))

class ShoulderExtensionAnalyzer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.stage = "WAITING"
        self.buffer = []
        self.frame_count = 0
        self.hit_top = False
        self.hold_start = 0.0

    def process(self, landmarks_3d, worker):
        import time
        from engine import calculate_angle_3d, extract_prmd_features, speak_async

        left_angle  = calculate_angle_3d(landmarks_3d[23], landmarks_3d[11], landmarks_3d[13])
        right_angle = calculate_angle_3d(landmarks_3d[24], landmarks_3d[12], landmarks_3d[14])
        avg_angle   = (left_angle + right_angle) / 2

        is_down   = avg_angle > 160.0
        is_raised = avg_angle < 145.0

        if self.stage == "WAITING":
            if is_down:
                self.stage = "HOLDING"
                self.hold_start = time.time()
                worker.system_status.emit("HOLD STILL...", "#ffaa00")

        elif self.stage == "HOLDING":
            if not is_down:
                self.stage = "WAITING"
                worker.system_status.emit("ARMS AT YOUR SIDES", "#ffaa00")
            elif time.time() - self.hold_start > HOLD_SECONDS:
                self.stage = "RECORDING"
                self.buffer = []
                self.frame_count = 0
                self.hit_top = False
                worker.system_status.emit("● RECORDING — EXTEND ARMS BACK", "#ff4444")
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
                self.stage = "WAITING"
                self.frame_count = 0
                worker.system_status.emit("TIMEOUT — ARMS AT YOUR SIDES", "#ffaa00")

        elif self.stage == "INFERENCE":
            try:
                if len(self.buffer) >= 5 and MODEL_GLOBAL:
                    normalized = normalize_live(self.buffer)
                    pred = float(MODEL_GLOBAL.predict(normalized, verbose=0)[0][0])
                    score = raw_to_score(pred)
                    feedback = "Excellent Form" if score >= 80 else "Compensatory Motion Detected"
                    worker.reps += 1
                    log_entry = {"rep_num": worker.reps, "score": score, "issue": feedback}
                    worker.session_log.append(log_entry)
                    worker.stats_update.emit({"reps": worker.reps, "score": score, "feedback": feedback})
                    speak_text = f"Rep {worker.reps}."
                    if feedback != "Excellent Form":
                        speak_text += f" {feedback}."
                    speak_async(speak_text)
            except Exception as e:
                print(f"[ShoulderExtension] Inference error: {e}")
            self.stage = "WAITING"
            worker.system_status.emit("ARMS AT YOUR SIDES", "#ffaa00")
