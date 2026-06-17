import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
import random

# --- 1. CONFIG: THE UNSEEN DATA ---
# We point this specifically to fold1 to guarantee the AI has never seen it
TEST_DIR = "UIPRMD_STS/fold2"
MODEL_PATH = "sit_to_stand_robust.keras"


# --- 2. THE NORMALIZATION FUNCTION ---
# This must be identical to the one you used to train it!
def normalize_skeleton_sts(data):
    B, T, F = data.shape
    J = F // 3
    data = data.reshape(B, T, J, 3)

    # Center at Mid-Hip
    root = data[:, :, 0:1, :]
    data = data - root

    # Scale by Pelvis Width
    left_hip = data[:, :, 18:19, :]
    right_hip = data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    pelvis_width = np.maximum(pelvis_width, 0.0001)

    data = data / pelvis_width
    return data.reshape(B, T, F)


# --- 3. LOAD THE EXAM DATA ---
print(f"Loading unseen exam data from {TEST_DIR}...")
try:
    # Notice we are loading the fold1 files
    x_exam = np.load(os.path.join(TEST_DIR, "x_test_fold2.npy"))
    y_exam = np.load(os.path.join(TEST_DIR, "y_test_fold2.npy"))
except FileNotFoundError:
    print(f"ERROR: Could not find fold2 data. Please ensure {TEST_DIR} exists and has the fold2 .npy files.")
    exit()

# Shape correction
if x_exam.shape[1] == 66:
    x_exam = x_exam.transpose(0, 2, 1)

print("Normalizing Exam Data...")
x_exam = normalize_skeleton_sts(x_exam)

# --- 4. LOAD THE MODEL & GRADE ---
print(f"Loading trained model: {MODEL_PATH}...")
model = load_model(MODEL_PATH)

print("\n--- FORMAL EVALUATION ---")
# evaluate() runs the model across the entire dataset and returns the average loss and MAE
results = model.evaluate(x_exam, y_exam, verbose=0)
print(f"Overall Exam Mean Absolute Error (MAE): {results[1]:.4f}")
print(f"(This means the AI is off by an average of {results[1] * 100:.2f}% per guess)")

# --- 5. THE REALITY CHECK (Side-by-Side Comparison) ---
print("\n--- BLIND TEST: AI vs REAL DOCTOR ---")
predictions = model.predict(x_exam, verbose=0)

# Pick 5 random repetitions from the exam to show you
num_samples = len(x_exam)
random_indices = random.sample(range(num_samples), 5)

for i in random_indices:
    actual_score = y_exam[i]
    ai_guess = predictions[i][0]
    difference = abs(actual_score - ai_guess)

    print(
        f"Repetition #{i:03d} | Doctor Score: {actual_score:.3f} | AI Guess: {ai_guess:.3f} | Error: {difference:.3f}")

print("\nTesting Complete.")