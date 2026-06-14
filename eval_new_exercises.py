"""
eval_new_exercises.py
=====================
Evaluation script for the 5 new elderly-focused exercise models.
Loads held-out fold1 test data, normalizes, runs model.evaluate(),
and shows a blind test of 5 random predictions vs ground truth.

Usage:
    python eval_new_exercises.py --exercise knee_extension   # Evaluate one model
    python eval_new_exercises.py --all                       # Evaluate all models
    python eval_new_exercises.py --list                      # List available exercises
"""

import argparse
import os
import sys
import numpy as np
from keras.models import load_model

# ---------------------------------------------------------------------------
#  1. EXERCISE CONFIGURATION (mirrors train_new_exercises.py)
# ---------------------------------------------------------------------------

EXERCISES = {
    "knee_extension": {
        "name": "Seated Knee Extension",
        "data_dir": "UIPRMD_reg/SASLR",       # fold1 lives inside here
        "data_acronym": "SASLR",
        "time_steps": 63,
        "num_features": 66,
        "output_model": "knee_extension_robust.keras",
    },
    "wall_pushup": {
        "name": "Wall Push-Up",
        "data_dir": "UIPRMD_reg/IL",
        "data_acronym": "IL",
        "time_steps": 77,
        "num_features": 66,
        "output_model": "wall_pushup_robust.keras",
    },
    "calf_raise": {
        "name": "Calf Raise (Heel Raise)",
        "data_dir": "UIPRMD_reg/STS",
        "data_acronym": "STS",
        "time_steps": 88,
        "num_features": 66,
        "output_model": "calf_raise_robust.keras",
    },
    "hip_march": {
        "name": "Seated Hip March",
        "data_dir": "UIPRMD_reg/HS",
        "data_acronym": "HS",
        "time_steps": 69,
        "num_features": 66,
        "output_model": "hip_march_robust.keras",
    },
    "w_raise": {
        "name": "Seated W Raise (Scapular Retraction)",
        "data_dir": "UIPRMD_reg/SSA",
        "data_acronym": "SSA",
        "time_steps": 74,
        "num_features": 66,
        "output_model": "w_raise_robust.keras",
    },
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
#  2. PELVIS ANCHOR NORMALIZATION (exact copy from train.py / train_new_exercises.py)
# ---------------------------------------------------------------------------

def normalize_skeleton(data):
    """
    1. Reshape to (Batch, Time, Joints, 3)
    2. Center at Mid-Hip (Joint 0)
    3. Scale so Pelvis Width = 1.0 (Rigid Anchor)
    4. Flatten back
    """
    B, T, F = data.shape
    J = F // 3
    data = data.reshape(B, T, J, 3)

    root = data[:, :, 0:1, :]
    data = data - root

    left_hip = data[:, :, 18:19, :]
    right_hip = data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    pelvis_width = np.maximum(pelvis_width, 0.0001)
    data = data / pelvis_width

    return data.reshape(B, T, F)


# ---------------------------------------------------------------------------
#  3. EVALUATION PIPELINE
# ---------------------------------------------------------------------------

def evaluate_exercise(key):
    """Run full evaluation for a single exercise."""
    cfg = EXERCISES[key]
    name = cfg["name"]
    time_steps = cfg["time_steps"]
    num_features = cfg["num_features"]
    model_path = os.path.join(BASE_DIR, cfg["output_model"])
    fold1_path = os.path.join(BASE_DIR, cfg["data_dir"], "fold1")

    print("=" * 70)
    print(f"  EVALUATING: {name} ({key})")
    print("=" * 70)

    # --- Check prerequisites ---
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("        Train the model first with train_new_exercises.py")
        return None

    if not os.path.isdir(fold1_path):
        print(f"[ERROR] Fold1 test data not found: {fold1_path}")
        print("        Run download_data.py first.")
        return None

    # --- Load fold1 test data ---
    print(f"[1/4] Loading fold1 test data from {fold1_path}...")
    x_test = np.load(os.path.join(fold1_path, "x_test_fold1.npy"))
    y_test = np.load(os.path.join(fold1_path, "y_test_fold1.npy"))

    # --- Transpose if needed ---
    if x_test.shape[1] == num_features:
        print("       Transposing data to (Batch, Time, Features)...")
        x_test = x_test.transpose(0, 2, 1)

    print(f"       X_Test Shape: {x_test.shape} | Samples: {x_test.shape[0]}")

    # --- Normalize ---
    print("[2/4] Normalizing with Pelvis Anchor...")
    x_test = normalize_skeleton(x_test)

    # --- Load model ---
    print(f"[3/4] Loading model: {cfg['output_model']}...")
    model = load_model(model_path)

    # --- Evaluate ---
    print("[4/4] Running model.evaluate()...")
    loss, mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  ┌───────────────────────────────────┐")
    print(f"  │  Loss (MSE) : {loss:.6f}            ")
    print(f"  │  MAE        : {mae:.6f}            ")
    print(f"  └───────────────────────────────────┘")

    # --- Blind Test: 5 random samples ---
    print(f"\n  BLIND TEST — 5 Random Samples")
    print(f"  {'─' * 60}")

    n_samples = x_test.shape[0]
    indices = np.random.choice(n_samples, size=min(5, n_samples), replace=False)
    indices.sort()

    predictions = model.predict(x_test[indices], verbose=0).flatten()

    for i, idx in enumerate(indices):
        doctor_score = y_test[idx]
        ai_guess = predictions[i]
        error = abs(doctor_score - ai_guess)
        print(f"  Repetition #{idx:>4d} | Doctor Score: {doctor_score:.3f} "
              f"| AI Guess: {ai_guess:.3f} | Error: {error:.3f}")

    print(f"  {'─' * 60}")
    print()

    return mae


# ---------------------------------------------------------------------------
#  4. CLI
# ---------------------------------------------------------------------------

def list_exercises():
    """Pretty-print the available exercise keys."""
    print("\nAvailable exercises:")
    print("-" * 55)
    for key, cfg in EXERCISES.items():
        print(f"  {key:<20s}  {cfg['name']}")
    print("-" * 55)
    print(f"\nTotal: {len(EXERCISES)} exercises\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate elderly-focused exercise quality models on held-out fold1 data."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exercise", type=str, help="Evaluate a single exercise (see --list)")
    group.add_argument("--all", action="store_true", help="Evaluate all exercises")
    group.add_argument("--list", action="store_true", help="List available exercises")
    args = parser.parse_args()

    if args.list:
        list_exercises()
        return

    if args.all:
        print(f"\n{'=' * 70}")
        print(f"  BATCH EVALUATION: {len(EXERCISES)} exercises")
        print(f"{'=' * 70}\n")
        results = {}
        for key in EXERCISES:
            mae = evaluate_exercise(key)
            results[key] = f"MAE = {mae:.6f}" if mae is not None else "SKIPPED"

        print("\n" + "=" * 70)
        print("  BATCH EVALUATION SUMMARY")
        print("=" * 70)
        for key, status in results.items():
            name = EXERCISES[key]["name"]
            print(f"  {key:<20s}  {name:<35s}  {status}")
        print("=" * 70)
        return

    # Single exercise
    key = args.exercise
    if key not in EXERCISES:
        print(f"[ERROR] Unknown exercise '{key}'. Use --list to see available options.")
        sys.exit(1)
    evaluate_exercise(key)


if __name__ == "__main__":
    main()
