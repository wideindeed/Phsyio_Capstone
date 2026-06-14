"""
train_new_exercises.py
======================
Parameterized training script for 5 new elderly-focused exercise models.
Architecture and normalization are identical to the existing train.py
(Pelvis Anchor normalization, Bidirectional LSTM).

Usage:
    python train_new_exercises.py --exercise knee_extension   # Train one model
    python train_new_exercises.py --all                       # Train all sequentially
    python train_new_exercises.py --list                      # List available exercises
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from keras.callbacks import ModelCheckpoint

# ---------------------------------------------------------------------------
#  1. EXERCISE CONFIGURATION
# ---------------------------------------------------------------------------

EXERCISES = {
    "knee_extension": {
        "name": "Seated Knee Extension",
        "data_dir": "UIPRMD_reg/SASLR/fold0",  # Standing Active Straight Leg Raise
        "time_steps": 63,
        "num_features": 66,  # 22 joints * 3D
        "output_model": "knee_extension_robust.keras",
    },
    "wall_pushup": {
        "name": "Wall Push-Up",
        "data_dir": "UIPRMD_reg/IL/fold0",  # Inline Lunge
        "time_steps": 77,
        "num_features": 66,
        "output_model": "wall_pushup_robust.keras",
    },
    "calf_raise": {
        "name": "Calf Raise (Heel Raise)",
        "data_dir": "UIPRMD_reg/STS/fold0",  # Sit to Stand
        "time_steps": 88,
        "num_features": 66,
        "output_model": "calf_raise_robust.keras",
    },
    "hip_march": {
        "name": "Seated Hip March",
        "data_dir": "UIPRMD_reg/HS/fold0",  # Hurdle Step
        "time_steps": 69,
        "num_features": 66,
        "output_model": "hip_march_robust.keras",
    },
    "w_raise": {
        "name": "Seated W Raise (Scapular Retraction)",
        "data_dir": "UIPRMD_reg/SSA/fold0",  # Standing Shoulder Abduction
        "time_steps": 74,
        "num_features": 66,
        "output_model": "w_raise_robust.keras",
    },
}

# Root of the project — used to build all paths robustly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
#  2. PELVIS ANCHOR NORMALIZATION (exact copy from train.py)
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

    # Center at Mid-Hip (Joint 0 in UI-PRMD)
    root = data[:, :, 0:1, :]
    data = data - root

    # UI-PRMD Mapping: Joint 14 is Right Hip, Joint 18 is Left Hip
    # We scale by the distance between them (a rigid, unchanging bone)
    left_hip = data[:, :, 18:19, :]
    right_hip = data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)

    # Avoid division by zero
    pelvis_width = np.maximum(pelvis_width, 0.0001)

    # Scale the entire skeleton
    data = data / pelvis_width

    return data.reshape(B, T, F)


# ---------------------------------------------------------------------------
#  3. DATA LOADING
# ---------------------------------------------------------------------------

def load_data(fold_path):
    """Load fold0 train/test splits from .npy files."""
    x_train = np.load(os.path.join(fold_path, "x_train_fold0.npy"))
    y_train = np.load(os.path.join(fold_path, "y_train_fold0.npy"))
    x_test = np.load(os.path.join(fold_path, "x_test_fold0.npy"))
    y_test = np.load(os.path.join(fold_path, "y_test_fold0.npy"))
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
#  4. MODEL BUILDER
# ---------------------------------------------------------------------------

def build_model(time_steps, num_features):
    """Construct the Bidirectional LSTM model (identical to train.py)."""
    model = Sequential([
        Input(shape=(time_steps, num_features)),

        # Widened first layer to capture the complex Center of Gravity shift
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),

        Bidirectional(LSTM(64)),
        Dropout(0.3),

        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Outputs final score between 0 and 1
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


# ---------------------------------------------------------------------------
#  5. TRAINING LOOP FOR A SINGLE EXERCISE
# ---------------------------------------------------------------------------

def train_exercise(key):
    """Full train pipeline for one exercise key."""
    cfg = EXERCISES[key]
    name = cfg["name"]
    time_steps = cfg["time_steps"]
    num_features = cfg["num_features"]
    data_path = os.path.join(BASE_DIR, cfg["data_dir"])
    model_path = os.path.join(BASE_DIR, cfg["output_model"])

    print("=" * 70)
    print(f"  TRAINING: {name} ({key})")
    print(f"  Data    : {data_path}")
    print(f"  Shape   : ({time_steps}, {num_features})")
    print(f"  Output  : {model_path}")
    print("=" * 70)

    # --- Load ---
    if not os.path.isdir(data_path):
        print(f"[ERROR] Data directory not found: {data_path}")
        print("        Run download_data.py first.")
        return False

    print(f"[1/4] Loading {name} data...")
    x_train, y_train, x_test, y_test = load_data(data_path)

    # --- Transpose if needed (Batch, Features, Time) → (Batch, Time, Features) ---
    if x_train.shape[1] == num_features:
        print("       Transposing data to (Batch, Time, Features)...")
        x_train = x_train.transpose(0, 2, 1)
        x_test = x_test.transpose(0, 2, 1)

    print(f"       X_Train Shape: {x_train.shape} | Expected: (Batch, {time_steps}, {num_features})")

    # --- Normalize ---
    print(f"[2/4] Normalizing with Pelvis Anchor...")
    x_train = normalize_skeleton(x_train)
    x_test = normalize_skeleton(x_test)

    # --- Build ---
    print(f"[3/4] Building Bidirectional LSTM model...")
    model = build_model(time_steps, num_features)
    model.summary()

    # --- Train ---
    print(f"[4/4] Starting training (50 epochs, batch_size=16)...")
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_mae',
        save_best_only=True,
        mode='min',
        verbose=1,
    )

    model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint],
    )

    print(f"\n✓ Training Complete. Best model saved as '{cfg['output_model']}'")
    print()
    return True


# ---------------------------------------------------------------------------
#  6. CLI
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
        description="Train elderly-focused exercise quality models."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exercise", type=str, help="Train a single exercise (see --list)")
    group.add_argument("--all", action="store_true", help="Train all exercises sequentially")
    group.add_argument("--list", action="store_true", help="List available exercises")
    args = parser.parse_args()

    if args.list:
        list_exercises()
        return

    if args.all:
        print(f"\n{'='*70}")
        print(f"  BATCH TRAINING: {len(EXERCISES)} exercises")
        print(f"{'='*70}\n")
        results = {}
        for key in EXERCISES:
            ok = train_exercise(key)
            results[key] = "SUCCESS" if ok else "FAILED"

        print("\n" + "=" * 70)
        print("  BATCH TRAINING SUMMARY")
        print("=" * 70)
        for key, status in results.items():
            print(f"  {key:<20s}  {status}")
        print("=" * 70)
        return

    # Single exercise
    key = args.exercise
    if key not in EXERCISES:
        print(f"[ERROR] Unknown exercise '{key}'. Use --list to see available options.")
        sys.exit(1)
    train_exercise(key)


if __name__ == "__main__":
    main()
