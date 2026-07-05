"""
Normalization ablation: re-runs every UI-PRMD-trained PhysioVision model on
its held-out test fold WITH vs WITHOUT the pelvis/spine/shoulder-anchor
normalization step, to measure how much the normalization actually
contributes to R²/MAE (Table II currently only reports the "with
normalization" numbers).

For each exercise:
  - "with normalization"    = exact same centering + bone-length scaling the
                               model was trained with (see MODELS below).
  - "without normalization" = raw skeleton coordinates, reshaped only
                               (no centering, no scaling).

Held-out fold: fold0's OWN test split (x_test_fold0.npy). NOTE: fold2's test
split (used by scripts/evaluate_fold.py) is NOT actually held out -- it is a
verified exact subset of fold0's train split (standard k-fold CV file
structure), so it was rejected after a data-leakage check. See
ablation_results.txt for the full verification writeup.
Results are printed and appended to ablation_results.txt at the repo root.
"""

import os
import json
from datetime import datetime

import numpy as np
from keras.models import load_model
from sklearn.metrics import r2_score, mean_absolute_error

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
RESULTS_PATH = os.path.join(REPO_ROOT, "ablation_results.txt")
# IMPORTANT: these are k-fold CV files -- fold0's train split is the UNION of
# folds 1-4's test splits (verified: every fold2 test sample is an exact
# duplicate inside fold0's train set). The only split that is genuinely
# disjoint from fold0's train data is fold0's OWN test split. Do not use any
# other fold's test split as a "held-out" set for a fold0-trained model.
TEST_FOLD = 0

PELVIS_ROOT_JOINT = 0
LEFT_HIP_JOINT = 18
RIGHT_HIP_JOINT = 14
SPINE_TOP_JOINT = 2
LEFT_SHOULDER_JOINT = 6
RIGHT_SHOULDER_JOINT = 10


def center_at_root(data):
    root = data[:, :, PELVIS_ROOT_JOINT:PELVIS_ROOT_JOINT + 1, :]
    return data - root


def scale_by_pelvis_width(data):
    left_hip = data[:, :, LEFT_HIP_JOINT:LEFT_HIP_JOINT + 1, :]
    right_hip = data[:, :, RIGHT_HIP_JOINT:RIGHT_HIP_JOINT + 1, :]
    width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    return data / np.maximum(width, 0.0001)


def scale_by_spine_length(data):
    spine_top = data[:, :, SPINE_TOP_JOINT:SPINE_TOP_JOINT + 1, :]
    length = np.linalg.norm(spine_top, axis=3, keepdims=True)
    return data / np.maximum(length, 0.0001)


def scale_by_shoulder_width(data):
    l_sh = data[:, :, LEFT_SHOULDER_JOINT:LEFT_SHOULDER_JOINT + 1, :]
    r_sh = data[:, :, RIGHT_SHOULDER_JOINT:RIGHT_SHOULDER_JOINT + 1, :]
    width = np.linalg.norm(l_sh - r_sh, axis=3, keepdims=True)
    return data / np.maximum(width, 0.0001)


def normalize_pelvis_width(data):
    return scale_by_pelvis_width(center_at_root(data))


def normalize_spine_length(data):
    return scale_by_spine_length(center_at_root(data))


def normalize_shoulder_width(data):
    return scale_by_shoulder_width(center_at_root(data))


def normalize_none(data):
    return data


# exercise_key -> (UI-PRMD folder abbr, model filename, normalize_fn, display name)
MODELS = {
    "squat":     ("DS",    "deep_squat_robust.keras",         normalize_spine_length,   "Deep Squat"),
    "sts":       ("STS",   "sit_to_stand_robust.keras",       normalize_pelvis_width,   "Sit to Stand"),
    "lateral":   ("SSA",   "w_raise_robust.keras",            normalize_pelvis_width,   "Lateral Raise"),
    "knee_ext":  ("SASLR", "knee_extension_robust.keras",     normalize_pelvis_width,   "Knee Extension"),
    "wall_push": ("IL",    "wall_pushup_robust.keras",        normalize_pelvis_width,   "Wall Push-Up"),
    "hip_march": ("HS",    "hip_march_robust.keras",          normalize_pelvis_width,   "Hip March"),
    "sh_ext":    ("SSE",   "shoulder_extension_robust.keras", normalize_pelvis_width,   "Shoulder Extension"),
    "sh_scap":   ("SSS",   "shoulder_scaption_robust.keras",  normalize_pelvis_width,   "Shoulder Scaption"),
}


def load_fold(abbr, fold):
    fold_dir = os.path.join(REPO_ROOT, f"UIPRMD_{abbr}", f"fold{fold}")
    x = np.load(os.path.join(fold_dir, f"x_test_fold{fold}.npy"))
    y = np.load(os.path.join(fold_dir, f"y_test_fold{fold}.npy"))
    return x, y


def reshape_to_joints(x):
    # Stored as (Batch, Features, Time) -> transpose to (Batch, Time, Features)
    if x.shape[1] == 66:
        x = x.transpose(0, 2, 1)
    B, T, F = x.shape
    J = F // 3
    return x.reshape(B, T, J, 3), (B, T, F)


def evaluate(model, x_flat, y):
    preds = model.predict(x_flat, verbose=0).reshape(-1)
    y = y.reshape(-1)
    return r2_score(y, preds), mean_absolute_error(y, preds)


def run_ablation():
    lines = []
    header = f"NORMALIZATION ABLATION — {datetime.now().isoformat(timespec='seconds')}"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append(f"{'Exercise':<20}{'Source':<8}{'R2 (norm)':<12}{'MAE (norm)':<12}"
                 f"{'R2 (raw)':<12}{'MAE (raw)':<12}{'Delta R2':<10}")

    for key, (abbr, model_file, normalize_fn, display_name) in MODELS.items():
        model_path = os.path.join(MODEL_DIR, model_file)
        try:
            fold_dir = os.path.join(REPO_ROOT, f"UIPRMD_{abbr}", f"fold{TEST_FOLD}")
            if not os.path.isdir(fold_dir):
                print(f"[SKIP] {display_name}: no fold{TEST_FOLD} data found in UIPRMD_{abbr}")
                continue

            x_raw, y = load_fold(abbr, TEST_FOLD)
            data, (B, T, F) = reshape_to_joints(x_raw)

            model = load_model(model_path)

            x_norm = normalize_fn(data).reshape(B, T, F)
            r2_norm, mae_norm = evaluate(model, x_norm, y)

            x_none = normalize_none(data).reshape(B, T, F)
            r2_raw, mae_raw = evaluate(model, x_none, y)

            delta = r2_norm - r2_raw
            row = (f"{display_name:<20}{abbr:<8}{r2_norm:<12.4f}{mae_norm:<12.4f}"
                   f"{r2_raw:<12.4f}{mae_raw:<12.4f}{delta:<10.4f}")
            print(row)
            lines.append(row)

        except Exception as e:
            msg = f"[ERROR] {display_name} ({abbr}): {e}"
            print(msg)
            lines.append(msg)

    lines.append("")
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")

    print(f"\nResults appended to {RESULTS_PATH}")


if __name__ == "__main__":
    run_ablation()
