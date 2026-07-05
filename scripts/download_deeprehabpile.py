"""
Downloads the UI-PRMD regression folds (from DeepRehabPile's mirror) needed
to run the normalization ablation for every PhysioVision exercise that was
trained on UI-PRMD data. Bicep Curl is skipped (hand-collected dataset) and
Push-Up is skipped (not part of the UI-PRMD-trained set per the report).

Source: https://msd-irimas.github.io/pages/DeepRehabPile/
Mirror:  https://maxime-devanne.com/datasets/RehabPile/UIPRMD_reg/<ABBR>/foldN/

Mapping notes:
  - Deep Squat, Sit to Stand, Shoulder Extension, Shoulder Scaption match
    UI-PRMD exercises by name (DS, STS, SSE, SSS).
  - Lateral Raise -> SSA (Standing Shoulder Abduction), confirmed by the
    "UIPRMD_REG_SSA training pipeline" comment in src/engine.py.
  - Knee Extension -> SASLR, Wall Push-Up -> IL (Inline Lunge), Hip March
    -> HS (Hurdle Step): confirmed not by name but by an exact match
    between each dataset's per-exercise time-step count (fold array shape)
    and the hardcoded TIME_STEPS constant in each exercise's analyzer file
    (knee_extension_analyzer.py TIME_STEPS=63 == SASLR T=63;
    wall_pushup_analyzer.py TIME_STEPS=77 == IL T=77, NOT SSIER's T=74;
    hip_march_analyzer.py TIME_STEPS=69 == HS T=69). This is a reliable
    fingerprint since every UI-PRMD sub-exercise has a distinct frame count.
"""

import os
import urllib.request
import urllib.error

BASE_URL = "https://maxime-devanne.com/datasets/RehabPile/UIPRMD_reg"
OUT_ROOT = os.path.join(os.path.dirname(__file__), "..")

# exercise_key -> UI-PRMD abbreviation
EXERCISE_TO_ABBR = {
    "squat": "DS",
    "sts": "STS",
    "sh_ext": "SSE",
    "sh_scap": "SSS",
    "lateral": "SSA",
    "knee_ext": "SASLR",   # confirmed by TIME_STEPS=63 fingerprint match
    "wall_push": "IL",     # confirmed by TIME_STEPS=77 fingerprint match
    "hip_march": "HS",     # confirmed by TIME_STEPS=69 fingerprint match
}

FILE_KINDS = ["x_train", "y_train", "s_train", "x_test", "y_test", "s_test"]
MAX_FOLDS = 5


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"  skip (exists): {dest_path}")
        return True
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  downloaded: {dest_path}")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def download_exercise(exercise_key, abbr):
    print(f"\n=== {exercise_key} ({abbr}) ===")
    local_dir = os.path.join(OUT_ROOT, f"UIPRMD_{abbr}")
    any_fold_found = False

    for fold in range(MAX_FOLDS):
        fold_dir = os.path.join(local_dir, f"fold{fold}")
        first_file_url = f"{BASE_URL}/{abbr}/fold{fold}/x_train_fold{fold}.npy"
        probe_dest = os.path.join(fold_dir, f"x_train_fold{fold}.npy")

        os.makedirs(fold_dir, exist_ok=True)
        found = download_file(first_file_url, probe_dest)
        if not found:
            os.rmdir(fold_dir) if not os.listdir(fold_dir) else None
            if fold == 0:
                print(f"  no data found at {BASE_URL}/{abbr}/fold0/ -- check abbreviation")
            break

        any_fold_found = True
        for kind in FILE_KINDS[1:]:
            url = f"{BASE_URL}/{abbr}/fold{fold}/{kind}_fold{fold}.npy"
            dest = os.path.join(fold_dir, f"{kind}_fold{fold}.npy")
            download_file(url, dest)

    if not any_fold_found:
        print(f"  WARNING: nothing downloaded for {exercise_key} ({abbr})")


def main():
    for exercise_key, abbr in EXERCISE_TO_ABBR.items():
        download_exercise(exercise_key, abbr)

    print("\nDone. Bicep Curl skipped (hand-collected dataset).")
    print("Push-Up skipped (not part of the UI-PRMD-trained set).")


if __name__ == "__main__":
    main()
