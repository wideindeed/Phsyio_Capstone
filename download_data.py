"""
download_data.py
================
Downloads training data from DeepRehabPile for the 5 new elderly-focused exercises.
Uses only stdlib (urllib.request) — no extra pip installs needed.

For each exercise, downloads fold0/ and fold1/ containing:
    x_train_fold{N}.npy, y_train_fold{N}.npy,
    x_test_fold{N}.npy,  y_test_fold{N}.npy

Usage:
    python download_data.py --exercise knee_extension   # Download one exercise
    python download_data.py --all                       # Download all exercises
    python download_data.py --list                      # List available exercises
"""

import argparse
import os
import sys
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
#  1. EXERCISE CONFIGURATION
# ---------------------------------------------------------------------------

EXERCISES = {
    "knee_extension": {
        "name": "Seated Knee Extension",
        "acronym": "SASLR",
    },
    "wall_pushup": {
        "name": "Wall Push-Up",
        "acronym": "IL",
    },
    "calf_raise": {
        "name": "Calf Raise (Heel Raise)",
        "acronym": "STS",
    },
    "hip_march": {
        "name": "Seated Hip March",
        "acronym": "HS",
    },
    "w_raise": {
        "name": "Seated W Raise (Scapular Retraction)",
        "acronym": "SSA",
    },
}

BASE_URL = "https://maxime-devanne.com/datasets/RehabPile/UIPRMD_reg"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOLDS = [0, 1]
FILE_TEMPLATES = [
    "x_train_fold{fold}.npy",
    "y_train_fold{fold}.npy",
    "x_test_fold{fold}.npy",
    "y_test_fold{fold}.npy",
]


# ---------------------------------------------------------------------------
#  2. DOWNLOAD HELPERS
# ---------------------------------------------------------------------------

def download_file(url, dest_path):
    """Download a single file with progress bar using urllib."""
    try:
        # Open connection to get file size (if available)
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            chunk_size = 8192

            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress bar
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        bar_len = 30
                        filled = int(bar_len * downloaded / total_size)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        size_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        print(f"\r       [{bar}] {pct:5.1f}%  "
                              f"({size_mb:.1f}/{total_mb:.1f} MB)", end='', flush=True)
                    else:
                        size_kb = downloaded / 1024
                        print(f"\r       Downloaded {size_kb:.0f} KB...", end='', flush=True)

            print()  # Newline after progress bar
            return True

    except urllib.error.HTTPError as e:
        print(f"\n       [HTTP ERROR {e.code}] {url}")
        if os.path.exists(dest_path):
            os.remove(dest_path)  # Clean up partial file
        return False
    except urllib.error.URLError as e:
        print(f"\n       [CONNECTION ERROR] {e.reason}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False
    except Exception as e:
        print(f"\n       [ERROR] {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


# ---------------------------------------------------------------------------
#  3. DOWNLOAD PIPELINE FOR A SINGLE EXERCISE
# ---------------------------------------------------------------------------

def download_exercise(key):
    """Download all folds and files for a single exercise."""
    cfg = EXERCISES[key]
    acronym = cfg["acronym"]
    name = cfg["name"]

    print("=" * 70)
    print(f"  DOWNLOADING: {name} ({key})")
    print(f"  Dataset    : UIPRMD_reg/{acronym}")
    print("=" * 70)

    total_files = 0
    skipped_files = 0
    downloaded_files = 0
    failed_files = 0

    for fold in FOLDS:
        fold_dir = os.path.join(BASE_DIR, "UIPRMD_reg", acronym, f"fold{fold}")

        # Create directory if it doesn't exist
        os.makedirs(fold_dir, exist_ok=True)
        print(f"\n  fold{fold}/")

        for template in FILE_TEMPLATES:
            filename = template.format(fold=fold)
            dest_path = os.path.join(fold_dir, filename)
            url = f"{BASE_URL}/{acronym}/fold{fold}/{filename}"
            total_files += 1

            # Skip if file already exists
            if os.path.isfile(dest_path):
                size_kb = os.path.getsize(dest_path) / 1024
                print(f"    ✓ {filename:<25s}  (exists, {size_kb:.0f} KB — skipped)")
                skipped_files += 1
                continue

            print(f"    ↓ {filename:<25s}  ← {url}")
            ok = download_file(url, dest_path)
            if ok:
                size_kb = os.path.getsize(dest_path) / 1024
                print(f"       ✓ Saved ({size_kb:.0f} KB)")
                downloaded_files += 1
            else:
                failed_files += 1

    print(f"\n  Summary: {downloaded_files} downloaded, {skipped_files} skipped, {failed_files} failed "
          f"(out of {total_files} total files)")
    print()

    return failed_files == 0


# ---------------------------------------------------------------------------
#  4. CLI
# ---------------------------------------------------------------------------

def list_exercises():
    """Pretty-print available exercises."""
    print("\nAvailable exercises:")
    print("-" * 55)
    for key, cfg in EXERCISES.items():
        print(f"  {key:<20s}  {cfg['name']:<30s}  ({cfg['acronym']})")
    print("-" * 55)
    print(f"\nTotal: {len(EXERCISES)} exercises")
    print(f"Source: {BASE_URL}/\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download DeepRehabPile training data for new exercises."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exercise", type=str, help="Download data for a single exercise (see --list)")
    group.add_argument("--all", action="store_true", help="Download data for all exercises")
    group.add_argument("--list", action="store_true", help="List available exercises")
    args = parser.parse_args()

    if args.list:
        list_exercises()
        return

    if args.all:
        print(f"\n{'=' * 70}")
        print(f"  BATCH DOWNLOAD: {len(EXERCISES)} exercises")
        print(f"  Source: {BASE_URL}/")
        print(f"{'=' * 70}\n")

        results = {}
        for key in EXERCISES:
            ok = download_exercise(key)
            results[key] = "OK" if ok else "ERRORS"

        print("\n" + "=" * 70)
        print("  BATCH DOWNLOAD SUMMARY")
        print("=" * 70)
        for key, status in results.items():
            name = EXERCISES[key]["name"]
            print(f"  {key:<20s}  {name:<30s}  [{status}]")
        print("=" * 70)
        return

    # Single exercise
    key = args.exercise
    if key not in EXERCISES:
        print(f"[ERROR] Unknown exercise '{key}'. Use --list to see available options.")
        sys.exit(1)
    download_exercise(key)


if __name__ == "__main__":
    main()
