"""
main.py — PhysioVision entry point
===================================
PyInstaller targets this file. All it does is call the real
application launcher from dashboard.py.

Do not add logic here — keep it a clean single-line delegate
so PyArmor and PyInstaller have the simplest possible target.
"""

import multiprocessing
import sys
import os

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Ensure the src/ package is importable whether running from source
    # or from a PyInstaller --onefile bundle (where _MEIPASS is the tmp dir).
    _base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    _src = os.path.join(_base, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

    from dashboard import launch_app
    launch_app()