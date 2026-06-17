import os
import sys

# Must be set before any AI/GPU library is imported
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']   = '-1'
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"


# PhysioVision: make src/ importable from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from dashboard import run_application

if __name__ == "__main__":
    run_application()