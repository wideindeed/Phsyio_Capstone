# Physio-Vision: Hybrid Biomechanical Analysis Engine

**Version:** 0.8.0 (Prototype)  
**Status:** Active Development

## Project Overview
Physio-Vision is a computer vision application designed to analyze human movement mechanics in real-time. Unlike traditional fitness trackers that rely solely on 2D pose estimation or "black box" AI classification, this system utilizes a **Hybrid Neuro-Symbolic Architecture**. It combines a Long Short-Term Memory (LSTM) neural network for temporal motion analysis with strict 3D vector calculus for geometric safety enforcement.

The current build is optimized for the **Deep Squat** movement, capable of detecting subtle biomechanical faults such as thoracic rounding, excessive trunk lean, and stance asymmetry using standard webcam input.

## Key Features

* **3D Depth Inference:** Utilizes MediaPipe World Landmarks to calculate true 3D angles, mitigating perspective errors caused by user rotation or camera distance.
* **Hybrid Scoring System:** Merges AI-predicted "Smoothness" scores with penalty-based logic derived from physical heuristics.
* **Biomechanical Fault Detection:**
    * **Thoracic Kyphosis Proxy:** Detects upper back rounding by analyzing 3D shoulder protraction vectors.
    * **Trunk Flexion Monitor:** Measures spinal lean relative to gravity, calibrated for users with longer anthropometry (femur/torso length).
    * **Stance Symmetry:** Detects "staggered" or uneven foot placement.
* **Memory-Based Grading:** Tracks the maximum error observed during the eccentric (lowering) phase of a rep and applies the penalty to the final grade, ensuring momentary faults are not ignored.

## System Architecture

The engine operates on a two-layer pipeline:

1.  **The Perception Layer (AI):** An LSTM network processes an 81-frame buffer of normalized skeletal data to judge the tempo, fluidity, and overall pattern of the movement.
2.  **The Logic Layer (Math):** A symbolic physics engine runs parallel to the AI, checking the live skeleton against defined safety thresholds (e.g., *Is the spine angle > 40 degrees?*).

## Installation

### Prerequisites
* Python 3.9 or higher
* Webcam

### Dependencies
Install the required packages using pip:

```bash
pip install opencv-python numpy tensorflow mediapipe
```

*Note: If you encounter specific version conflicts, `tensorflow==2.15.0` and `mediapipe==0.10.9` are the recommended stable versions for this build.*

## Usage

1.  **Configuration:** Open the main script and update the user profile constants at the top of the file to match the subject:

```python
USER_HEIGHT_CM = 193.0
USER_WEIGHT_KG = 84.0
```

2.  **Execution:** Run the script via terminal:

```bash
python main.py
```

3.  **Operation Flow:**
    * **Calibration:** Stand in a strict side-profile view. Remain still until the green bar fills.
    * **Warmup:** The system buffers frames for the AI engine.
    * **Session:** Perform squats. The system will auto-count reps and provide real-time feedback (e.g., "Keep Chest Up"). Press 'Q' to exit.

## Current Limitations (Work in Progress)

* **View Angle:** The current heuristic model strictly requires a **Side Profile View**. Front-facing or diagonal views will result in inaccurate physics calculations.
* **Load Blindness:** The physics engine calculates work (Joules) based on body weight only; it cannot currently account for external barbell loads.
* **Lighting Sensitivity:** Low-light environments may introduce jitter in the depth tracking, affecting the stability of the rounding detection.

## Disclaimer

This software is for educational and research purposes only. It is not a medical device. Users should consult a healthcare professional before beginning any new exercise regime. The developers are not responsible for injuries sustained while using this software.
