# Physio-Vision: Enterprise Biomechanical Analysis Engine

---

## Project Overview

**Physio-Vision** is a high-performance computer vision application designed to analyze human movement mechanics in real time. Unlike traditional fitness trackers that rely solely on 2D pose estimation or opaque, "black box" AI classification, this system employs a **Hybrid Neuro-Symbolic Architecture**.

It combines **MediaPipe's neural network–based skeletal perception** with a **strict 3D Vector Calculus Engine** for deterministic geometric safety enforcement.

The current build is optimized for the **Deep Squat** movement, capable of detecting subtle biomechanical faults such as **thoracic rounding**, **excessive trunk lean**, and **valgus collapse** using a standard webcam. It also features a **procedural Augmented Reality (AR) positioning system** to standardize subject placement and a **multi-threaded GUI** for lag-free visualization.

---

## Key Features

### 3D Depth Inference

* Utilizes **MediaPipe World Landmarks** to calculate true 3D joint angles.
* Mitigates perspective errors caused by subject rotation or camera distance.

### Holographic AR Guidance

* Implements a procedural **"Reactor Ring" targeting system**.
* Projects perspective-corrected floor guides to standardize subject distance and orientation.

### Biomechanical Fault Detection

* **Thoracic Kyphosis Proxy**
  Detects upper back rounding by analyzing 3D shoulder protraction vectors.

* **Trunk Flexion Monitor**
  Measures spinal lean relative to gravity, dynamically calibrated for varying user anthropometry.

* **Kinematic State Machine**
  Uses angular thresholds to accurately track concentric and eccentric movement phases.

### Live Engineering Console

* Floating debugging interface.
* Allows real-time adjustment of sensitivity thresholds (e.g., squat depth, critical lean angle) during active sessions.

### Longitudinal Telemetry

* Automatically serializes session metrics into structured patient history logs.
* Tracks repetition counts and form scores across sessions.

---

## System Architecture

The engine operates on a **concurrent, two-layer pipeline**:

### Perception Layer (Neural)

* A dedicated **Vision Worker thread** manages camera I/O.
* Runs MediaPipe inference at **30+ FPS**.
* Outputs normalized 3D skeletal landmark data.

### Logic Layer (Symbolic)

* A parallel **vector physics engine** evaluates the live skeleton against deterministic safety constraints.
* Example checks:

  * `Is spine angle > 40°?`
  * `Is knee valgus vector exceeding tolerance?`
* Triggers **asynchronous audio feedback** without blocking the vision pipeline.

---

## Installation

### Prerequisites

* Python **3.10** or **3.11**
* Webcam

### Dependencies

Install required packages using `pip`. **NumPy version is strictly constrained** to maintain TensorFlow compatibility.

```bash
pip install opencv-contrib-python mediapipe PyQt5 pyttsx3 qfluentwidgets
pip install "numpy<2.0.0" --force-reinstall
```

> **Note:** If version conflicts arise, the following versions are recommended for stability:
>
> * `tensorflow==2.15.0`
> * `mediapipe==0.10.9`

---

## Usage

### Configuration

Open the main script `test.py` and update the `GUIDE_PATH` variable in the `AppState` class to reference your squat guide video:

```python
class AppState:
    # ...
    GUIDE_PATH = "C:\\Path\\To\\Reference\\squat_guide.mp4"
```

### Execution

Run the application from the terminal:

```bash
python test.py
```

---

## Operation Flow

1. **AR Alignment**
   Enable *Holographic Guidance* in Settings. Align feet with the projected floor target until the status indicator turns **Gold (LOCKED)**.

2. **Calibration**
   The system performs a brief static analysis to tare biometric measurements.

3. **Session**
   Perform squats normally. The engine automatically:

   * Tracks repetitions
   * Evaluates form
   * Provides real-time audio cues (e.g., *"Keep chest up"*)

4. **Review**
   Navigate to *Patient Records* to view generated session reports and longitudinal metrics.

---

## Current Limitations (Work in Progress)

* **View Angle Dependency**
  Optimized for a **side-profile view**. Front-facing or oblique angles may reduce spinal flexion accuracy.

* **Load Blindness**
  The physics engine estimates work using body weight only. External loads (e.g., barbells) are not yet modeled.

* **Lighting Sensitivity**
  Extreme low-light conditions may introduce depth jitter, impacting rounding detection stability.

---

## Disclaimer

This software is provided **for educational and research purposes only**. It is **not a medical device**.

Users should consult a qualified healthcare professional before beginning any new exercise program. The developers assume **no responsibility for injuries** sustained while using this software.
