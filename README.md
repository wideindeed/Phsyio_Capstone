# PhysioVision — Clinical AI Physiotherapy Assessment Platform

<p align="center">
  <img src="docs/assets/banner.png" alt="PhysioVision Banner" width="100%"/>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img alt="Keras" src="https://img.shields.io/badge/Keras-2.15-D00000?style=flat-square&logo=keras&logoColor=white"/>
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img alt="PyQt5" src="https://img.shields.io/badge/PyQt5-Desktop-41CD52?style=flat-square&logo=qt&logoColor=white"/>
  <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-BlazePose-0097A7?style=flat-square"/>
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows&logoColor=white"/>
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

<p align="center">
  Real-time AI-powered rehabilitation exercise assessment — delivered as an accessible, cloud-connected desktop application.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Exercise Catalogue](#exercise-catalogue)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Client Installation](#client-installation)
  - [Server Deployment](#server-deployment)
  - [Environment Variables](#environment-variables)
- [Training New Models](#training-new-models)
- [API Reference](#api-reference)
- [Model Evaluation](#model-evaluation)
- [Team](#team)
- [Acknowledgements](#acknowledgements)

---

## Overview

PhysioVision is a capstone research project that brings clinical-grade physiotherapy assessment to consumer hardware. Using a standard USB camera and MediaPipe pose estimation, the platform scores exercise form in real time using a suite of Bidirectional LSTM models trained on the [UI-PRMD rehabilitation dataset](https://doi.org/10.3390/data3010002).

The application is built as a **PyQt5 desktop app** with a full HTML/CSS/JavaScript single-page frontend rendered through `QWebEngineView`, communicating with Python via `QWebChannel`. A production-hardened **FastAPI backend** running on a Raspberry Pi provides multi-user authentication, cloud-synced session history, goals, achievements, and structured course progressions.

---

## Features

| Category | Feature |
|---|---|
| **AI Assessment** | Real-time form scoring (0–100) per rep using Bidirectional LSTM |
| **Pose Estimation** | MediaPipe BlazePose — 33 3D world-space landmarks at ~30 fps |
| **Voice Feedback** | Text-to-speech form corrections and rep counts per session |
| **Session History** | Cloud-synced records with per-rep breakdown and pain tracking |
| **Goals** | User-defined targets (reps, score, sessions) with live progress bars |
| **Achievements** | 12 auto-unlocking milestone badges (streaks, rep counts, score thresholds) |
| **Courses** | Structured rehabilitation programmes with sequential step unlock |
| **Analytics** | Per-exercise score timelines with trend visualisation |
| **Clinical Export** | PDF session report generation |
| **AR Overlay** | Hologram projection mode for spatial positioning guidance |
| **Authentication** | Email/password + Google OAuth, JWT access/refresh token rotation |
| **Security** | bcrypt (cost=14), Cloudflare Turnstile, slowapi rate limiting, rotating audit logs |

---

## Exercise Catalogue

| # | Exercise | Dataset | Input Shape | Focus |
|---|---|---|---|---|
| 1 | Deep Squat | UI-PRMD DS | `(1, 81, 66)` | Knee & hip mobility |
| 2 | Sit to Stand | UI-PRMD STS | `(1, 88, 66)` | Geriatric fall-risk assessment |
| 3 | Push-Up | UI-PRMD TSPU | `(1, 60, 66)` | Upper body & core stabilisation |
| 4 | Bicep Curl | Proprietary CSV | `(1, 40, 99)` | Elbow ROM & cheat-pattern classification |
| 5 | Lateral Raise | UI-PRMD SSA | `(1, 74, 66)` | Shoulder abduction & symmetry |
| 6 | Knee Extension | UI-PRMD SASLR | `(1, 63, 66)` | Seated leg raise & ROM |
| 7 | Wall Push-Up | UI-PRMD IL | `(1, 77, 66)` | Shoulder & upper limb mobility |
| 8 | Hip March | UI-PRMD HS | `(1, 69, 66)` | Hip flexor mobility & gait rehab |
| 9 | Shoulder Extension | UI-PRMD SSE | `(1, 67, 66)` | Posture correction |
| 10 | Shoulder Scaption | UI-PRMD SSS | `(1, 66, 66)` | Rotator cuff rehabilitation |

> **Note:** Exercises 1–5 are original models. Exercises 6–10 were trained as part of this project using the UI-PRMD dataset.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DESKTOP APPLICATION                           │
│                                                                      │
│   USB Camera → MediaPipe BlazePose → PRMD Feature Extraction        │
│         (33 landmarks)              (22 joints × 3D = 66 feat/frame)│
│                         ↓                                            │
│              cv2.resize → Fixed-T Frame Buffer                       │
│                         ↓                                            │
│            Keras BiLSTM Model → Form Score (0–100)                  │
│                         ↓                                            │
│            PyQt5 + QWebEngineView ←→ HTML/CSS/JS SPA               │
│                  (QWebChannel Bridge)                                │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ HTTPS (JWT Bearer)
                         ↓
┌──────────────────────────────────────────────────────────────────────┐
│                   CLOUD BACKEND  (Raspberry Pi 24/7)                │
│                                                                      │
│   FastAPI → SQLite (WAL mode)                                        │
│   JWT auth · bcrypt · Rate limiting · Cloudflare Turnstile          │
│   Tables: users · sessions · goals · achievements ·                 │
│           course_step_log · refresh_tokens · failed_logins           │
└──────────────────────────────────────────────────────────────────────┘
```

### Shared BiLSTM Model Architecture

All regression models share the same architecture, differing only in the `T` (time steps) dimension:

```
Input (1, T, 66)
    → Bidirectional LSTM (128 units, return_sequences=True)
    → Dropout (0.3)
    → Bidirectional LSTM (64 units)
    → Dropout (0.3)
    → Dense (32, ReLU)
    → Dense (1, Sigmoid)          # Output: quality score 0.0–1.0
```

The Bicep Curl model uses a standard LSTM classifier with 5-class softmax output (Drag · Half · Heave · Perfect · Swing).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Desktop shell | Python 3.13 · PyQt5 · QWebEngineView |
| Frontend SPA | HTML5 · CSS3 · Vanilla JavaScript |
| Python↔JS bridge | QWebChannel |
| Pose estimation | MediaPipe BlazePose |
| AI models | Keras 2.15 · TensorFlow 2.15 |
| Video processing | OpenCV (cv2) |
| Backend API | FastAPI · Uvicorn |
| Database | SQLite (WAL mode, FK enforced) |
| Auth | JWT (HS256) · bcrypt (cost=14) · Google OAuth 2.0 |
| Email | Resend API |
| Bot protection | Cloudflare Turnstile |
| Rate limiting | slowapi (per-IP, tiered per endpoint) |
| Server hardware | Raspberry Pi (24/7 deployment) |

---

## Project Structure

```
PhysioVision/
│
├── main.py                          # Entry point (PyInstaller target, delegates to src/dashboard.py)
├── config.example.py                # Copy to config.py and set API_URL for your deployment
│
├── src/
│   ├── engine.py                    # AI pipeline, camera loop, VisionWorker (QThread)
│   ├── dashboard.py                 # PyQt5 Bridge (QWebChannel), launch_app(), MJPEG server
│   ├── auth.py                      # Login window (PyQt5), Google OAuth handler
│   ├── goals.py                     # Client-side GoalTracker & achievement definitions
│   ├── groq_feedback.py             # Groq LLaMa-3.1 knowledge-grounded feedback layer
│   ├── feedback_knowledge_base.py   # Curated per-mistake feedback cue lookup table
│   ├── latency_logger.py            # Pipeline/API latency instrumentation
│   ├── splash_player.py             # Startup splash screen
│   └── *_analyzer.py                # Per-exercise state machine classes
│       ├── knee_extension_analyzer.py
│       ├── wall_pushup_analyzer.py
│       ├── hip_march_analyzer.py
│       ├── shoulder_extension_analyzer.py
│       └── shoulder_scaption_analyzer.py
│
├── frontend/
│   ├── index.html                   # SPA entry point (Hub · Analysis · Records · Analytics
│   │                                 #                  Goals · Achievements · Courses · Settings)
│   ├── app.js                       # Frontend logic, QWebChannel bindings, chart rendering
│   └── styles.css                   # Design system (dark sidebar, clinical blue, sharp geometry)
│
├── scripts/
│   ├── train.py                     # Per-exercise BiLSTM training script
│   ├── train_curl_model.py          # Bicep Curl cheat-pattern classifier training script
│   ├── download_deeprehabpile.py    # Downloads UI-PRMD fold data from RehabPile
│   ├── evaluate_fold.py             # Cross-fold evaluation: MAE, RMSE, R² per model
│   └── normalization_ablation.py    # Pelvis-anchor normalization ablation experiment
│
├── models/                          # Trained .keras model files (committed)
├── UIPRMD_STS/                      # UI-PRMD Sit-to-Stand dataset folds (sample folds committed)
│
└── requirements.txt
```

> The FastAPI backend (multi-user auth, cloud session sync, goals/achievements) runs on a separate Raspberry Pi deployment and is **not included in this repository**. The API Reference section below documents that backend's contract for anyone standing up their own instance.

---

## Getting Started

### Prerequisites

- Python 3.13
- Windows 10/11 (PyQt5 + QWebEngineView is tested on Windows)
- A USB or built-in webcam
- Access to the PhysioVision API server (or run your own — see [Server Deployment](#server-deployment))

### Client Installation

```bash
# 1. Clone the repository
git clone https://github.com/wideindeed/Physiology-LLM-Capstone.git
cd Physiology-LLM-Capstone

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure the API URL
copy config.example.py config.py   # Windows
# then edit config.py and set API_URL to your deployment

# 5. Trained .keras models are already included under models/
# (see Training New Models to retrain from scratch)

# 6. Launch the application
python main.py
```

### Server Deployment

The API server is designed to run on a **Raspberry Pi** (or any Linux host) behind a reverse proxy (Cloudflare Tunnel recommended).

```bash
# On the Raspberry Pi:

# 1. Clone and set up
git clone https://github.com/wideindeed/Physiology-LLM-Capstone.git
cd Physiology-LLM-Capstone
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment variables (see below)
cp .env.example .env
nano .env

# 3. Run the server
uvicorn api_server:app --host 127.0.0.1 --port 8000

# Or as a systemd service (recommended for 24/7 operation):
sudo systemctl start physiovision
```

### Environment Variables

Create a `.env` file in the project root for the API server. **Never commit this file.**

```env
# JWT secrets — generate with: python -c "import secrets; print(secrets.token_hex(64))"
JWT_ACCESS_SECRET=your_access_secret_here
JWT_REFRESH_SECRET=your_refresh_secret_here

# Email delivery (https://resend.com)
RESEND_API_KEY=re_xxxxxxxxxxxx

# Google OAuth (https://console.cloud.google.com)
GOOGLE_CLIENT_ID=your_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_client_secret

# Cloudflare Turnstile (https://dash.cloudflare.com)
TURNSTILE_SECRET_KEY=your_turnstile_secret

# Public URLs
PUBLIC_SERVER_URL=https://api.your-domain.com
FRONTEND_URL=https://your-domain.com
```

For the desktop client, only the API URL is required, set in `config.py` (copied from `config.example.py`):

```python
API_URL = "https://api.your-domain.com"
```

---

## Training New Models

Use `scripts/train.py` to retrain a per-exercise BiLSTM model from scratch using the UI-PRMD dataset.

```bash
# 1. Download the required dataset folds
python scripts/download_deeprehabpile.py

# 2. Train a model
python scripts/train.py

# Output: a .keras file per exercise, written to models/
```

Bicep Curl uses a separate classifier trained on a proprietary CSV dataset via `scripts/train_curl_model.py`.

**Training configuration:**

| Parameter | Value |
|---|---|
| Architecture | BiLSTM(128) → BiLSTM(64) → Dense(32) → Dense(1, sigmoid) |
| Optimizer | Adam |
| Loss | Mean Squared Error (MSE) |
| Epochs | 100 (max) with EarlyStopping patience=25 |
| Batch size | 8 |
| Dropout | 0.3 after each BiLSTM layer |
| Normalisation | Pelvis-anchor (root = mid-hip, scale = pelvis width) |
| Resampling | `cv2.resize` bilinear → fixed T frames per exercise |
| Framework | Keras 2.15 · TensorFlow 2.15 |

To add a new exercise, see the integration pattern in any existing `src/*_analyzer.py` file.

---

## API Reference

The FastAPI server exposes the following endpoints. All protected routes require a `Bearer` token in the `Authorization` header.

### Authentication

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/register` | Public | Create a new account (sends verification email) |
| `GET` | `/verify` | Public | Verify email address via token link |
| `POST` | `/login` | Public | Authenticate and receive JWT token pair |
| `POST` | `/refresh` | Public | Rotate refresh token, receive new access token |
| `POST` | `/logout` | Public | Revoke refresh token |
| `GET` | `/auth/google/login` | Public | Initiate Google OAuth flow |
| `GET` | `/auth/google/callback` | Public | Google OAuth redirect handler |
| `POST` | `/complete_profile` | Public | Complete profile after Google sign-up |

### Sessions & Data

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/log_session` | ✅ | Log a completed exercise session |
| `GET` | `/get_history` | ✅ | Retrieve session history (last 500) |
| `GET` | `/dashboard_data` | ✅ | Aggregated stats for the dashboard |

### Goals

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/get_goals` | ✅ | Fetch active goals with computed progress |
| `POST` | `/set_goal` | ✅ | Create a new goal |
| `DELETE` | `/delete_goal/{id}` | ✅ | Delete a goal by ID |

### Achievements

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/get_achievements` | ✅ | Full achievement catalogue with locked/unlocked status |

### Courses

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/get_course_progress` | ✅ | Fetch all completed course steps |
| `POST` | `/log_course_step` | ✅ | Log or update a completed step (UPSERT) |
| `DELETE` | `/reset_course/{course_id}` | ✅ | Clear all progress for a course |

### System

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | Public | Server status check |

---

## Model Evaluation

`scripts/evaluate_fold.py` computes MAE, RMSE, and R² for a held-out fold. It ships pointed at the Sit-to-Stand model/fold as an example — edit `MODEL_PATH` and `TEST_DIR` at the top of the script to evaluate a different exercise/fold.

```bash
python scripts/evaluate_fold.py
```

### Results

| Exercise | Test Samples | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---|---|---|
| Knee Extension | 146 | 0.024 | 0.040 | **0.883** |
| Lateral Raise | 126 | 0.043 | 0.073 | **0.881** |
| Hip March | 110 | 0.046 | 0.071 | **0.871** |
| Shoulder Extension | 126 | 0.044 | 0.068 | **0.836** |
| Sit to Stand | 104 | 0.024 | 0.033 | **0.805** |
| Wall Push-Up | 102 | 0.083 | 0.105 | 0.782 |
| Shoulder Scaption | 108 | 0.078 | 0.110 | 0.671 |
| Deep Squat † | 180 | 0.028 | 0.043 | 0.543 |
| **Average** | **922** | **0.046** | **0.068** | **0.784** |
| Bicep Curl ‡ | 539 | — | — | **100% acc.** |

> † Deep Squat uses a pre-existing smaller architecture (110k parameters vs 368k for all other models), which accounts for the lower R².  
> ‡ Bicep Curl is a 5-class cheat-pattern classifier evaluated on the full proprietary dataset (classification accuracy, not regression).

*MAE scale: 0–1 (labels are normalised quality scores). Lower MAE = smaller average error.*

---

## Team

| Name | Role |
|---|---|
| **Mahmood Muwafi** | Lead Developer |
| **Afzal M. Harish** | AI & Model Development |
| **Abdulsalam Alturk** | Backend & Systems |

**British University in Dubai** · Computer Science & Engineering · Capstone 2026

---

## Acknowledgements

- [UI-PRMD Dataset](https://doi.org/10.3390/data3010002) — Vakanski, A. et al. (2018). University of Idaho Physical Rehabilitation Movement Data. *Data, 3*(1), 2.
- [MediaPipe](https://github.com/google/mediapipe) — Lugaresi, C. et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. *arXiv:1906.08172*
- [RehabPile Benchmark](https://msd-irimas.github.io/pages/DeepRehabPile/) — Ismail-Fawaz, A. et al. (2026). A Standardized Benchmark for Skeleton-Based Rehabilitation Assessment. *IEEE FG 2026*

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
