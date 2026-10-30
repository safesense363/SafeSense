# SafeSense

## Real-Time Human Activity Recognition & Risk Detection System

SafeSense is an AI-powered computer vision system designed to recognize
human activities from video streams and identify potentially risky
activities in real time.

The system combines computer vision, human pose/skeleton representation,
deep learning-based temporal activity recognition, risk assessment,
real-time inference and automated alert generation.

## Live Webcam Pose and Activity

Install the runtime dependencies, then start the live camera from the project
root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --camera 0
```

The native OpenCV window draws the YOLO pose and labels the current activity
using a pretrained Slow R50 AVA action model. Press **Q** or **Esc** while the
window is focused to stop. The first run may download model weights. By
default, `--device auto` uses CUDA when available: it selects the larger
`yolo26s-pose.pt` pose model and moves AVA inference to the GPU. CPU-only
systems use `yolo26n-pose.pt` and CPU inference. Use `--device cpu` to force
CPU, or `--device cuda` to require CUDA. The activity prediction is updated
periodically from short clips. The first activity label requires 30 detected
person frames; on a CPU-only machine this can take noticeable time. Labels and
scores are not calibrated probabilities or a guarantee of correct activity
recognition.

For headless inference use `--no-display`. To select another local pose model,
pass `--pose-model path\to\model.pt`. If the webcam cannot be opened, check
Windows camera privacy permissions, close other apps using the camera, or try
another index with `--camera 1`.

---

## Problem Statement

Traditional surveillance systems primarily depend on human monitoring
and predefined rule-based mechanisms. Continuous manual monitoring is
difficult to scale and may result in delayed detection of potentially
dangerous events.

SafeSense aims to provide an intelligent monitoring layer capable of
understanding human activities from video and identifying activities
that may require attention.

---

## Objectives

- Detect humans from video streams.
- Extract human pose/skeleton information.
- Recognize human activities using deep learning.
- Identify potentially risky activities.
- Assign risk levels to detected events.
- Generate real-time alerts.
- Provide a monitoring dashboard.
- Maintain an event and alert history.
- Provide measurable model and system performance.

---

## System Pipeline

Camera / Video
      ↓
Frame Processing
      ↓
Human Detection / Pose Estimation
      ↓
Skeleton & Motion Features
      ↓
Activity Recognition
      ↓
Risk Detection Engine
      ↓
Alert Generation
      ↓
Dashboard & Event Logging

---

## Technology Stack

### AI / Machine Learning
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn

### Computer Vision
- OpenCV
- Human Pose Estimation
- Skeleton-based representations

### Backend
- FastAPI
- Python

### Frontend
- To be finalized during implementation

### Database
- To be finalized based on deployment requirements

### MLOps
- Git
- GitHub
- DVC
- MLflow

### Deployment
- Docker
- Cloud deployment

---