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

## Live Camera Demo

The live camera pipeline opens a native OpenCV window; the browser dashboard
does not access or start the webcam. The default activity recognizer is the
pretrained PyTorchVideo Slow R50 detector, fine-tuned on AVA person-action
labels. Its labels include standing, walking, sitting, running, falling, lying
down, and other AVA actions. First startup downloads the pretrained weights;
CPU inference is slower than the local GRU and action scores are not calibrated
probabilities. In this workspace, a real-camera action prediction took about
1.3 seconds; performance depends on the CPU and camera workload.

Install dependencies if needed, then start the API in one PowerShell terminal
if you want events to appear on the dashboard:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload
```

In a second terminal, start camera capture and live pose/activity inference:

```powershell
.\.venv\Scripts\python.exe main.py --camera 0
```

By default, `--device auto` selects CUDA when the installed PyTorch build can
use it. CUDA runs use the larger `yolo26s-pose.pt` checkpoint for pose
detection and run the selected activity recognizer on the GPU. CPU-only
systems use `yolo26n-pose.pt` and CPU inference. The larger pose weights are
downloaded automatically by Ultralytics on first use. Select a device
explicitly with `--device cpu` or `--device cuda`; an explicit CUDA request
fails with setup guidance if CUDA is unavailable. CUDA acceleration can reduce
latency, but does not by itself guarantee more accurate activity labels.

The window overlays detected pose landmarks with the latest activity,
confidence, and risk label. The first activity prediction appears after 30
detected pose frames; subsequent predictions use a five-frame stride. Press
**Q** or **Esc** while the camera window is focused to stop the pipeline.
Use `--camera 1` to try another device index or `--no-display` for headless
inference. Use `--activity-model gru` to run the older local skeleton GRU
instead. Close other apps using the webcam and allow desktop camera access in
Windows **Settings > Privacy & security > Camera** if the device cannot open.

The pretrained AVA model recognizes actions from general video footage; its
performance can vary with camera angle, distance, occlusion, and lighting. It
does not guarantee perfect activity recognition. Validate it with labeled
clips from the target camera before relying on any safety alert.

### Improve accuracy for your camera

The webcam-calibrated GRU trains on your camera's pose sequences and is much
lighter at inference than the AVA RGB detector. Collect each activity in
separate train, validation, and test recordings. Use distinct recording IDs;
never split overlapping windows from one recording across those partitions.
For a personal model, `--allow-subject-overlap` explicitly allows the same
person in all three partitions while keeping recordings separate. This only
measures performance on new recordings of that person, not on unseen people.

Example for one recording (repeat with a new recording ID for each activity
and each split; the collector supports `standing`, `walking`, `sitting`,
`running`, `falling`, and `lying_down`):

```powershell
.\.venv\Scripts\python.exe tools\collect_webcam.py --subject me --recording standing-train-01 --label standing --split train --duration 30 --allow-subject-overlap
.\.venv\Scripts\python.exe tools\collect_webcam.py --subject me --recording standing-validation-01 --label standing --split validation --duration 30 --allow-subject-overlap
.\.venv\Scripts\python.exe tools\collect_webcam.py --subject me --recording standing-test-01 --label standing --split test --duration 30 --allow-subject-overlap
```

Once every class has enough labeled windows in all three splits, train and
evaluate without replacing the pretrained/default model:

```powershell
.\.venv\Scripts\python.exe -m features.train_webcam_gru --allow-subject-overlap
```

The trainer requires at least 10 training and 3 validation/test sequences per
class, selects checkpoints using recording-level validation macro-F1, and
reports both window-level and recording-level held-out metrics. It refuses
incomplete datasets and won't overwrite an existing checkpoint unless
`--overwrite` is supplied. Review
`reports/safesense_gru_webcam_metrics.json` before using the personalized model.
To run that model:

```powershell
.\.venv\Scripts\python.exe main.py --activity-model gru --model models/safesense_gru_webcam.pt
```

More overlapping windows from the same clip do not count as independent
evidence; collect separate takes, and include varied lighting, distance, and
camera angles relevant to expected use.

The API and camera pipeline are separate processes. Starting the dashboard or
API alone does not launch the camera; when the API is unavailable, inference
events are written to `reports/live_events.jsonl`.

The API persists accepted events in `reports/safesense_events.sqlite3`; its
event history and total count survive API restarts. Set
`SAFESENSE_EVENT_DB_PATH` before starting the API to use another SQLite file.
An ingestion request receives a success response only after the database
commit succeeds. Existing JSONL fallback logs are retained and are not
automatically imported into SQLite.

The browser dashboard shows API, camera, model, database, and recent-inference
health separately from safety events. The inference indicator is based on
whether an event arrived in the last five seconds; it is not a direct camera
process heartbeat. Frame, pose, and FPS metrics are not currently measured.
Zone bounds are loaded from `configs/risk_config.json`, but the dashboard does
not receive live subject coordinates. Event acknowledgement is currently a
local banner dismissal only; persistent incident review and resolution are
not implemented.

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