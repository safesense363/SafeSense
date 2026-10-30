# SafeSense AI — Dataset Card: Le2i Fall Dataset Export

**Document Version:** 1.0  
**Date:** October 2026  
**Component:** Data Ingestion & Validation  
**Auditor:** SafeSense Engineering Team  

---

## 1. Executive Summary

This dataset card documents the provenance, physical inventory, label verification, and engineering limitations of the dataset located at `data/raw/le2i/fall_dataset/`.

Visual inspection and coordinate analysis confirm that this directory contains a **static single-frame image dataset with YOLO bounding-box annotations**, likely sourced from an aggregated web/Roboflow export rather than continuous Le2i laboratory video clips. 

> [!WARNING]
> **Critical Architectural Boundary:**  
> This dataset contains static single images and **cannot be used to train temporal activity recognition models** (such as GRU, LSTM, or temporal Transformers requiring sequence shapes `(30, 17, 3)`). It does not capture motion velocity, transition dynamics, or temporal fall events.

---

## 2. Dataset Provenance & Physical Inventory

* **Local Directory:** `data/raw/le2i/fall_dataset/`
* **Format:** YOLO Bounding Box text files (`class_id x_center y_center width height`, coordinates normalized in `[0.0, 1.0]`).
* **Source Characteristics:** Heterogeneous image resolutions (ranging from 158×319 to 4288×2848), including indoor security scenes, stock photography, and watermarked internet images.

### Inventory Breakdown

| Partition | Label Files (`.txt`) | Image Files (`.jpg`) | Status / Anomaly |
| :--- | :---: | :---: | :--- |
| **Train** | 374 | 374 | 1:1 matching image/label pairs |
| **Validation** | 111 | 89 | **22 missing images** (label files exist without JPGs) |
| **Total** | **485** | **463** | 463 valid pairs; 22 orphaned label files |

### Missing Validation Image Files
The following 22 label files in `labels/val/` have no corresponding image in `images/val/`:
* `not fallen020.txt` through `not fallen041.txt` (inclusive).

---

## 3. Label Analysis & Visual Disambiguation

Because the export lacked an official `dataset.yaml` or metadata mapping file, the engineering team executed visual inspection via [`tools/inspect_le2i_annotations.py`](file:///e:/SafeSense%20-%20Copy/tools/inspect_le2i_annotations.py) to map class IDs `0`, `1`, and `2` to verified visual semantics.

### Class ID Semantics Verified via Inspection

| Class ID | Visual Posture / Semantic | Dominant Aspect Ratio ($W/H$) | Representative Samples |
| :---: | :--- | :---: | :--- |
| **0** | **Fallen / Lying Down** (horizontal body pose on floor or bed) | High ($\ge 1.5 - 2.8$) | `train_fall001`, `train_fall010`, `train_not fallen058` |
| **1** | **Standing / Walking** (upright vertical body pose) | Low ($\approx 0.35 - 0.45$) | `train_not fallen017` (background hallway) |
| **2** | **Sitting / Bending / Crouching** (seated or non-upright non-prone) | Medium ($\approx 0.7 - 1.1$) | `train_not fallen001`, `train_not fallen017`, `train_fall010` |

### Bounding Box Distribution by Partition

| Partition | Class 0 (Fallen/Lying) | Class 1 (Standing/Walking) | Class 2 (Sitting/Bending) | Total Bounding Boxes |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 213 | 120 | 120 | 453 |
| **Validation** | 72 | 23 | 19 | 114 |
| **Combined** | **285** | **143** | **139** | **567** |

### Filename vs Class ID Disconnect Explained
Filenames (`fall*.jpg` vs `not fallen*.jpg`) represent **scene-level filenames**, whereas annotations represent **instance-level bounding boxes for individual persons**:
1. Multi-person scenes contain mixed classes (e.g., in `fall010`, one fallen person is labeled `Class 0`, while two attending individuals are labeled `Class 2`).
2. Images labeled `not fallen` can contain `Class 0` instances (e.g., `not fallen058` depicts an individual lying in bed, correctly tagged as horizontal pose `Class 0`).
3. Images labeled `fall` frequently contain bystanders (`Class 1` or `Class 2`).

---

## 4. Engineering Risks & Limitations

1. **No Temporal Continuity:** Each image is an isolated static frame. Training temporal models on static frames produces meaningless sequence representations.
2. **Missing Validation Files:** Any automated loader iterating over `labels/val/` without checking image existence will raise `FileNotFoundError` or image loading errors on 22 samples.
3. **Severe Class Imbalance:** `Class 0` comprises $>50\%$ of all bounding boxes, while `Class 1` and `Class 2` are underrepresented.
4. **Subject Overlap / Leakage Unknown:** Because provenance metadata is missing, train and validation partitions cannot guarantee subject or scene independence.

---

## 5. SafeSense Usage Recommendations

1. **Do Not Use for Temporal GRU Training:** SafeSense activity recognition requires input shape `(30, 17, 3)` across consecutive frames. This dataset must not be fed into the temporal pipeline.
2. **Potential Use as Pose Estimator Sanity Check:** This dataset may be retained solely to evaluate whether `PoseEstimator` (YOLO-Pose) accurately detects keypoints on lying-down persons in difficult static orientations.
3. **Sequence Pipeline Source:** Temporal models should be trained exclusively on verified sequence data (such as the SAFER-Activities dataset or custom-recorded laboratory sessions captured with [`tools/collect_webcam.py`](file:///e:/SafeSense%20-%20Copy/tools/collect_webcam.py)).
