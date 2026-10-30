# SafeSense AI — Model Card: Temporal Activity Recognition GRU Baseline

**Model Version:** 1.0 (Baseline)  
**Date:** October 2026  
**Module:** Temporal Activity Recognition  
**Framework:** PyTorch (CPU Inference Optimized)  
**Checkpoint Path:** [`models/safesense_gru_baseline.pt`](file:///e:/SafeSense%20-%20Copy/models/safesense_gru_baseline.pt)  

---

## 1. Model Overview

SafeSenseGRU is a compact, CPU-efficient recurrent neural network designed to classify fixed-length temporal skeleton sequences into 6 laboratory safety activity classes.

* **Architecture:** 2-Layer Bidirectional/Unidirectional Gated Recurrent Unit (GRU).
* **Trainable Parameters:** 49,702 parameters (~195 KB on disk).
* **Target Inference Hardware:** Intel i3-1215U / standard laptop CPU.
* **Measured CPU Inference Latency:** **0.105 ms per 30-frame window** ($<0.01\%$ frame budget at 15 FPS).

---

## 2. Input / Output Contracts

### Input Sequence
* **Tensor Shape:** `(batch_size, 30, 51)` (derived by flattening 30 frames $\times$ 17 COCO keypoints $\times$ 3 features).
* **Per-Joint Feature Format:** `[x_norm, y_norm, confidence]`
  * $x_{norm} \in [0.0, 1.0]$: horizontal coordinate normalized by frame width.
  * $y_{norm} \in [0.0, 1.0]$: vertical coordinate normalized by frame height.
  * $confidence \in [0.0, 1.0]$: keypoint detection confidence from YOLO26n-Pose.
* **Temporal Window:** 30 frames ($\approx 2.0$ seconds at 15 FPS), sliding stride = 5 frames.

### Output Classes
The model outputs unnormalized logits for the 6 canonical SafeSense activity classes defined in [`configs/activity_labels.json`](file:///e:/SafeSense%20-%20Copy/configs/activity_labels.json):
0. `standing`
1. `walking`
2. `sitting`
3. `running`
4. `falling`
5. `lying_down`

---

## 3. Training & Evaluation Protocol

### Dataset & Partitioning
Trained on 10,914 standardized continuous sequences from SAFER-Activities / NTU RGB+D with **strict subject-level split isolation**:
* **Train Partition (4,962 sequences):** Subjects `P01` and `P02`.
* **Validation Partition (3,780 sequences):** Subjects `P03` and `p05` (early stopping monitor).
* **Held-Out Test Partition (2,172 sequences):** Subject `P04`.

> [!IMPORTANT]
> Zero Subject Leakage: No frames, recordings, or overlapping sliding windows from the test subject appear in the training or validation sets.

### Optimization Hyperparameters
* **Loss Function:** Cross-Entropy with inverse-frequency class weights (square-root dampened).
* **Optimizer:** Adam ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$).
* **Batch Size:** 64.
* **Early Stopping:** Patience = 6 epochs monitoring validation macro-F1 (converged at epoch 24 in 66.8s).

---

## 4. Empirical Evaluation Results (Held-Out Test Set)

* **Overall Test Accuracy:** **52.16%**
* **Macro-Averaged F1:** **21.15%**
* **Weighted-Averaged F1:** **43.11%**

### Per-Class Detailed Performance

| Class ID | Class Name | Precision | Recall | F1-Score | Support (Windows) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 0 | `standing` | 0.07 | 0.00 | 0.01 | 318 |
| 1 | `walking` | 0.63 | 0.87 | 0.73 | 1200 |
| 2 | `sitting` | 0.00 | 0.00 | 0.00 | 144 |
| 3 | `running` | 0.01 | 0.01 | 0.01 | 354 |
| 4 | `falling` | 0.10 | 0.07 | 0.08 | 42 |
| 5 | `lying_down` | 0.32 | 0.72 | 0.44 | 114 |

### Artifacts Generated
* Checkpoint: [`models/safesense_gru_baseline.pt`](file:///e:/SafeSense%20-%20Copy/models/safesense_gru_baseline.pt)
* Metrics JSON: [`reports/safesense_gru_baseline_metrics.json`](file:///e:/SafeSense%20-%20Copy/reports/safesense_gru_baseline_metrics.json)
* Confusion Matrix CSV: [`reports/safesense_gru_confusion_matrix.csv`](file:///e:/SafeSense%20-%20Copy/reports/safesense_gru_confusion_matrix.csv)
* Confusion Matrix Plot: [`reports/safesense_gru_confusion_matrix.png`](file:///e:/SafeSense%20-%20Copy/reports/safesense_gru_confusion_matrix.png)

---

## 5. Architectural Analysis & Known Limitations

1. **Walking & Lying Down Robustness:** The baseline reliably captures cyclic translational motion (`walking`, recall 87%) and static horizontal ground posture (`lying_down`, recall 72%).
2. **Stationary Posture Ambiguity:** Raw normalized joint coordinates without body orientation or velocity struggle to disambiguate stationary `sitting` from `standing`.
3. **Dynamic Fall Transience:** Falling is a rapid, 0.5–1.0 second transition. In a 30-frame window, the fall event occupies only a fraction of the window before transitioning to `lying_down`.
4. **Planned Enhancements for Phase 8:**
   * Feature engineering: append joint displacement velocities ($\Delta x, \Delta y$) and torso angle $\theta_{torso}$ (shoulder-hip inclination).
   * Two-stage hierarchical classifier or focal loss to strengthen fall transition sensitivity.
