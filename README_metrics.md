# 📊 Metrics for Evaluating Semantic Segmentation Models

This README is based on the implemented code in **train.py**, as well as the following reference:  
https://www.jeremyjordan.me/evaluating-image-segmentation-models/.

Evaluating the performance of semantic segmentation models involves measuring both **prediction accuracy** and **computational efficiency**. Below are three key metrics commonly used for this purpose:

---

## 1. 🎯 mIoU% (Mean Intersection over Union)

The **Mean Intersection over Union (mIoU)** is a widely-used metric that quantifies the overlap between the predicted segmentation and the ground truth segmentation for each class.

### Definition:
For each class `c`, IoU is computed as:

```
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

Where:
- **TP** (True Positives): correctly predicted pixels belonging to class `c`
- **FP** (False Positives): pixels incorrectly predicted as class `c`
- **FN** (False Negatives): pixels of class `c` missed by the prediction

The **mean IoU** is the average IoU across all `N` classes:

```
mIoU = (1 / N) * sum(IoU_c for all classes c)
```

Typically expressed as a percentage.

### Why it's useful:
- Balances both over-segmentation and under-segmentation errors.
- Works well in multi-class settings.
- Provides an interpretable overview of per-class and global performance.

---

## 2. ⚡ Latency & FPS (Frames Per Second)

These metrics evaluate the **runtime performance** of the model, essential for real-time applications such as autonomous driving, robotics, or AR/VR.

### Definition:
- **Latency**: Average time in milliseconds (ms) required to process a single image.
- **FPS**: Number of images processed per second.

Relationship:

```
FPS = 1 / Latency (in seconds)
```

Values are typically averaged over many iterations for consistency.

### Why it's useful:
- Indicates whether a model can meet real-time constraints.
- Balances speed with accuracy.
- Helps in selecting and deploying lightweight, responsive models.

---

## 3. 🧮 FLOPs (Floating Point Operations)

**FLOPs** indicate the total number of floating-point operations required to perform a single forward pass of the model.

### Definition:
FLOPs are computed by summing all mathematical operations performed during inference:

```
FLOPs ≈ sum(operations_per_layer * activation_size)
```

Depends on:
- The input resolution
- Model architecture
- Operation types (e.g., convolutions, matrix multiplications)

### Why it's useful:
- Hardware-independent measure of computational complexity.
- Helps compare model efficiency and scalability.
- Essential for deployment on edge devices with limited resources.

---

## 📋 Summary Table

| Metric        | Purpose                             | Unit         | Formula / Concept                                     | Key Benefit                                      |
|---------------|-------------------------------------|--------------|-------------------------------------------------------|--------------------------------------------------|
| **mIoU**      | Segmentation accuracy                | Percentage % | `mIoU = (1 / N) * sum(IoU_c)`                         | Balanced accuracy over all classes               |
| **Latency**   | Inference time per image             | ms           | Time to process a single image                        | Measures model speed                              |
| **FPS**       | Processing speed                     | Frames/sec   | `FPS = 1 / Latency`                                   | Real-time performance indicator                  |
| **FLOPs**     | Computational cost                   | #operations  | `FLOPs ≈ sum(operations_per_layer * activation_size)` | Hardware-agnostic efficiency comparison          |

---