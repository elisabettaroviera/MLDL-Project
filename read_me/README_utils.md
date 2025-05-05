### Summary of the utils.py Code

**What it implements:**
- Utility functions for **learning rate scheduling** and **evaluation metrics** (specifically **IoU** for semantic segmentation).

### Key Details

| Function Name           | Purpose 
|-------------------------|---------
| `poly_lr_scheduler`     | Applies **polynomial decay** to the learning rate during training 
| `fast_hist`             | Quickly computes the **confusion matrix** between ground truth and predictions 
| `per_class_iou`         | Computes the **Intersection over Union (IoU)** score per class 


### Function-by-function breakdown

**1. `poly_lr_scheduler`**
- Reduces the learning rate according to a **polynomial decay** formula:
  
  \[
  \text{lr} = \text{init\_lr} \times (1 - \frac{\text{iter}}{\text{max\_iter}})^{\text{power}}
  \]
  
- Updates the optimizer’s learning rate directly (`optimizer.param_groups[0]['lr'] = lr`).
- Parameters:
  - `init_lr`: starting learning rate.
  - `iter`: current training iteration.
  - `lr_decay_iter`: frequency of decay (default = every iteration).
  - `max_iter`: maximum number of iterations (default = 300).
  - `power`: shape of the decay curve (default = 0.9).
- **Notes**: even though `lr_decay_iter` is listed, it is **not actually used** in the current code.

**2. `fast_hist`**
- Builds a **confusion matrix** (`hist`) between predictions and ground-truth labels efficiently.
- Logic:
  - Only considers valid labels (`a >= 0 and a < n`).
  - Uses `np.bincount` for fast computation.
  - Reshapes the output into an `n x n` confusion matrix.
- Input:
  - `a`: ground truth labels.
  - `b`: predicted labels.
  - `n`: number of classes.

**3. `per_class_iou`**
- Calculates the **IoU (Intersection over Union)** for each class from the confusion matrix.
- Formula for each class:

  \[
  \text{IoU}_i = \frac{\text{True Positive}_i}{\text{True Positive}_i + \text{False Positive}_i + \text{False Negative}_i + \epsilon}
  \]

- Where:
  - `np.diag(hist)`: gives true positives.
  - `hist.sum(1)`: total ground truth instances per class.
  - `hist.sum(0)`: total predicted instances per class.
- A small `epsilon` is added for numerical stability (avoiding division by zero).

---

### Important Design Choices

| Aspect                          | Design 
|---------------------------------|--------
| Learning Rate Decay Type        | Polynomial 
| Learning Rate Update            | Manual in optimizer 
| Confusion Matrix Computation    | Vectorized with `np.bincount` 
| IoU Calculation                 | Per-class, robust to division by zero 
