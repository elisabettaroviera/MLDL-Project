# BISENET
To make this run we consider modifying
- Learning rate
- Loss

Let us now analyse how we might change the learning rate.

## 1. Learning rate
With a **batch size of 4**, it is fair to re-evaluate the **learning rate** from that presented in the papers, as a smaller batch size implies:

* noisier gradient estimates
* higher probability of instability
* but also higher implicit regularisation

##### 📌 *Practical (linear) rule*.

Many papers, including that of [Goyal et al., 2017](https://arxiv.org/abs/1706.02677), suggest that the learning rate can **scalar linearly with batch size**:

$$
\text{LR}_{\text{new}} = \text{LR}_{\text{base}} \times \frac{\text{BS}_{\text{new}}}{\text{BS}_{\text{base}}}
$$

In your case:

$$
\text{LR} = 0.025 \times \frac{4}{16} = 0.00625
$$

So the recommended **learning rate with batch size 4:**

$$
\boxed{\text{lr} = 0.00625}
$$

with:

* **SGD**
* **momentum = 0.9**
* **weight decay = 1e-4**
* **poly scheduler with power = 0.9**

## 2. Loss
Comparing the `CrossEntropyLoss` with alternatives is a useful strategy to improve **robustness**, **contour accuracy** and **class imbalance management**, especially with a dataset like **Cityscapes**, where some classes (e.g. traffic signs or poles) are much less frequent.

✅ 1. **Focal Loss**

* Useful when there is **balance between classes**.
* Penalises difficult examples more (and easy examples less).
* Formula:

  $$
  FL(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)
  $$

  * Typical: $\gamma = 2$, $\alpha_t = 0.25$.
* 💡 **PyTorch**: you can use [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch) or implement it from scratch.

✅ 2. **Dice Loss**

* Improves the **Overlap (IoU-like)**.
* Formula:

  $$
  \text{Dice} = 1 - \frac{2 \cdot |P \cap G|}{|P| + |G|}
  $$
* It is differentiable and often combined with CrossEntropy.
* Ideal for **small objects** and **precise contours**.

✅ 3. **Lovász-Softmax Loss**

* Directly optimises **mIoU**, unlike EC.
* Widely used in competitions (e.g. on Kaggle).
* Requires softmax on logits before calculation.

✅ 4. **Tversky Loss**

* General of Dice. Allows balancing FP and FN:

  $$
  Tversky(P, G) = \frac{|P \cap G|}{|P \cap G| + \alpha |P - G| + \beta |G - P|}
  $$

  Where: $\alpha$ penalises FPs, $\beta$ FNs.
* Useful if you are more interested in certain types of errors.

Let us now write the implementation of the various losses

✅ **Cross Entropy Loss (excludes class void)**

```python
import torch
import torch.nn as nn

def cross_entropy_loss(pred, target, ignore_index=255):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return criterion(pred, target)
```

---

✅ **Dice Loss (multi-class, batch-wise)**

```python
import torch.nn.functional as F

def dice_loss(pred, target, num_classes, ignore_index=255, epsilon=1e-6):
    ‘"’
    pred: (N, C, H, W) logits
    target: (N, H, W) long tensor
    ‘"’
    pred = F.softmax(pred, dim=1)

    # create one-hot encoding, ignoring void
    mask = target != ignore_index
    target = target * mask # set void class to 0 to prevent index error

    one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
    one_hot = one_hot * mask.unsqueeze(1) # zero-out void regions

    dims = (0, 2, 3)
    intersection = torch.sum(pred * one_hot, dims)
    union = torch.sum(pred + one_hot, dims)

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return 1 - dice.mean()
```

---

✅ **Lovász-Softmax Loss (with ignore_index)**

You can use the [`lovasz-softmax`] library directly (https://github.com/bermanmaxim/LovaszSoftmax).
Install it first with:

```bash
pip install git+https://github.com/bermanmaxim/LovaszSoftmax.git
```

Then:

```python
from lovasz_losses import lovasz_softmax

def lovasz_losses(pred, target, ignore_index=255):
    ‘"’
    pred: (N, C, H, W) logits
    target: (N, H, W) long tensor
    ‘"’
    # mask void
    mask = target != ignore_index
    pred = pred.permute(0, 2, 3, 1).contiguous() # N, H, W, C
    return lovasz_softmax(pred, target, ignore=ignore_index)
```

---

🔀 **Combinations**

🔧 **CrossEntropy + Dice**

```python
def combined_ce_dice_loss(pred, target, num_classes, lambda_dice=0.5, ignore_index=255):
    ce = cross_entropy_loss(pred, target, ignore_index)
    dice = dice_loss(pred, target, num_classes, ignore_index)
    return ce + lambda_dice * dice
```

---

🔧 **CrossEntropy + Lovász**

```python
def combined_ce_lovasz_loss(pred, target, lambda_lovasz=0.5, ignore_index=255):
    ce = cross_entropy_loss(pred, target, ignore_index)
    lv = lovasz_loss(pred, target, ignore_index)
    return ce + lambda_lovasz * lv
```
