# Preprocessing Cityscapes
### ✅ Complete pre-processing checklist for Cityscapes images

#### 1. 📥 **Input RGB (images)**
The resize part is fine, but you have to complete the **transform** with:
- **Conversion to tensor** (`ToTensor`)
- **Normalisation** with mean/std of ImageNet

```python
transforms.Compose([
    transforms.Resize((1024, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet mean
                         std=[0.229, 0.224, 0.225])    # ImageNet std
])
```

---

#### 2. 🏷️ **Target (ground truth/label)**
Cityscapes labels are not ready to use as they are:
- Each pixel has a value between 0 and 33 (34 classes)
- **Only 19 classes are to be considered** in the training → labels need to be **mapped**.
- Pixels that do not fit into the 19 classes must be marked as `ignore_index=255` (standard CrossEntropy)

✔️ So you need to **map each value [0-33] → [0-18] or 255**.

💡 Typical mapping code (to put in the dataset):
```python
ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255,
    5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 255, 11: 2, 12: 3, 13: 4, 14: 255,
    15: 255, 16: 255, 17: 5, 18: 255, 19: 6,
    20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
    30: 255, 31: 16, 32: 17, 33: 18
}

def encode_target(target):
    target_copy = target.copy()
    for k, v in ID_TO_TRAINID.items():
        target_copy[target == k] = v
    return target_copy
```

And then in the `__getitem__` of your `CityscapesDataset`:
```python
target = np.array(Image.open(label_path))
target = encode_target(target)
target = Image.fromarray(target.astype(np.uint8))
```

---

### 🎯 Summary: what to add to your transform
If you want a complete `transform_cityscapes()`:

#### ✅ For images:
```python
def transform_cityscapes_img():
    return transforms.Compose([
        transforms.Resize((1024, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
```

#### ✅ For labels:
- Resize only (without normalisation or `ToTensor`)
- Then convert it to `torch.LongTensor` *after* mapping with `encode_target()`

```python
def transform_cityscapes_label():
    return transforms.Resize((1024, 512), interpolation=Image.NEAREST)
```

# Mini batch-size 
### 🧠 2. Mini-batch Size in Semantic Segmentation

Choosing and tuning the mini-batch size is critical for training efficiency, convergence, and generalization.

#### 📌 How to Choose Mini-Batch Size

- **Hardware Limitations**: 
  - The batch size is primarily constrained by the GPU memory. Semantic segmentation models are memory-intensive due to large input sizes (e.g., 1024×2048 in Cityscapes).
  - Start with a small batch size (e.g., 2 or 4) and increase it until you hit OOM (Out of Memory) errors.

- **Normalization Layers**:
  - If you use **Batch Normalization**, small batch sizes can cause noisy statistics. Consider replacing it with **GroupNorm** or **InstanceNorm** for small batches.

- **Gradient Stability**:
  - Larger batches generally offer more stable gradients and faster convergence, but they may require lower learning rates.
  - Smaller batches introduce more noise, potentially aiding generalization, but might slow convergence.

#### 📅 When to Choose the Batch Size

- Before training begins, based on:
  - Memory profiling on a subset of the dataset.
  - Your model architecture and input resolution.

#### 🛠️ Tuning Strategy

- **Learning Rate Scaling**: If you change the batch size `B`, scale the learning rate `η` proportionally:  
`η_new = η_original × (B_new / B_original)`


- **Gradient Accumulation**: If you want a large effective batch size but are constrained by memory, accumulate gradients over multiple steps.

- **Validation Performance**: Tune the batch size by observing its effect on:
  - Validation loss curves
  - mIoU (mean Intersection over Union)

#### 🧪 Empirical Advice

- For DeepLabV2 + Cityscapes on a single GPU, batch sizes of 2 to 8 are typical.
- Use mixed-precision training (`torch.cuda.amp`) to save memory if using PyTorch.


## How to deal with the batch-size? 
### ✅ **1. Where the batch size** is used
- The **batch size** is a parameter you give to your PyTorch `DataLoader`:
  ```python
  train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, ...)
  ```
- In each iteration of the training loop, the network receives **batch_size images and labels**, and calculates the loss and backward on that batch.

---

### 🔍 **2. How to choose the batch size (real criteria)**

#### 🧠 Technical criteria:
- **Heavy dataset + large model (e.g. DeepLabV2)** ⇒ small batch size
- **Input 1024x512 + ResNet-101 + ASPP** ≈ ~2-4 images per GPU with 12-24 GB VRAM

#### 🔧 Practical tip:
- Start with `batch_size=2`.
- Try `batch_size=4` and see if you have **Out Of Memory (OOM)**.
- If you want a larger ‘virtual’ batch, you can use **gradient accumulation**

---

### 🛠️ **3. When choosing**
You must choose the batch size **before training**, and keep it fixed (unless experimenting or tuning). Changing it **often** also implies an adjustment of the learning rate.

---

### ⚖️ **4. Batch Size Effects**
| Batch Size | Positive Effects | Negative Effects |
|------------|------------------------------------------|---------------------------------------|
| Small (1-2) | Low memory; + generalisation | Noisier gradients; slow training |
| Medium (4-8) | Good memory/stability balance | Requires tuning of LR |
| Large (8+) | Stable gradients; fast training | Requires a lot of memory |

---

### 📏 **5. Learning rate and batch size**
As per theory and practice:
> **If you double the batch size, consider doubling the learning rate**.

For example:
- `batch_size = 2` → `lr = 0.01`.
- batch_size = 4` → `lr = 0.02`.

DeepLabV2 uses `base_lr = 2.5e-4` by default for `batch_size=10` in the paper. Adjust accordingly.

---

### 🔧 Automatic tuning (optional)
Want to do **batch size dynamic tuning**? You can use `torch.cuda.amp` (mixed precision) and tools like [PyTorch Lightning's auto_scale_batch_size](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#auto-scaling-of-batch-size), but *it's not necessary* in a manual pipeline.
