## 🧠 STEP-BY-STEP PSEUDO-CODE FOR THE TRAINING PART

---

### 1. 📂 **Dataset: Loading and Transformations**
DONE

---

### 🔁 **DataLoader**

DONE

---

### 3. 🏗️ **Model: DeepLabV2 + ResNet101**

DONE FOR NOW --> we have the pretained model (ResNet) and the function for deeplabv2.

**TODO** We have to understand
- How we can use ResNet (pratically, how can we import the pretained net in our project) --> ```python get_deeplab_v2 ```
- If and in which way we have to modify the DeppLabV2 model 
- How we can save the trained model --> compute some epochs, save it, start to train again from where we left off 
- How we can tune the hyperparameter --> have we to tune? which parameters?

---

### 4. 🎯 **Loss Function, Optimiser and Scheduler**

**TODO**: We have to select the correct loss function, optimiser and scheduler.
- Which loss function can we use?
- Which optimiser can we use?
- Which scheduler can we use?

**I.E.**
```python
loss_fn = CrossEntropyLoss(ignore_index=255) # ignore label ‘void’ in cityscapes

optimiser = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

scheduler = StepLR(optimiser, step_size=25, gamma=0.1)
# or use PolyLR with decay of the type (1 - epoch / max_epochs) ^ 0.9
```

---

### 5. 🔁 **Training Loop**

##### TODO: We have to better understand the schema that we want to implement in the trianing part!!!!!!

```python
for epoch in range(1, 51):
    model.train()
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimiser.zero_grad()
        outputs = model(images) # [B, C=19, H, W].
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimiser.step()

    scheduler.step()

    # validation
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_targets.append(masks)

    # compute mIoU
    miou = compute_mIoU(all_preds, all_targets, num_classes=19)
    print(f ‘Epoch {epoch} - mIoU: {miou:.4f}’)
```

---

### 6. 📐 **Metrics**

DONE: we have already impleemnted the following metrics
- mIoU
- Latency & FPS
- Flops

---

## ✅ Important Things to Consider

- `ignore_index=255`: Cityscapes uses the 255 label for ignored classes (void). --> **We have to understand this thing!!!!**
- `Normalize`: values must use the **mean and std of ImageNet**, because the backbone is pre-trained there.
- The DeepLabV2 network is not included in PyTorch natively, so you may have to use an open-source repo (e.g. [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) or similar) or a custom implementation. --> **THIS IS VERY IMPORTATN!!**
- The ‘coarse’ classes should be excluded, use `mode=‘fine’` in the dataset.