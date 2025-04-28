In this folder I added the pre-trained weights file for DeepLab, downloaded from the link

**DeepLab petrained weights**: https://drive.google.com/file/d/1ZX0UCXvJwqd2uBGCX7LI2n-DfMg3t74v/view?usp=sharing

This binari file contains a pretrained ResNet model, trained on the ImageNet dataset. It stores the weights and parameters of the model in a binary PyTorch .pth format, meant for easy loading and fine-tuning in deep learning projects.

**How to use this file?** Through chatgpt I can find that the correct use is
```python
model = torchvision.models.deeplabv2()  # or resnet101 --> we have to put the name of the model used
model.load_state_dict(torch.load('deeplab_resnet_pretrained_imagenet.pth'))
```

**NOTE**. In the file *deeplabv2* is implemented this function that I think we can exploit to use this pretained model.
```python
def get_deeplab_v2(num_classes=19, pretrain=True, pretrain_model_path='DeepLab_resnet_pretrained_imagenet.pth'):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes)

    # Pretraining loading
    if pretrain:
        print('Deeplab pretraining loading...')
        saved_state_dict = torch.load(pretrain_model_path)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params, strict=False)

    return model
```


# deeplabv2.py
### Summary of the deeplabv2.py Code

**What it implements:**
- This code builds a **DeepLab v2-like architecture** based on a modified **ResNet-101** backbone for **semantic segmentation** tasks.
- It defines three main classes:
  1. `Bottleneck`: a ResNet-style bottleneck block (3 conv layers with residual connection).
  2. `ClassifierModule`: a classifier head made of **multiple dilated convolutions** to capture multiscale context.
  3. `ResNetMulti`: the full model that stacks the backbone layers and the classifier module.

### Key Details

**Bottleneck block:**
- Standard three-layer bottleneck:
  - `1x1 conv` (reducing dimensions)
  - `3x3 dilated conv`
  - `1x1 conv` (expanding dimensions back)
- BatchNorm layers are used but their parameters' `requires_grad` is set to `False`.
- Residual connections are implemented, and downsampling is used if needed.

**ClassifierModule:**
- Takes high-level features and applies several parallel **3x3 dilated convolutions** with different dilation rates (6, 12, 18, 24).
- Their outputs are **summed together** to form the final classification map.
- Each convolution layer's weights are initialized with a small normal distribution.

**ResNetMulti:**
- Based on ResNet-101 configuration: `[3, 4, 23, 3]` Bottleneck blocks.
- Layers 3 and 4 use **dilated convolutions** (dilation = 2, 4 respectively) instead of strided convolutions to maintain higher resolution.
- The last classifier head (`layer6`) generates the final class predictions.
- Final output is **upsampled via bilinear interpolation** to match the original input size.

**Training vs Inference:**
- During training (`self.training == True`), the model returns `(x, None, None)` to match older DeepLab training pipelines.
- During inference, it returns only the segmentation map `x`.

**Learning Rate Scheduling:**
- `get_1x_lr_params_no_scale()`: returns parameters of the backbone (base learning rate).
- `get_10x_lr_params()`: returns parameters of the final classification head (10× higher learning rate).
- `optim_parameters(lr)`: bundles parameters with different learning rates for optimizers.

**Pretraining:**
- `get_deeplab_v2()` builds the model.
- If `pretrain=True`, it loads pretrained weights (e.g., from ImageNet) by **remapping the keys** of the pretrained model to match the model structure.

---

### Important Design Choices

| Aspect                   | Design 
|--------------------------|--------
| Backbone                 | ResNet-101 
| Dilations in backbone    | 2, 4 (layer3 and layer4) 
| Classifier dilations     | 6, 12, 18, 24
| BatchNorm parameters     | Frozen (no gradient) 
| Pretraining support      | Yes 
| Multi-learning rate      | Different rates for backbone vs head 