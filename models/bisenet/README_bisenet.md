# build_bisenet.py

### Summary of the build_bisenet.py Code

**What it implements:**
- This code implements the **BiSeNet (Bilateral Segmentation Network)**, a neural network architecture designed for **real-time semantic segmentation** tasks.
- BiSeNet splits the feature extraction into two paths:
  - **Spatial Path**: preserves spatial resolution to retain detailed information.
  - **Context Path**: captures high-level semantic features using a backbone network (either **ResNet-18** or **ResNet-101**).
- Several key modules are defined:
  - **ConvBlock**: A basic block combining convolution, batch normalization, and ReLU activation.
  - **Spatial_path**: A stack of three ConvBlocks progressively downsampling the input while increasing the number of channels.
  - **AttentionRefinementModule (ARM)**: Enhances feature maps from the context path by applying global attention mechanisms.
  - **FeatureFusionModule (FFM)**: Merges spatial and context features, with an attention-based feature reweighting mechanism to refine the final features before prediction.

**How to use it:**
- You instantiate the `BiSeNet` class with:
  ```python
  model = BiSeNet(num_classes=<number_of_classes>, context_path='resnet18' or 'resnet101')
  ```
- **Training mode**: When `model.train()` is set, the forward pass returns **three outputs**:
  - The final segmentation map (after feature fusion and upsampling).
  - Two auxiliary supervision outputs (`cx1_sup` and `cx2_sup`) from intermediate features, useful for deep supervision during training.
- **Evaluation mode**: When `model.eval()` is set, the forward pass only returns the final segmentation map.
- Pre-trained backbones (`resnet18` or `resnet101`) are required through the `build_contextpath` function (you must ensure this is implemented correctly).

**Other important points:**
- **Initialization**: Convolutions are initialized using **Kaiming Normal** initialization; BatchNorm layers are initialized to default weights and biases.
- **Upsampling**: The outputs are upsampled using **bilinear interpolation** to match the input resolution.
- **Warnings are suppressed**: The script ignores warning messages, which might hide important training information — this should be used cautiously.
- **Multiple learning rates**: The `self.mul_lr` list keeps track of modules where different learning rates might be applied during training (though the exact learning rate strategy must be handled externally, e.g., by using different parameter groups in the optimizer).
- **Dependencies**: Requires PyTorch (`torch`, `torch.nn`) and access to the `build_contextpath` method, which must create a ResNet-based feature extractor.


# build_contextpath.py
### Summary of the build_contextpath.py Code

**What it implements:**
- This code defines two custom PyTorch modules:
  - `resnet18` and `resnet101`, which are **wrappers** around the standard `torchvision` ResNet-18 and ResNet-101 models.
- Both classes **extract intermediate feature maps** at different resolutions from the ResNet backbone, which are typically needed for **semantic segmentation tasks** like BiSeNet.
- Specifically, the models output:
  - `feature3`: output after layer3 (downsampled by a factor of 16 compared to the input size).
  - `feature4`: output after layer4 (downsampled by a factor of 32).
  - `tail`: a global average pooled feature map from `feature4`, used to provide additional global context.

**How to use it:**
- You can instantiate a backbone by calling:
  ```python
  context_path_model = build_contextpath('resnet18')  # or 'resnet101'
  ```
- The `build_contextpath` function selects and returns either a **ResNet-18** or **ResNet-101** backbone model, **pre-trained on ImageNet** (`pretrained=True` by default).
- When calling `.forward(input)`, it returns three outputs needed by models like BiSeNet:
  1. `feature3` (1/16 resolution features)
  2. `feature4` (1/32 resolution features)
  3. `tail` (global context from feature4 via global average pooling)

**Other important points:**
- **Pretraining**: The models use the ImageNet-pretrained weights from `torchvision.models` unless specified otherwise.
- **Global Average Pooling (GAP)**: Instead of using a formal GAP layer, pooling is performed manually by averaging along the last two dimensions (height and width).
- **Lightweight Wrapper**: No changes are made to the internal architecture of ResNet; the code just **selects specific outputs** needed for downstream segmentation models.
- **Hardcoded models**: Only 'resnet18' and 'resnet101' are supported in `build_contextpath`. Trying another name will cause a `KeyError`.
