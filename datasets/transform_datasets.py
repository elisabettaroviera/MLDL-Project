import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch

# Define transformations for the Cityscapes
# Add here any specific transformations you want to apply to the Cityscapes dataset
# resizing is useful beacuse the images  have high rtesolution and we want to reduce the size to speed up the training
# while from the papers we see that the results are not affected too mucnh by the resizing only about 1-2%
def transform_cityscapes():
    # NOTE: The training resolution and the val resolution are equal in Cityscapes
    # Hence, we can use the same transform for both train and test
    transform = transforms.Compose([
                                    transforms.Resize((512, 1024)), # Resize to 1024x512
                                    transforms.ToTensor() # Convert to tensor
                                    ])
    return transform

"""    transform = transforms.Compose([
        transforms.Resize((512, 1024), interpolation=Image.BILINEAR),  # Resize image (note: H, W)
        transforms.ToTensor(),  # Convert to [0,1] float tensor
        transforms.Normalize(   # Normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            you DON'T need to normalize masks because  the contain class labels not pixel values"""

def transform_cityscapes_mask():
    def to_tensor_no_normalization(mask):
        # Convert the PIL mask to a NumPy array and then to a tensor with integer labels (no normalization)
        mask_np = np.array(mask, dtype=np.uint8)  # Ensure the mask is of uint8 type for class IDs
        return torch.from_numpy(mask_np).long()  # Convert to tensor with long type (integers)

    # Compose the transformations: Resize + Convert to tensor
    transform = transforms.Compose([
        transforms.Resize((512, 1024), interpolation=Image.NEAREST),  # Resize with nearest neighbor to preserve label IDs
        transforms.Lambda(lambda mask: to_tensor_no_normalization(mask))  # Apply the custom tensor conversion
    ])
    return transform
