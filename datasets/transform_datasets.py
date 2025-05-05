import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch

# Define transformations for the Cityscapes
# Add here any specific transformations you want to apply to the Cityscapes dataset
def transform_cityscapes():
    # NOTE: The training resolution and the val resolution are equal in Cityscapes
    # Hence, we can use the same transform for both train and test
    transform = transforms.Compose([
                                    transforms.Resize((1024, 512)), # Resize to 1024x512
                                    transforms.ToTensor() # Convert to tensor
                                    ])
    return transform

def transform_cityscapes_mask():
    def to_tensor_no_normalization(mask):
        # Convert the PIL mask to a NumPy array and then to a tensor with integer labels (no normalization)
        mask_np = np.array(mask, dtype=np.uint8)  # Ensure the mask is of uint8 type for class IDs
        return torch.from_numpy(mask_np).long()  # Convert to tensor with long type (integers)

    # Compose the transformations: Resize + Convert to tensor
    transform = transforms.Compose([
        transforms.Resize((1024, 512), interpolation=Image.NEAREST),  # Resize with nearest neighbor to preserve label IDs
        transforms.Lambda(lambda mask: to_tensor_no_normalization(mask))  # Apply the custom tensor conversion
    ])
    return transform
