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
    # Resize using nearest neighbor to preserve class IDs
    mask = mask.resize((1024, 512), Image.NEAREST)
    
    # Convert to numpy array (preserving integer labels)
    mask_np = np.array(mask, dtype=np.uint8)

    # Convert to tensor of type long (required for segmentation loss)
    return torch.from_numpy(mask_np).long()
