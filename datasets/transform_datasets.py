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
        # Convert the PIL image to a NumPy array, then to a tensor without normalization
        return torch.from_numpy(np.array(mask)).long()

    transform = transforms.Compose([
        # Resize the segmentation mask to 1024x512 using NEAREST interpolation
        transforms.Resize((1024, 512), interpolation=Image.NEAREST),
        to_tensor_no_normalization  # Custom function to preserve label IDs
    ])
    return transform
