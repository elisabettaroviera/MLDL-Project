import torchvision.transforms as transforms
from PIL import Image

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
    transform =  transforms.Compose([
                                    # Resize the segmentation mask to 256x512 using NEAREST interpolation
                                    # This is important to preserve label IDs (integers) without interpolation artifacts
                                    transforms.Resize((1024, 512), interpolation=Image.NEAREST), 
                                    transforms.ToTensor()
                                ])
    return transform
