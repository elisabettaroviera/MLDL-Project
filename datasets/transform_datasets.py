import torchvision.transforms as transforms

# Define transformations for the Cityscapes
# Add here any specific transformations you want to apply to the Cityscapes dataset
def transform_cityscapes():
    transform = transforms.Resize((1024, 512))  # Resize to 1024x512
    return transform