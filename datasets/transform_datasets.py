import torchvision.transforms as transforms

# Define transformations for the Cityscapes
# Add here any specific transformations you want to apply to the Cityscapes dataset
def transform_cityscapes():
    # NOTE: The training resolution and the test resolution are equal in Cityscapes
    # Hence, we can use the same transform for both train and test
    transform = transforms.Resize((1024, 512))  # Resize to 1024x512
    return transform