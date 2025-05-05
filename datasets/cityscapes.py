from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

# TODO: implement here your custom dataset class for Cityscapes

# FIRST PROBLEM: What we have to do HERE?? 
# 1. Upload the dataset?
# 2. Modify it in order to use it?

"""
Method       | What it does                                               | When it is called
__init__     | Prepares the dataset (paths, file lists, transformations)  | Once, when the dataset object is created
__getitem__  | Returns the image and corresponding mask at a given index  | Every time the DataLoader requests a batch
__len__      | Returns the total number of samples in the dataset         | At the beginning and during epoch creation

"""

class CityScapes(Dataset):
    def __init__(self, root_dir, split, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.masks = []

        # Construct the paths for images and masks under the 'Cityscapes' folder
        image_dir = os.path.join(root_dir, 'Cityspaces', 'images', split)
        mask_dir = os.path.join(root_dir, 'Cityspaces', 'gtFine', split)

        # Loop through each city folder and get image and mask paths
        for city in os.listdir(image_dir):
            city_image_dir = os.path.join(image_dir, city)
            city_mask_dir = os.path.join(mask_dir, city)
            
            if os.path.isdir(city_image_dir):  # Ensure it's a directory
                for img_name in os.listdir(city_image_dir):
                    if img_name.endswith('.png'):  # Filter for images (assuming .png extension)
                        # Add image paths to list
                        self.images.append(os.path.join(city_image_dir, img_name))
                        
                        # Assume the corresponding mask has the same name with "_labelIds" suffix
                        mask_name = img_name.replace('.png', '_labelIds.png')
                        self.masks.append(os.path.join(city_mask_dir, mask_name))

        # Debugging: Print out the number of images and masks loaded
        print(f"Loaded {len(self.images)} images and {len(self.masks)} masks from {split} set.")
        if len(self.images) == 0 or len(self.masks) == 0:
            raise ValueError(f"No images or masks were found in the {split} set!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image and mask
        img = Image.open(self.images[idx])
        mask = Image.open(self.masks[idx])

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            mask = self.target_transform(mask)

        return img, mask
