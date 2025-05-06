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
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        super(CityScapes, self).__init__()

        self.images = []
        self.masks = []
        self.transform = transform
        self.target_transform = target_transform

        # Define image and mask directories
        image_dir = os.path.join(root_dir, 'Cityspaces/images', split)
        mask_dir = os.path.join(root_dir, 'Cityspaces/gtFine', split)

        # Check if paths exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        # Iterate over cities
        for city in os.listdir(image_dir):
            #print(f"City: {city}")  # Debugging line
            img_city_path = os.path.join(image_dir, city)
            mask_city_path = os.path.join(mask_dir, city)

            # Iterate over image files
            for img_name in os.listdir(img_city_path):
                #print(f"  Image: {img_name}")  # Debugging line
                if img_name.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(img_city_path, img_name)

                    # Generate corresponding mask name
                    base_name = img_name.replace('_leftImg8bit.png', '')
                    mask_name = base_name + '_gtFine_labelTrainIds.png'
                    mask_path = os.path.join(mask_city_path, mask_name)

                    # Check if the mask exists
                    if not os.path.exists(mask_path):
                        print(f"Warning: mask not found for image {img_name}")
                        continue  # Skip if no mask is found

                    self.images.append(img_path)
                    self.masks.append(mask_path)

        print(f"Loaded {len(self.images)} images and {len(self.masks)} masks from {split} set.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        filename = os.path.basename(self.images[idx])
        return image, mask, filename



