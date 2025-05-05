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

# Define the CityScapes dataset class
class CityScapes(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        super(CityScapes, self).__init__()

        self.root_dir = root_dir
        self.split = split  # either 'train' or 'val'
        self.transform = transform
        self.target_transform = target_transform

        self.images = []
        self.masks = []

        # Construct the paths for images and masks under the 'Cityspaces' folder
        images_base = os.path.join(root_dir, 'Cityspaces', 'images', split)
        masks_base = os.path.join(root_dir, 'Cityspaces', 'gtFine', split)

        # Loop through all cities in the split
        for city in os.listdir(images_base):
            img_dir = os.path.join(images_base, city)
            mask_dir = os.path.join(masks_base, city)

            # Loop through all image files in the city's folder
            for file_name in os.listdir(img_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    base_name = file_name.replace('_leftImg8bit.png', '')
                    img_path = os.path.join(img_dir, file_name)
                    mask_name = base_name + '_gtFine_labelIds.png'
                    mask_path = os.path.join(mask_dir, mask_name)

                    # Only include samples that have both image and mask
                    if os.path.exists(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)

    def __getitem__(self, idx):
        # Load the image and corresponding segmentation mask
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])  # mask is a grayscale image

        # Apply image transform (e.g., resize, ToTensor)
        if self.transform:
            image = self.transform(image)

        # Apply mask transform (e.g., resize with NEAREST)
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # Convert mask to tensor without normalization (preserve label IDs)
            mask = transforms.PILToTensor()(mask).squeeze(0).long()  # shape: (H, W)

        return image, mask
    
    def __len__(self):
        # Return the number of samples
        return len(self.images)

