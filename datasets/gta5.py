from torch.utils.data import Dataset
import os
from PIL import Image
import random
import albumentations as A
import numpy as np
from datasets.transform_datasets import augmentation_transform

class GTA5(Dataset):

    def __init__(self, root_dir, transform=None, target_transform=None, augmentation = False, type_aug = None):
        super(GTA5, self).__init__()
        
        # Initialize lists to store image and label paths
        self.images = []
        self.masks = []
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation = augmentation
        self.type_aug = type_aug

        # Define image and label directories
        image_dir = os.path.join(root_dir, 'images')
        label_dir = os.path.join(root_dir, 'labels')

        # Check if the directories exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

        # Iterate over all image files
        for img_name in os.listdir(image_dir):
            # Only consider common image formats (all images are png actually)
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, img_name)
                label_path = os.path.join(label_dir, img_name)  # Assume same filename for label

                # Skip if the corresponding label does not exist
                if not os.path.exists(label_path):
                    print(f"Warning: label not found for image {img_name}")
                    continue

                # Store the valid image-label pair paths
                self.images.append(img_path)
                self.masks.append(label_path)

        print(f"Loaded {len(self.images)} images and {len(self.masks)} masks.")


    def __len__(self):
            # Return total number of samples
            return len(self.images)
    from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import ToPILImage
import os
from PIL import Image
import numpy as np
from albumentations.pytorch import ToTensorV2
from datasets.transform_datasets import augmentation_transform, augmentation_transform_oneof_col3_wea
import torch


# GTA5 dataset class
class GTA5(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None,
                 augmentation=False, type_aug=None, debug=False):
        super().__init__()
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation = augmentation
        self.type_aug = type_aug
        self.debug = debug

        self.images = []
        self.labels = []
        for fname in sorted(os.listdir(self.image_dir)):
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.image_dir, fname)
                label_path = os.path.join(self.label_dir, fname)
                if os.path.exists(label_path):
                    self.images.append(img_path)
                    self.labels.append(label_path)

        if self.debug:
            print(f"[DEBUG] Loaded {len(self.images)} image-label pairs.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        toPil = ToPILImage()
        
        image = decode_image(self.images[idx]).to(dtype=torch.uint8)
        mask =  decode_image(self.labels[idx]).to(dtype=torch.uint8)

        if self.augmentation:
            # Applichiamo l'augmentazione con OneOf che include NoOp
            #augmented = augmentation_transform_oneof_col3_wea(image=np.array(image), mask=np.array(label)) # one of 3 best color and best wea
            #label = Image.fromarray(augmented['mask'])
            #image = Image.fromarray(augmented['image'])
            augmented = augmentation_transform(image=toPil(image), mask=toPil(mask), type_aug=self.type_aug) # one of 3 best color and best wea
            image = augmented['image']
            mask = augmented['mask']
            # If ToTensorV2 is not included in augmentation_transform, uncomment below:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        else:
            if self.transform:
                image = self.transform(toPil(image))
            if self.target_transform:
                mask = self.target_transform(toPil(mask))
        filename = ""
        return image, mask, filename
    


