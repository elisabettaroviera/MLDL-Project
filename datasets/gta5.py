from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import ToPILImage
import os
from PIL import Image
import numpy as np
from albumentations.pytorch import ToTensorV2
from datasets.transform_datasets import augmentation_transform
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

        if self.augmentation and self.type_aug:
            augmented = augmentation_transform(image=toPil(image), mask=toPil(mask), type_aug=self.type_aug)
            image = augmented['image']
            mask = augmented['mask']
            # If ToTensorV2 is not included in augmentation_transform, uncomment below:
            # image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            # mask = torch.from_numpy(mask).long()
        else:
            if self.transform:
                image = self.transform(toPil(image))
            if self.target_transform:
                mask = self.target_transform(toPil(mask))
        filename = ""
        return image, mask, filename
