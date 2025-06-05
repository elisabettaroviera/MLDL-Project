from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import albumentations as A
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from datasets.transform_datasets import augmentation_transform
import torch


class GTA5(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None,
                 augmentation=False, type_aug=None, cache=False, debug=False):
        super().__init__()
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation = augmentation
        self.type_aug = type_aug
        self.cache = cache
        self.debug = debug

        # Caricamento e ordinamento dei path
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

        # Caching in memoria (opzionale)
        self.images_data = None
        self.labels_data = None
        if self.cache:
            self.images_data = [np.array(Image.open(p).convert('RGB')) for p in self.images]
            self.labels_data = [np.array(Image.open(p)) for p in self.labels]
            if self.debug:
                print("[DEBUG] Cached images and masks.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.cache:
            image = self.images_data[idx]
            mask = self.labels_data[idx]
            image_path = self.images[idx]
        else:
            image_path = self.images[idx]
            label_path = self.labels[idx]
            image = np.array(Image.open(image_path).convert('RGB'))
            mask = np.array(Image.open(label_path))
    
        if self.augmentation and self.type_aug:
            augmented = augmentation_transform(image=image, mask=mask, type_aug=self.type_aug)
            image = augmented['image']
            mask = augmented['mask']
    
            # Se augmentation_transform NON include ToTensorV2:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
    
        else:
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
    
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)
    
        filename = os.path.basename(image_path)
        return image, mask, filename

