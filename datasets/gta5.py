from torch.utils.data import Dataset
import os
from PIL import Image
import random
import albumentations as A
import numpy as np
from transform_datasets import augmentation_transform

# TODO: implement here your custom dataset class for GTA5


# GTA5 dataset class

class GTA5(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, augmentation = False):
        super(GTA5, self).__init__()
        
        # Initialize lists to store image and label paths
        self.images = []
        self.masks = []
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation = augmentation

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
            # Only consider common image formats
            # all images are png actually
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, img_name)
                label_path = os.path.join(label_dir, img_name)  # assume same filename for label

                # Skip if the corresponding label does not exist
                if not os.path.exists(label_path):
                    print(f"Warning: label not found for image {img_name}")
                    continue

                # Store the valid image-label pair paths
                self.images.append(img_path)
                self.masks.append(label_path)

        print(f"Loaded {len(self.images)} images and {len(self.masks)} masks.")

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = Image.open(self.masks[idx])

        # Seed deterministico per coerenza
        seed = hash(image_path) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        # A sottolineato?
        A.set_seed(seed)

        if self.augmentation:
            # Applichiamo l'augmentazione con OneOf che include NoOp
            augmented = augmentation_transform(image=np.array(image), mask=np.array(label))
            image_aug = Image.fromarray(augmented['image'])
            label_aug = Image.fromarray(augmented['mask'])

            # Verifico se è stata trasformata (NoOp mantiene immagine identica)
            if np.array_equal(np.array(image), np.array(image_aug)):
                # NoOp: non trasformata → ritorna None per scartarla
                return None
            else:
                image = image_aug
                label = label_aug

        # Applico sempre le trasformazioni base
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        filename = os.path.basename(image_path)
        return image, label, filename

    def __len__(self):
        # Return total number of samples
        return len(self.images)

