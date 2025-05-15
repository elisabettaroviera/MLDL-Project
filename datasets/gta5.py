from torch.utils.data import Dataset
import os
from PIL import Image


# TODO: implement here your custom dataset class for GTA5


# GTA5 dataset class

class GTA5(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        super(GTA5, self).__init__()
        
        # Initialize lists to store image and label paths
        self.images = []
        self.masks = []
        self.transform = transform
        self.target_transform = target_transform

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
        # Load image and label
        image = Image.open(self.images[idx]).convert('RGB')  # convert image to RGB
        label = Image.open(self.masks[idx])  # do not convert label (keep as is)
                
        # If the random number is < 0.5, we apply the augmentation to the photo
        if np.random.random() < self.augmentation_probability:
            augmentation = self.augmentation_transforms[np.random.randint(0, len(self.augmentation_transforms)-1)]
            image = augmentation(image)
            mask = augmentation(mask)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # Return image, label, and the filename
        filename = os.path.basename(self.images[idx])
        return image, label, filename

    def __len__(self):
        # Return total number of samples
        return len(self.images)

