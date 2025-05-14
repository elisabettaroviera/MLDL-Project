import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch

# Define transformations for the Cityscapes
# Add here any specific transformations you want to apply to the Cityscapes dataset
# Resizing is useful beacuse the images  have high rtesolution and we want to reduce the size to speed up the training
# while from the papers we see that the results are not affected too mucnh by the resizing only about 1-2%
def transform_cityscapes(): 
    # NOTE: The training resolution and the val resolution are equal in Cityscapes
    # Hence, we can use the same transform for both train and test
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),  # Resize to 1024x512 (note: H, W)
        transforms.ToTensor(),           # Convert to [0,1] float tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])
    return transform

def to_tensor_no_normalization(mask):
    # Convert the PIL mask to a NumPy array and then to a tensor with integer labels (no normalization)
    mask_np = np.array(mask, dtype=np.uint8)  # Ensure the mask is of uint8 type for class IDs
    return torch.from_numpy(mask_np).long()  # Convert to tensor with long type (integers)

def transform_cityscapes_mask():
    # Compose the transformations: Resize + Convert to tensor
    transform = transforms.Compose([
        transforms.Resize((512, 1024), interpolation=Image.NEAREST),  # Resize with nearest neighbor to preserve label IDs
        transforms.Lambda(lambda mask: to_tensor_no_normalization(mask))  # Apply the custom tensor conversion
    ])
    # You DON'T need to normalize masks because  the contain class labels not pixel value

    return transform

# Mapping from GTA5 IDs to Cityscapes IDs
def transform_gta_to_cityscapes_label(mask):
    # Map of the values GTA5 -> Cityscapes
    id_to_trainid = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
        22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
        31: 16, 32: 17, 33: 18
    }
    
    # Create a new mask initialized to 255 (ignore value)
    mapped = torch.full_like(mask, fill_value=255)
    #se una certa regione dell'immagine ha un ID che non è presente nel dizionario
    #id_to_trainid, viene impostato su 255 per segnalarlo come "non valido" o "da ignorare".
    for gta_id, train_id in id_to_trainid.items():
        mapped[mask == gta_id] = train_id

    return mapped


## TODO: 
def transform_gta(): 
    # NOTE: The training resolution and the val resolution are equal in Cityscapes
    # Hence, we can use the same transform for both train and test
    transform = transforms.Compose([
        transforms.Resize((720, 1280)),  # Resize to 1024x512 (note: H, W)
        transforms.ToTensor(),           # Convert to [0,1] float tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])
    return transform

def transform_gta_mask():
    transform = transforms.Compose([
        transforms.Resize((720, 1280), interpolation=Image.NEAREST),
        transforms.Lambda(lambda mask: to_tensor_no_normalization(mask)),
        transforms.Lambda(transform_gta_to_cityscapes_label)
    ])
    return transform
