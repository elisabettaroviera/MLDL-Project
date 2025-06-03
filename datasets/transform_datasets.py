import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import torch
import random

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


#3b_GTA5_to_CITY_augmented_color_2_random_tranform_color_OR_ALL_g_h_i_100_percent
def augmentation_transform(image, mask, type_aug):
    """
    Con probabilità 0.5 applica:
    - 2 trasformazioni casuali dalla lista 'color' di type_aug
    Oppure:
    - Tutte e 3 le trasformazioni atmosferiche: RandomFog, RandomRain, ISONoise
    
    Le trasformazioni vengono applicate con p=1.0.
    
    type_aug deve essere del tipo:
    type_aug = {
        'color': ['HueSaturationValue', 'RGBShift', 'CLAHE', ...]
    }
    """
    # Trasformazioni di colore disponibili
    color_transforms = {
        'HueSaturationValue': A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
        'CLAHE': A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        'GaussNoise': A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0),
        'RGBShift': A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
        'RandomBrightnessContrast': A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
    }

    # Trasformazioni "weather"
    weather_transforms = [
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, alpha_coef=0.1, p=1.0), #g)
        A.RandomRain(blur_value=2, drop_length=10, drop_width=1, brightness_coefficient=0.95, p=1.0), #h)
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0)] #i)]

    use_color = random.random() < 0.5  # 50% di probabilità

    if use_color:
        selected_names = [name for name in type_aug.get('color', []) if name in color_transforms]
        if len(selected_names) < 2:
            raise ValueError("Servono almeno 2 trasformazioni di colore valide.")
        chosen_color = random.sample(selected_names, 2)
        selected_transforms = [color_transforms[name] for name in chosen_color]
    else:
        selected_transforms = weather_transforms  # tutte e 3

    # Componi e applica le trasformazioni
    transform = A.Compose(selected_transforms, p=1.0)
    augmented = transform(image=image, mask=mask)
    return augmented

    '''
### DATA AUGMENTATION VECCHIO PER SCEGLEIRE PIU TRASFORMAZIONI INSIEME ###
def augmentation_transform(image, mask, type_aug): 
    # HorizontalFlip: ruota orizzontalmente l’immagine e la maschera con probabilità del 50%
    # RGBShift: modifica i canali rosso, verde e blu con uno shift casuale nei valori di pixel
    # RandomBrightnessContrast : cambia casualmente luminosità e contrasto
    # MotionBlur : applica una leggera sfocatura da movimento
    # GaussNoise : aggiunge rumore gaussiano (tipo "grana") all’immagine.
    # ShiftScaleRotate : trasla, scala e ruota leggermente l’immagine e la maschera.  
    # NB: le p sono le probabilità con cui quella trasformazione viene applicata

    if 'color' in type_aug:
        n_trans = random.randint(1, 3)          # 1‑3 trasformazioni
        aug_transform = A.Compose([
            A.OneOf([
                A.NoOp(),
                A.SomeOf([
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                                            val_shift_limit=10, p=1.0),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0),
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10,
                                b_shift_limit=10, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2,
                                                contrast_limit=0.2, p=1.0)
                ], n=n_trans, replace=False)
            ], p=1.0)
        ])
    elif 'weather' in type_aug :
        # WEATHER AND ILLUMINATION 
        # RandomShadow — per aggiungere ombre stradali o da edifici.
        # RandomFog o RandomRain — per simulare condizioni meteo.
        # ISONoise — aggiunge rumore simile a fotocamere reali.
        # MotionBlur o GaussianBlur — per simulare imperfezioni delle fotocamere in movimento.
        n_trans = random.randint(1, 2)          # 1‑2 trasformazioni
        aug_transform = A.Compose([
            A.OneOf([
                A.NoOp(),
                A.SomeOf([
                    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1),
                                   num_shadows_lower=1, num_shadows_upper=2, p=1.0),
                    A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15,
                                alpha_coef=0.1, p=1.0),
                    A.RandomRain(blur_value=2, drop_length=10, drop_width=1,
                                 brightness_coefficient=0.95, p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05),
                               intensity=(0.1, 0.3), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0.5, p=1.0)
                ], n=n_trans, replace=False)
            ], p=1.0)
        ])
    elif 'geometric' in type_aug:
        # GEOMETRIC TRANSFORMATIONS Geometric Transforms
        # RandomCrop — per forzare la stessa FOV o porzione visiva.
        # Affine con leggera rotazione/traslazione.
        # Resize — per uniformare la risoluzione.
        # Perspective (con cautela) — per rendere la prospettiva più simile a Cityscapes.
        n_trans = random.randint(1, 2)          # 1‑2 trasformazioni
        aug_transform = A.Compose([
            A.OneOf([
                A.NoOp(),
                A.SomeOf([
                    A.RandomCrop(height=720, width=1280, p=1.0),
                    A.Affine(scale=(0.95, 1.05),
                            translate_percent=(0.02, 0.05),
                            rotate=(-5, 5),
                            shear=(-2, 2), p=1.0),
                    A.Perspective(scale=(0.02, 0.05),
                                keep_size=True, p=1.0)
                ], n=n_trans, replace=False)
            ], p=1.0)
        ])

    else:   # fallback sicuro, nel caso in cui non passo nessuno dei tipi tra color, wheather e geometric
        aug_transform = A.Compose([A.NoOp()])

    # ---------- applica le trasformazioni ----------
    augmented = aug_transform(image=image, mask=mask)
    return augmented
'''