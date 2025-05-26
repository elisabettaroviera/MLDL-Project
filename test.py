# TEST FILE

############################################################################################################
################################################# STEP 2.a #################################################


# 1- To check if the download of the image works
# Create output dir if needed
import os
import gdown
import torch
from data.dataloader import dataloader
from datasets.cityscapes import CityScapes
from datasets.transform_datasets import transform_cityscapes, transform_cityscapes_mask
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from utils.metrics import compute_flops, compute_latency_and_fps, compute_parameters
from utils.utils import save_metrics_on_file

"""

os.makedirs('./outputs', exist_ok=True)

# Get first batch
first_batch = next(iter(dataloader_cs_train))
# I need filenames to save the images, but i don't think we need it when doing training
# The idea is to save some images and masks for the report (maybe we can understand if the image is so much worse at the first
# epochs is so much worse that at the 50 and discuss some comparison in the report)
images, masks, filenames = first_batch

# Number of samples you want to save from the batch
num_to_save = min(5, len(images))  # e.g., save 5 or fewer

for i in range(num_to_save):
    img_tensor = images[i]
    mask_tensor = masks[i]

    # Check the pixel values of the first mask in the batch, each value should be in the range [0, 18]? (bc 19 classes) + 255 for void
    mask = mask_tensor.cpu().numpy()  # Convert mask tensor to NumPy array

    # Show the unique class values in the mask
    print(f"Unique class values in the mask: {np.unique(mask)}")

    img_pil = TF.to_pil_image(img_tensor.cpu())
    # Convert mask tensor to PIL image, i am using long int64 to keep the class labels but image fromarray doen't support them
    mask_pil = Image.fromarray(mask_tensor.byte().cpu().numpy())  # Convert to uint8 before Image.fromarray

    base_filename = filenames[i].replace("leftImg8bit", "")
    img_path = f'./outputs/{base_filename}_image.png'
    mask_path = f'./outputs/{base_filename}_mask.png'

    img_pil.save(img_path)
    mask_pil.save(mask_path)

    print(f"Saved image to {img_path}")
    print(f"Saved mask to {mask_path}")

# 2- TRYING OUT COMPUTE_MIOU
print("************trying out compute_miou:***************")
# Dummy ground truth and prediction with 3 classes 
gt_images = [
np.array([
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2]
]),
np.array([
    [2, 2, 2],
    [1, 1, 1],
    [0, 0, 0]
])]
pred_images = [
np.array([
    [0, 1, 2],   # correct
    [0, 0, 2],   # 1 → 0 (mistake)
    [0, 1, 1]    # 2 → 1 (mistake)
]),
np.array([
    [2, 1, 2],   # 2 → 1 (mistake)
    [1, 1, 1],   # correct
    [0, 1, 0]    # 1 → 0 (mistake)
])]

mean_iou_dummy, iou_per_class_dummy, intersections_dummy, unions_dummy = compute_miou(gt_images, pred_images, num_classes=3)
print("__________dummy try_________")
print("Mean IoU:", mean_iou_dummy)
print("IoU per class:", iou_per_class_dummy)
print("Intersections:", intersections_dummy)
print("Unions:", unions_dummy)

print("_________try with saved masks________")
gt_mask = np.array(Image.open("outputs/monchengladbach_000000_019500_.png_mask.png").convert("L"))
#gt_mask = np.array(Image.open("outputs/hanover_000000_006922_.png_mask.png").convert("L"))
print("GT labels:", np.unique(gt_mask))
valid_mask = gt_mask != 255
num_classes = int(np.max(gt_mask[valid_mask]) + 1)
mean_iou, iou_per_class, intersections, unions = compute_miou(gt_mask, gt_mask, num_classes)
print(f"mean iou = {mean_iou}")
print(f"iou per class= {iou_per_class}")
print("Intersections:", intersections)
print("Unions:", unions)
"""

# print("************STEP 2.a: TRAINING DEEPLABV2 ON CITYSCAPES***************")
# # Define transformations
# print("Define transformations")
# transform = transform_cityscapes()
# target_transform = transform_cityscapes_mask()

# # Load the datasets (Cityspaces)
# print("Load the datasets")
# cs_train = CityScapes('./datasets/Cityscapes', 'train', transform, target_transform)
# cs_val = CityScapes('./datasets/Cityscapes', 'val', transform, target_transform)

# # DataLoader
# # also saving filenames for the images, when doing train i should not need them
# # each batch is a nuple: images, masks, filenames 
# # I modify the value of the batch size because it has to match with the one of the model
# batch_size = 2 # 3 or the number the we will use in the model
# print("Create the dataloaders")
# dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, cs_val, batch_size, True, True)

# # Definition of the parameters for CITYSCAPES
# # Search on the pdf!! 
# print("Define the parameters")
# num_epochs = 50 # Number of epochs

# # The void label is not considered for evaluation (paper 2)
# # Hence the class are 0-18 (19 classes in total) without the void label
# num_classes = 19 # Number of classes in the dataset (Cityscapes)
# ignore_index = 255 # Ignore index for the loss function (void label in Cityscapes)
# learning_rate = 0.001 # Learning rate for the optimizer
# momentum = 0.9 # Momentum for the optimizer
# weight_decay = 0.0005 # Weight decay for the optimizer
# batch_size = 2 # Batch size for the DataLoader
# iter = 0 # Initialize the iteration counter
# max_iter = num_epochs * len(dataloader_cs_train) # Maximum number of iterations (epochs * batches per epoch)

# # Pretrained model path 
# print("Pretrained model path")
# pretrain_model_path = "./pretrained/deeplabv2_cityscapes.pth"
# if not os.path.exists(pretrain_model_path):
#     os.makedirs(os.path.dirname(pretrain_model_path), exist_ok=True)
#     print("Scarico i pesi pre-addestrati da Google Drive...")
#     url = "https://drive.google.com/uc?id=1HZV8-OeMZ9vrWL0LR92D9816NSyOO8Nx"
#     gdown.download(url, pretrain_model_path, quiet=False)

# print("Load the model")
# model = get_deeplab_v2(num_classes=19, pretrain=True, pretrain_model_path=pretrain_model_path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # === TEST ===
# try:
#     print("Testing compute_latency_and_fps...")
#     latency, std_latency, fps, std_fps = compute_latency_and_fps(model)
#     print(f"Latency: {latency:.2f} ± {std_latency:.2f} ms | FPS: {fps:.2f} ± {std_fps:.2f}")

#     print("\nTesting compute_flops...")
#     flops = compute_flops(model)
#     print(f"Total numer of FLOPS: {flops} GigaFLOPs")

#     print("\nTesting compute_parameters...")
#     total, trainable = compute_parameters(model)
#     print(f"Total Params: {total}, Trainable: {trainable}")

# except Exception as e:
#     print(f"Errore durante il test: {e}")

""""
# === TEST METRICS FILE OUTPUT ===
epoch = 1
metrics_train = {
    'mean_loss': 3,
    'mean_iou': 3,
    'iou_per_class': 3,
    'mean_latency' : 3,
    'std_latency' : 3,
    'mean_fps' : 3,
    'std_fps' : 3,
    'num_flops' : 3,
    'trainable_params': 3
}

metrics_val = {
    'mean_loss': 3,
    'mean_iou': 3,
    'iou_per_class': 3,
    'mean_latency' : 3,
    'std_latency' : 3,
    'mean_fps' : 3,
    'std_fps' : 3,
    'num_flops' : 3,
    'trainable_params': 3
}

save_metrics_on_file(1, metrics_train, metrics_val)
save_metrics_on_file(50, metrics_train, metrics_val)"""
'''''''''

import torch
import numpy as np
from PIL import Image
import os

# Funzione di test
def test_save_images():
    # Crea un directory temporanea per il salvataggio delle immagini
    save_dir = './test_outputs'
    os.makedirs(save_dir, exist_ok=True)

    # Dati di esempio per il test
    file_name_1 = "frankfurt_000001_054640_leftImg8bit.png"
    file_name_2 = "frankfurt_000001_062016_leftImg8bit.png"

    # Simula un batch di dati
    inputs = torch.randn(2, 3, 256, 256)  # 2 immagini, 3 canali, 256x256
    file_names = [file_name_1, file_name_2]
    preds = np.random.randint(0, 19, (2, 256, 256))  # 2 immagini, ogni pixel è una classe predetta (tra 0 e 18)
    color_targets = np.random.randint(0, 256, (2, 256, 256, 3), dtype=np.uint8)  # 2 immagini con maschere colorate

    # Chiamata alla funzione che salverà le immagini
    flag_save = 0
    save_images(flag_save, save_dir, inputs, file_names, preds, color_targets, file_name_1, file_name_2)

    # Verifica che le immagini siano state salvate correttamente
    saved_files = os.listdir(save_dir)
    assert f"{file_name_1}_original.png" in saved_files, f"Immagine originale {file_name_1} non salvata!"
    assert f"{file_name_1}_pred_color.png" in saved_files, f"Maschera predetta {file_name_1} non salvata!"
    assert f"{file_name_1}_color_target.png" in saved_files, f"Maschera target {file_name_1} non salvata!"
    
    assert f"{file_name_2}_original.png" in saved_files, f"Immagine originale {file_name_2} non salvata!"
    assert f"{file_name_2}_pred_color.png" in saved_files, f"Maschera predetta {file_name_2} non salvata!"
    assert f"{file_name_2}_color_target.png" in saved_files, f"Maschera target {file_name_2} non salvata!"

    # Aggiungi un messaggio di successo
    print("Test passed! Immagini salvate correttamente.")

# Funzione di test per salvare le immagini (già fornita)
def save_images(flag_save, save_dir, inputs, file_names, preds, color_targets, file_name_1, file_name_2):
    # color map       
    CITYSCAPES_COLORMAP = np.array([
        [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
        [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
        [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
        [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
    ], dtype=np.uint8)
    
    for input, file_name, pred, color_target in zip(inputs, file_names, preds, color_targets):
        if file_name in [file_name_1, file_name_2]:
            flag_save += 1

            # Salva l'immagine originale da 'inputs' (tensore)
            original_img = input.cpu().numpy().transpose(1, 2, 0)  # Converte (C, H, W) in (H, W, C)
            original_img = np.clip(original_img * 255, 0, 255).astype(np.uint8)  # Normalizza tra 0-255
            original_img_pil = Image.fromarray(original_img)
            original_img_pil.save(f"{save_dir}/{file_name}_original.png")

            # Salva la maschera predetta colorata
            color_mask = CITYSCAPES_COLORMAP[pred]
            color_mask_img = Image.fromarray(color_mask)
            color_mask_img.save(f"{save_dir}/{file_name}_pred_color.png")

            # Salva la maschera target colorata
            if color_target is not None:
                color_target_img = Image.fromarray(color_target)
                color_target_img.save(f"{save_dir}/{file_name}_color_target.png")

# Esegui il test
test_save_images()
'''
'''
import os
from models.deeplabv2.deeplabv2 import get_deeplab_v2
import torch
from torchvision.datasets import ImageFolder
from datasets.transform_datasets import *
from data.dataloader import dataloader
import numpy as np
import time
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torchvision.transforms.functional as TF
from datasets.cityscapes import CityScapes
import random
from train import train
from utils.utils import poly_lr_scheduler, save_metrics_on_file
from validation import validate
from utils.metrics import compute_miou
from torch import nn
import wandb
import gdown
from validation import validate, save_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################################################################################
################################################# STEP 2.a #################################################

print("************STEP 2.a: TRAINING DEEPLABV2 ON CITYSCAPES***************")
# Define transformations
print("Define transformations")
transform = transform_cityscapes()
target_transform = transform_cityscapes_mask()

# Load the datasets (Cityspaces)
print("Load the datasets")
cs_train = CityScapes('./datasets/Cityscapes', 'train', transform, target_transform)
cs_val = CityScapes('./datasets/Cityscapes', 'val', transform, target_transform)

# DataLoader
# also saving filenames for the images, when doing train i should not need them
# each batch is a nuple: images, masks, filenames 
# I modify the value of the batch size because it has to match with the one of the model
batch_size = 3 # 3 or the number the we will use in the model
print("Create the dataloaders")
dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, cs_val, batch_size, True, True)

# Definition of the parameters for CITYSCAPES
# Search on the pdf!! 
print("Define the parameters")
num_epochs = 50 # Number of epochs

# The void label is not considered for evaluation (paper 2)
# Hence the class are 0-18 (19 classes in total) without the void label
num_classes = 19 # Number of classes in the dataset (Cityscapes)
ignore_index = 255 # Ignore index for the loss function (void label in Cityscapes)
learning_rate = 0.0001 # Learning rate for the optimizer - 1e-4
momentum = 0.9 # Momentum for the optimizer
weight_decay = 0.0005 # Weight decay for the optimizer
iter = 0 # Initialize the iteration counter
max_iter = num_epochs * len(dataloader_cs_train) # Maximum number of iterations (epochs * batches per epoch)

# Pretrained model path 
print("Pretrained model path")
pretrain_model_path = "./pretrained/deeplabv2_cityscapes.pth"
if not os.path.exists(pretrain_model_path):
    os.makedirs(os.path.dirname(pretrain_model_path), exist_ok=True)
    print("Scarico i pesi pre-addestrati da Google Drive...")
    url = "https://drive.google.com/uc?id=1HZV8-OeMZ9vrWL0LR92D9816NSyOO8Nx"
    gdown.download(url, pretrain_model_path, quiet=False)

print("Load the model")
model = get_deeplab_v2(num_classes=19, pretrain=True, pretrain_model_path=pretrain_model_path)
model = model.to(device)

# Definition of the optimizer for the first epoch
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay) # Optimizer (Stochastic Gradient Descent)
print("Optimizer loaded")

# Defintion of the loss function
loss = nn.CrossEntropyLoss(ignore_index=ignore_index) # Loss function (CrossEntropyLoss for segmentation tasks)
print("loss loaded")

print("Validation step")
metrics_val = validate(1, model, dataloader_cs_val, loss, num_classes) # Compute the accuracy on the validation set
print("Validation step done")
# Stampa i risultati
print("Metrics output:")
for k, v in metrics_val.items():
    print(f"{k}: {v}")

'''

#test per vedere se applicare le augmentations funziona
import os
import torch

import random
import numpy as np
from datasets.gta5 import GTA5
from torchvision import transforms
from datasets.transform_datasets import transform_gta, transform_gta_mask, transform_cityscapes, transform_cityscapes_mask
from torch.utils.data import ConcatDataset, Subset


transform_gta_dataset = transform_gta()
target_transform_gta = transform_gta_mask()

type_aug = {'weather': ['RandomShadow', 'RandomFog', 'ISONoise','ISONoise', 'GaussianBlur']} #i,l
gta_train_nonaug = GTA5('./datasets/GTA5', transform_gta_dataset, target_transform_gta, augmentation=False, type_aug={}) # No type_aug 
# Contains all pictures bc they are all augmented
gta_train_aug = GTA5('./datasets/GTA5', transform_gta_dataset, target_transform_gta, augmentation=True, type_aug=type_aug) # Change the augm that you want

# Choose with probability 1 PER IL TEST  the augmented images
num_augmented = int(1* len(gta_train_aug))
indices = random.sample(range(len(gta_train_aug)), num_augmented)
gta_train_aug = Subset(gta_train_aug, indices)
output_dir = "./test_augmented_output"
for i in range(3):
    image_tensor, label_tensor, filename = gta_train_aug[i]

    # Converti in immagini PIL per salvataggio
    image = transforms.ToPILImage()(image_tensor)
    label = transforms.ToPILImage()(label_tensor)

    # Salvataggio
    image.save(os.path.join(output_dir, f"augmented_{i}_{filename}"))
    label.save(os.path.join(output_dir, f"mask_{i}_{filename}"))

    print(f"Salvati: augmented_{i}_{filename} e mask_{i}_{filename}")