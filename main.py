# Import necessary libraries
import os
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

if __name__ == "__main__":

    #### STEP 2.a
    # Load the dataset CITYSCAPES
    # Define transformations
    transform = transform_cityscapes()
    target_transform = transform_cityscapes_mask()

    # Load the datasets (Cityspaces)
    cs_train = CityScapes('./datasets/Cityscapes', 'train', transform, target_transform)
    cs_val = CityScapes('./datasets/Cityscapes', 'val', transform, target_transform)

    # DataLoader
    # also saving filenames for the images, when doing train i should not need them
    # each batch is a nuple: images, masks, filenames 
    dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, cs_val, 64, True, True)

    # Create output dir if needed
    os.makedirs('./outputs', exist_ok=True)

    # Get first batch
    first_batch = next(iter(dataloader_cs_train))
    # i need filenames to save the images, but i don't think we need it when doing training
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

    # Definition of the parameters
    # Search on the pdf!!
    num_epochs = 50 # Number of epochs

    # FOR LOOP ON THE EPOCHS
    for epoch in range(1, num_epochs + 1):
        # 1. Obtain the pretrained model
        model = None # Put the model with wandb

        # Training
        metrics_train, new_model = train(epoch, model, dataloader_cs_train, criterion, optimizer)
        # PRINT all the metrics!

        # Validation
        metrics_val = validate(new_model, val_loader, criterion) # Compute the accuracy on the validation set
        
        # PRINT all the metrics!
