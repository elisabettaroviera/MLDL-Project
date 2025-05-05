# TODO: Define here your training and validation loops.

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

"""   

    DEFINION OF THE METRICS
    1. mIoU% --> compute_miou
    2. Latency & FPS --> compute_latency_and_fps
    3. FLOPs --> compute_flops

"""

# 1. mIoU% 
# Function to compute the mean Intersection over Union (mIoU) for a given set of predictions and targets
# gt_images := all the ground truth images (all the images of the dataset)
# pred_images := all the predicted images (all the images of the dataset)
# num_classes := number of classes in the dataset
def compute_miou(gt_images, pred_images, num_classes):
    iou_per_class = np.zeros(num_classes)
    eps = 1e-10  # To avoid division by 0

    for class_id in range(num_classes):
        total_intersection = 0
        total_union = 0

        # gt_im := ground truth image (only one image)
        # pred_im := predicted image (only one image)
        for gt_im, pred_im in zip(gt_images, pred_images):
            target = (gt_im == class_id)
            prediction = (pred_im == class_id)

            intersection = np.logical_and(target, prediction).sum()
            union = np.logical_or(target, prediction).sum()

            total_intersection += intersection
            total_union += union

        if total_union > 0:
            iou_per_class[class_id] = (total_intersection / (total_union + eps)) * 100  # Convert to percentage
        else:
            iou_per_class[class_id] = np.nan  # If the class is not present in the image, set to NaN

    mean_iou = np.nanmean(iou_per_class) * 100  # Convert to percentage
    return mean_iou, iou_per_class


# 2. Latency & FPS
# Function to compute the latency and FPS of a model on a given input size
# model := the model to be evaluated
# height := the height of the input image
# width := the width of the input image
# iterations := number of iterations to compute the latency and FPS
def compute_latency_and_fps(model, height=512, width=1024, iterations=1000):
    # Generates a random image with shape (3, height, width)
    image = np.random.rand(3, height, width).astype(np.float32)

    latencies = []
    fps_values = []

    for _ in range(iterations):
        start = time.time()

        output = model(image) # Execution of the model on the image

        end = time.time()

        # Latency is the time taken to process the image in seconds
        latency = end - start
        latencies.append(latency)

        # FPS is the number of frames processed per second
        fps = 1.0 / latency if latency > 0 else 0
        fps_values.append(fps)

    # Conversion of latency in milliseconds
    mean_latency = np.mean(latencies) * 1000
    std_latency = np.std(latencies) * 1000

    mean_fps = np.mean(fps_values)
    std_fps = np.std(fps_values)

    return mean_latency, std_latency, mean_fps, std_fps

# 3. FLOPs
# Function to compute the FLOPs of a model on a given input size
# model := the model to be evaluated
# height := the height of the input image
# width := the width of the input image
# iterations := number of iterations to compute the FLOPs
def compute_flops(model, height=512, width=1024):
    image = torch.zeros((3, height, width)) # Generesate a random image with shape (3, height, width) with all zeros

    flops = FlopCountAnalysis(model, image) # Compute the FLOPs of the model on the image

    return flop_count_table(flops)


def train(epoch, model, dataloader_train, criterion, optimizer): # criterion = loss
    model.train() # Here start the training of the model
    
    running_loss = 0.0 
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader_train): #(X,y)
        inputs, targets = inputs.cuda(), targets.cuda() # GPU

        # Compute prediction and loss
        outputs = model(inputs) # Contains the prediction of the model of the current batch
        # outputs contains for each row the tuple (class, prob)
        
        loss = criterion(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() # Update of the loss = contain the total loss of the epoch
        _, predicted = outputs.max(1) # Predicted class
        # outputs.max(0) = the maximum value for each row
        # outputs.max(1) = the predicted class
        total += targets.size(0) # Number of element in the current batch, in total there is the total number of processated batch
        correct += predicted.eq(targets).sum().item() # Sum of correctly predicted element

    train_loss = running_loss / len(dataloader_train) # Mean loss of the all dataset
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%') # epoch = is the current epoch




"""
# WHAT WE HAVE TO DO IN THE TRAINING - VALIDATION LOOP?

-- TRAIN --
Note that the number of epochs is fixed = 50

1. for epoche 
2.   for batch 
3. TODO: WE HAVE TO THINK ABOUT THE STRUCTURE!!!!!!!!!

    
          










-- VAL --
# TODO




"""





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
    dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, cs_val, 64, True, True)

    # Create output dir if needed
    os.makedirs('./outputs', exist_ok=True)

    # Get first batch
    first_batch = next(iter(dataloader_cs_train))
    images, masks, filenames = first_batch
    # Check the pixel values of the first mask in the batch
    mask = masks[23].cpu().numpy()  # Convert mask tensor to NumPy array

    # Show the unique class values in the mask
    print(f"Unique class values in the mask: {np.unique(mask)}")

    # Number of samples you want to save from the batch
    num_to_save = min(5, len(images))  # e.g., save 5 or fewer

    for i in range(num_to_save):
        img_tensor = images[i]
        mask_tensor = masks[i]

        img_pil = TF.to_pil_image(img_tensor.cpu())
        mask_pil = TF.to_pil_image(mask_tensor.cpu())

        base_filename = filenames[i].replace("leftImg8bit", "")
        img_path = f'./outputs/{base_filename}_image.png'
        mask_path = f'./outputs/{base_filename}_mask.png'

        img_pil.save(img_path)
        mask_pil.save(mask_path)

        print(f"Saved image to {img_path}")
        print(f"Saved mask to {mask_path}")
    # Definition of the parameters

    # 
