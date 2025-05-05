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

# 4. Parameters
def compute_parameters(model):
    # Compute the number of parameters in the model
    tot_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return tot_params, trainable_params
