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

 
# DEFINION OF THE METRICS
# 1. mIoU% --> compute_miou
# 2. Latency & FPS --> compute_latency_and_fps
# 3. FLOPs --> compute_flops

# 1. mIoU% 
# Function to compute the mean Intersection over Union (mIoU) for a given set of predictions and targets
def compute_miou(gt_images, pred_images, num_classes, return_raw=True):
    # return_raw = True --> return the count of per class intersections and unions for the batch, useful to compute iou for the ENTIRE train/val set 
    # reurn_raw = False --> only return the miou and iou per class, useful to compute the metrics for the SINGLE batch of images

    intersections = np.zeros(num_classes)
    unions = np.zeros(num_classes)
    eps = 1e-10  # Small epsilon to avoid division by zero
    
    # For every couple of gt_image and pred_image in the current batch
    for gt, pred in zip(gt_images, pred_images):
        # For every class
        for class_id in range(num_classes):
            # Compute intersection of gt and pred
            inter = np.logical_and(gt == class_id, pred == class_id).sum()
            # Compute union of gt and pred
            union = np.logical_or(gt == class_id, pred == class_id).sum()
            # Accumulate intersections
            intersections[class_id] += inter
            # Accumulate unions
            unions[class_id] += union

    # Compute iou per class of current batch
    iou_per_class = (intersections / (unions + eps)) * 100

    # Compute mean iou (discarting nan)
    mean_iou = np.nanmean(iou_per_class)

    if return_raw:
        return mean_iou, iou_per_class, intersections, unions
    else:
        return mean_iou, iou_per_class

# 2. Latency & FPS
# Function to compute the latency and FPS of a model on a given input size
def compute_latency_and_fps(model, height=512, width=1024, iterations=1000, warmup_runs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generates a random image with shape (1, 3, height, width) directly as a PyTorch tensor
    image = torch.randn(1, 3, height, width, device=device)

    model.eval()
    model.to(device)

    latencies = []
    fps_values = []

    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _ = model(image)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()

            latency = end - start
            latencies.append(latency)
            fps_values.append(1.0 / latency if latency > 0 else 0)

    # Conversion of latency in milliseconds
    mean_latency = np.mean(latencies) * 1000 
    std_latency = np.std(latencies) * 1000
    mean_fps = np.mean(fps_values)
    std_fps = np.std(fps_values)

    return mean_latency, std_latency, mean_fps, std_fps



# 3. FLOPs
# Function to compute the FLOPs of a model on a given input size
def compute_flops(model, height=512, width=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.zeros((1, 3, height, width), device=device)
    model = model.to(device)
    model.eval()

    flops = FlopCountAnalysis(model, image) # Table
    total_flops = flops.total()  # Totat FLOPs

    return total_flops / 1e9  # Number of FLOPS in GigaFLOPs

# 4. Parameters
def compute_parameters(model):
    # Compute the number of parameters in the model
    tot_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return tot_params, trainable_params
