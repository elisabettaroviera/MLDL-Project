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
# gt_images := the ground truth images (masks)
# pred_images := the predicted images (masks)
# num_classes := number of classes in the dataset
# with return_raw = True also return the count of per class intersections and unions for the batch
# NB: useful to compute iou for the entire train/val set 
# with reurn_raw = False only return the miou and iou per class
# NB: useful to compute the metrics for the single batch of images
def compute_miou(gt_images, pred_images, num_classes, return_raw=True):
    intersections = np.zeros(num_classes)
    unions = np.zeros(num_classes)
    eps = 1e-10  # Small epsilon to avoid division by zero
    
    # for every couple of gt_image and pred_image in the current batch
    for gt, pred in zip(gt_images, pred_images):
        # for every class
        for class_id in range(num_classes):
            # compute intersection of gt and pred
            inter = np.logical_and(gt == class_id, pred == class_id).sum()
            # compute union of gt and pred
            union = np.logical_or(gt == class_id, pred == class_id).sum()
            # accumulate intersections
            intersections[class_id] += inter
            # accumulate unions
            unions[class_id] += union
    # compute iou per class of current batch
    iou_per_class = (intersections / (unions + eps)) * 100
    # compute mean iou (discarting nan)
    mean_iou = np.nanmean(iou_per_class)

    # if return_raw = True also return the raw sum of intersections and unions
    if return_raw:
        return mean_iou, iou_per_class, intersections, unions
    # else only return the batch mean_iou and iou_per_class
    else:
        return mean_iou, iou_per_class

# 2. Latency & FPS
# Function to compute the latency and FPS of a model on a given input size
# model := the model to be evaluated
# height := the height of the input image
# width := the width of the input image
# iterations := number of iterations to compute the latency and FPS

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
# model := the model to be evaluated
# height := the height of the input image
# width := the width of the input image
# iterations := number of iterations to compute the FLOPs

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
