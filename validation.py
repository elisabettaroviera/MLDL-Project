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
from utils.metrics import compute_miou, compute_latency_and_fps, compute_flops, compute_parameters

# VALIDATION LOOP
def validate(new_model, val_loader, criterion, num_classes):
    # 1. Obtain the pretrained model 
    model = new_model
    
    # 2. Initialize the metrics variables    
    mean_loss = 0
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)

    # 3. Start the validation of the model
    model.eval()

    # 4. Loop on the batches of the dataset
    with torch.no_grad(): # NOT compute the gradient (we already computed in the previous step)
        for batch_idx, (inputs, targets, file_names) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # Compute output of the model
            outputs = model(inputs) # Predicted
            
            # Compute the loss
            loss = criterion(outputs, targets)

            # Update the running loss
            running_loss += loss.item() 

            ## Chat gpt dice: ##
            # Convert model outputs to predicted class labels
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            gts = targets.detach().cpu().numpy()
            
            # Accumulate intersections and unions per class
            _, _, inters, unions = compute_miou(gts, preds, num_classes)
            total_intersections += inters
            total_unions += unions

    # 5. Compute the metrics for the validation set 
    # 5.a Compute the accuracy metrics, i.e. mIoU and mean loss
    iou_per_class = (total_intersections / (total_unions + 1e-10)) * 100
    mean_iou = np.nanmean(iou_per_class)
    mean_loss = mean_loss / len(val_loader)

    # 5.b Compute the computation metrics, i.e. FLOPs
    mean_latency, std_latency, mean_fps, std_fps = compute_latency_and_fps(model, height=512, width=1024, iterations=1000)
    num_flops = compute_flops(model, height=512, width=1024)
    tot_params, trainable_params = compute_parameters(model)
   
    # 6. Return all the metrics
    metrics = {
        'mean_loss': mean_loss,
        'mean_iou': mean_iou,
        'iou_per_class': iou_per_class,
        'mean_latency' : mean_latency,
        'num_flops' : num_flops,
        'trainable_params': trainable_params
    }

    return metrics