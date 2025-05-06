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

# TRAIN LOOP
def train(epoch, old_model, dataloader_train, criterion, optimizer): # criterion = loss
    # 1. Obtain the pretrained model
    model = old_model 

    # 2. Initialize the metrics variables and hyperparameters(?)
    running_loss = 0.0 
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)

    # 3. Start the training of the model
    model.train() 
    
    # 4. Loop on the batches of the dataset
    for batch_idx, (inputs, targets) in enumerate(dataloader_train): #(X,y)
        inputs, targets = inputs.cuda(), targets.cuda() # GPU

        # Compute output of the train
        outputs = model(inputs) 
        
        # Compute the loss
        loss = criterion(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item() # Update of the loss = contain the total loss of the epoch

        ## Chat gpt dice: ##
        # Convert model outputs to predicted class labels
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        gts = targets.detach().cpu().numpy()
        
        # Accumulate intersections and unions per class
        _, _, inters, unions = compute_miou(gts, preds, num_classes)
        total_intersections += inters
        total_unions += unions

    # 5. Compute the metrics for the training set 
    # 5.a Compute the accuracy metrics, i.e. mIoU and mean loss
    iou_per_class = (total_intersections / (total_unions + 1e-10)) * 100
    mean_iou = np.nanmean(iou_per_class)
    mean_loss = running_loss / len(dataloader_train)    

    # 5.b Compute the computation metrics, i.e. FLOPs
    mean_latency, std_latency, mean_fps, std_fps = compute_latency_and_fps(model, height=512, width=1024, iterations=1000)
    num_flops = compute_flops(model, height=512, width=1024)
    tot_params, trainable_params = compute_parameters(model)

    # 6. SAVE THE PARAMETERS OF THE MODEL 
    # 6. SAVE MODEL!

    # 7. Return all the metrics
    metrics = {
        'mean_loss': mean_loss,
        'mean_iou': mean_iou,
        'iou_per_class': iou_per_class,
        'mean_latency' : mean_latency,
        'num_flops' : num_flops,
        'trainable_params': trainable_params
    }
    return metrics, new_model
