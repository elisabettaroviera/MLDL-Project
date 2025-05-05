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
def validate(new_model, val_loader, criterion):
    # 1. Obtai the pretrained model 
    model = new_model

    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad(): # NOT compute the gradient (we already computed in the previous step)
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs) # Predicted
            
            # Compute the accuracy metrics, i.e. mIoU 
            loss = criterion(outputs, targets) # Computation of the loss

            # As computed in the train part
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 2. Compute the computation metrics, i.e. FLOPs
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    # 3. Return all the metrics
    metrics = {}
    return metrics