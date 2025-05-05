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

    # 2. Set the hyperparameters of the model i.e. learning rate, optimizer, etc.
    running_loss = 0.0 
    correct = 0
    total = 0
    iou_per_class = {}

    # 3. Start the training of the model
    model.train() 
    
    # 4. Loop on the batches of the dataset
    for batch_idx, (inputs, targets) in enumerate(dataloader_train): #(X,y)
        inputs, targets = inputs.cuda(), targets.cuda() # GPU

        # Compute outout of the train
        outputs = model(inputs) 
        
        # Compute the loss
        loss = criterion(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the accuracy metrics, i.e. mIoU 
        running_loss += loss.item() # Update of the loss = contain the total loss of the epoch
        for target, output in zip(targets, outputs):
            iou_per_class = compute_miou(target, output, num_classes) 


        

    train_loss = running_loss / len(dataloader_train) # Mean loss of the all dataset
    
    # 5. Compute the computation metrics, i.e. FLOPs
    compute_latency_and_fps(model, height=512, width=1024, iterations=1000)
    compute_flops(model, height=512, width=1024)
    compute_parameters(model)

    # 6. SAVE THE PARAMETERS OF THE MODEL 
    # 6. SAVE MODEL!

    # 7. Return all the metrics
    metrics = {}
    return metrics, new_model
