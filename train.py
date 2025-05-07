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
from utils.utils import poly_lr_scheduler
import wandb

# TRAIN LOOP
def train(epoch, old_model, dataloader_train, criterion, optimizer, iter, learning_rate, num_classes, max_iter): # criterion = loss
    
    # 1. Obtain the pretrained model
    model = old_model 
    print("Training the model...")

    # 2. Initialize the metrics variables and hyperparameters(?)
    print("Initializing the metrics variables...")
    running_loss = 0.0 
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)

    # 3. Start the training of the model
    print("Starting the training of the model...")
    model.train() 

    print(f"Training on {len(dataloader_train)} batches")
    
    # IMPORTANT TO DISCUSS:
    # Problem to fix: decay of learning rate
    # In the paper we can see that the learning rate decays with each batch.
    # Starting from 0.001 and updating for all 20K iterations (number of batches * epochs).
    # We can use the poly_lr_scheduler function to update the learning rate.
    # But this way we have max_iter = 39K (number of batches * epochs) (39300 = 50 * 786)
    # So I think we can try to update the learning rate one iteration yes and one iteration no on the batches.


    # 4. Loop on the batches of the dataset
    for batch_idx, (inputs, targets, file_names) in enumerate(dataloader_train): #(X,y)
        if batch_idx % 100 == 0: # Print every 100 batches
            print(f"Batch {batch_idx}/{len(dataloader_train)}")

        iter += 1 # Increment the iteration counter

        inputs, targets = inputs.cuda(), targets.cuda() # GPU

        # Compute output of the train
        outputs = model(inputs)        

        # Compute the loss
        loss = criterion(outputs[0], targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the learning rate
        lr = poly_lr_scheduler(optimizer, init_lr=learning_rate, iter=iter, lr_decay_iter=1, max_iter=max_iter, power=0.9)

        # Update the running loss
        running_loss += loss.item() # Update of the loss = contain the total loss of the epoch

        ## Chat gpt dice: ##
        # Convert model outputs to predicted class labels
        preds = outputs[0].argmax(dim=1).detach().cpu().numpy()
        gts = targets.detach().cpu().numpy()
        
        # Accumulate intersections and unions per class
        _, _, inters, unions = compute_miou(gts, preds, num_classes)
        total_intersections += inters
        total_unions += unions

    # 5. Compute the metrics for the training set 
    # 5.a Compute the accuracy metrics, i.e. mIoU and mean loss
    print("Computing the metrics for the training set...")
    iou_per_class = (total_intersections / (total_unions + 1e-10)) * 100
    mean_iou = np.nanmean(iou_per_class)
    mean_loss = running_loss / len(dataloader_train)    

    # 5.b Compute the computation metrics, i.e. FLOPs
    print("Computing the computation metrics...")
    mean_latency, std_latency, mean_fps, std_fps = compute_latency_and_fps(model, height=512, width=1024, iterations=1000)
    print(f"Latency: {mean_latency:.2f} ± {std_latency:.2f} ms | FPS: {mean_fps:.2f} ± {std_fps:.2f}")
    num_flops = compute_flops(model, height=512, width=1024)
    print(f"Total numer of FLOPS: {num_flops} GigaFLOPs")
    tot_params, trainable_params = compute_parameters(model)
    print(f"Total Params: {tot_params}, Trainable: {trainable_params}")

    # 6. SAVE THE PARAMETERS OF THE MODEL 
    print("Saving the model")
    # Loggare la loss e altri dati per wandb
    wandb.log({
        "epoch": epoch,
        "loss": mean_loss,
        "lr": lr
    })

    # Salva i pesi del modello dopo ogni epoca
    model_save_path = f"model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': mean_loss,
    }, model_save_path)

    # Salva il modello su wandb
    wandb.save(model_save_path)

    # Alla fine del ciclo, termina il run di wandb
    wandb.finish()
    print("Model saved")

    # 7. Return all the metrics
    metrics = {
        'mean_loss': mean_loss,
        'mean_iou': mean_iou,
        'iou_per_class': iou_per_class,
        'mean_latency' : mean_latency,
        'num_flops' : num_flops,
        'trainable_params': trainable_params
    }
    return metrics, iter
