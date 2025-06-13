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
from utils.utils import poly_lr_scheduler, poly_lr_scheduler_warmup

import wandb
import gc
import torch.nn.functional as F
def get_boundary_map(target, kernel_size=3):
    # target: (B, H, W) con valori interi [0, num_classes-1]
    b, h, w = target.shape
    target = target.unsqueeze(1).float()  # (B,1,H,W)

    laplace_kernel = torch.tensor([[[[0, 1, 0],
                                     [1,-4, 1],
                                     [0, 1, 0]]]], device=target.device).float()
    
    boundary = F.conv2d(target, laplace_kernel, padding=1).abs()
    boundary = (boundary > 0).float()  # binarizza

    return boundary  # shape (B,1,H,W)
"""def compute_pidnet_loss(criterion, x_extra_p, x_main, x_extra_d, target, boundary,
                        lambda_0=0.4, lambda_1=0.6, lambda_2=1.0, lambda_3=1.0):"""


def compute_pidnet_loss(criterion, x_extra_p, x_main, x_extra_d, target, boundary,
                        lambda_0=0.4, lambda_1=20.0, lambda_2=1.0, lambda_3=1.0):
    #perche prima usavo t:0.5
    #lambda_0=0.4, lambda_1=0.6, lambda_2=1.0, lambda_3=0.1

    # L0: aux CE loss sulla P branch
    loss_aux = criterion(x_extra_p, target)

    # L1: Binary Cross Entropy sulla D branch (bordi)
    x_boundary = torch.sigmoid(x_extra_d)
    loss_bce = F.binary_cross_entropy(x_boundary, boundary)

    # L2: main CE loss finale
    loss_main = criterion(x_main, target)

    # L3: CE loss focalizzata sui bordi
   # boundary_mask = (boundary.squeeze(1) > 0.5)
    boundary_mask = (boundary.squeeze(1) > 0.8) #sul paper
    masked_target = target[boundary_mask]
    valid_mask = (masked_target != 255)
    if valid_mask.any():
        loss_boundary_ce = criterion(
            x_main.permute(0,2,3,1)[boundary_mask][valid_mask],
            masked_target[valid_mask]
        )
    else:
        loss_boundary_ce = torch.tensor(0.0, device=target.device)

    # Loss finale pesata
    total_loss = (
        lambda_0 * loss_aux +
        lambda_1 * loss_bce +
        lambda_2 * loss_main +
        lambda_3 * loss_boundary_ce
    )

    return total_loss, {
        "loss_aux": loss_aux.item(),
        "loss_bce": loss_bce.item(),
        "loss_main": loss_main.item(),
        "loss_boundary_ce": loss_boundary_ce.item()
    }

# TRAIN LOOP
def train_pidnet(epoch, old_model, dataloader_train, criterion, optimizer, iteration, learning_rate, num_classes, max_iter): # criterion == loss function

    # 1. Obtain the pretrained model
    model = old_model 
    print("Training the model...")

    # 2. Initialize the metrics variables and hyperparameters
    print("Initializing the metrics variables...")
    
    running_loss_total = 0.0
    running_loss_aux = 0.0
    running_loss_bce = 0.0
    running_loss_main = 0.0
    running_loss_boundary_ce = 0.0
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)

    # 3. Start the training of the model
    print("Starting the training of the model...")
    model.train() 

    print(f"Training on {len(dataloader_train)} batches")
    #lambda_1 = 20* (0.9 ** (epoch / 10))  # exponential decay lambda_1
    if epoch <16:
        lambda_1 = 20
    else:
        lambda_1 = 1
    
    # 4. Loop on the batches of the dataset
    for batch_idx, (inputs, targets, file_names) in enumerate(dataloader_train): 
        if batch_idx % 100 == 0: # Print every 100 batches
            print(f"Batch {batch_idx}/{len(dataloader_train)}")

        iteration += 1 # Increment the iteration counter

        inputs, targets = inputs.cuda(), targets.cuda() # GPU
        x_p, x_final, x_d = model(inputs)
        x_p_up = F.interpolate(x_p, size=targets.shape[1:], mode='bilinear', align_corners=False)
        x_final_up = F.interpolate(x_final, size=targets.shape[1:], mode='bilinear', align_corners=False)
        x_d_up = F.interpolate(x_d, size=targets.shape[1:], mode='bilinear', align_corners=False)

        boundaries = get_boundary_map(targets)

        loss, loss_dict = compute_pidnet_loss(criterion,x_p_up, x_final_up, x_d_up, targets, boundaries, lambda_1=lambda_1)
        #print(f"Loss: {loss.item():.4f} | Aux Loss: {loss_dict['loss_aux']:.4f} | BCE Loss: {loss_dict['loss_bce']:.4f} | Main Loss: {loss_dict['loss_main']:.4f} | Boundary CE Loss: {loss_dict['loss_boundary_ce']:.4f}")
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the learning rate
        #### CHANGE HERE !!!!!!!!!!
        # CHOSE ONE OF THE TWO POLY DECAY BELOW
        lr = poly_lr_scheduler(optimizer, init_lr=learning_rate, iter=iteration, lr_decay_iter=1, max_iter=max_iter, power=0.9)
        lr = poly_lr_scheduler_warmup(optimizer, base_lr=learning_rate, curr_iter=iter, max_iter=max_iter, power=0.9,  warmup_iters=3000, warmup_start_lr=1e-6) # WARM_ITER = the number of iteration you want the warmup := 393*number_epochs


         # Update running losses
        running_loss_total += loss.item()
        running_loss_aux += loss_dict['loss_aux']
        running_loss_bce += loss_dict['loss_bce']
        running_loss_main += loss_dict['loss_main']
        running_loss_boundary_ce += loss_dict['loss_boundary_ce']

        # Convert model outputs to predicted class labels
        preds = x_final_up.argmax(dim=1).detach().cpu().numpy()
        gts = targets.detach().cpu().numpy()
        
        # Accumulate intersections and unions per class
        _, _, inters, unions = compute_miou(gts, preds, num_classes)
        total_intersections += inters
        total_unions += unions

    # 5. Compute the metrics for the training set 
    # 5.a Compute the standard metrics for all the epochs
    print("Computing the metrics for the training set...")

    iou_per_class = (total_intersections / (total_unions + 1e-10)) * 100
    iou_non_zero = np.array(iou_per_class)
    iou_non_zero = iou_non_zero[np.nonzero(iou_non_zero)]

    # Compute the mean without considering NaN value
    mean_iou = np.nanmean(iou_non_zero) 
    mean_loss_total = running_loss_total / len(dataloader_train)  
    mean_loss_aux = running_loss_aux / len(dataloader_train)
    mean_loss_bce = running_loss_bce / len(dataloader_train)
    mean_loss_main = running_loss_main / len(dataloader_train)
    mean_loss_boundary_ce = running_loss_boundary_ce / len(dataloader_train)  

    # 5.b Compute the computation metrics, i.e. FLOPs, latency, number of parameters (only at the last epoch)
    if epoch == 50:
            print("Computing the computation metrics...")

            mean_latency, std_latency, mean_fps, std_fps = compute_latency_and_fps(model, height=512, width=1024, iterations=1000)
            print(f"Latency: {mean_latency:.2f} ± {std_latency:.2f} ms | FPS: {mean_fps:.2f} ± {std_fps:.2f}")

            num_flops = compute_flops(model, height=512, width=1024)
            print(f"Total numer of FLOPS: {num_flops} GigaFLOPs")

            tot_params, trainable_params = compute_parameters(model)
            print(f"Total Params: {tot_params}, Trainable: {trainable_params}")

    else:
        # NB: metric = -1 means we have not computed it (we compute only at the last epoch)
        mean_latency = -1
        std_latency = -1
        num_flops = -1
        trainable_params = -1
        mean_fps = -1
        std_fps = -1



    # 6. Save the parameter of the model 
    print("Saving the model")

    wandb.log({
        "epoch": epoch,
        'mean_loss': mean_loss_total,
        'mean_loss_boundary_ce': mean_loss_boundary_ce,
        'mean_loss_aux': mean_loss_aux,
        'mean_loss_bce': mean_loss_bce,
        'mean_loss_main': mean_loss_main,
        "lr": lr
    })

    # Save the model weight at each epoch
    model_save_path = f"model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': mean_loss_total,
    }, model_save_path)

    # Create a new artefact for the current epoch
    artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
    artifact.add_file(model_save_path) 

    # Store the artefact on wandb
    wandb.log_artifact(artifact)
    print(f"Model saved for epoch {epoch}")

    # 7. Return all the metrics
    metrics = {
        'mean_loss': mean_loss_total,
        'mean_loss_aux': mean_loss_aux,
        'mean_loss_bce': mean_loss_bce,
        'mean_loss_main': mean_loss_main,
        'mean_loss_boundary_ce': mean_loss_boundary_ce, 
        'mean_iou': mean_iou,
        'iou_per_class': iou_per_class,
        'mean_latency' : mean_latency,
        'std_latency' : std_latency,
        'mean_fps' : mean_fps,
        'std_fps' : std_fps,
        'num_flops' : num_flops,
        'trainable_params': trainable_params
    }

    return metrics, iteration
