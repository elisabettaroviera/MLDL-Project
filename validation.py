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
from PIL import Image
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


def compute_pidnet_loss(criterion, x_extra_p, x_main, x_extra_d, target, boundary,
                        lambda_0=0.4, lambda_1=0.6, lambda_2=1.0, lambda_3=0.1):
    # L0: aux CE loss sulla P branch
    loss_aux = criterion(x_extra_p, target, ignore_index=255)

    # L1: Binary Cross Entropy sulla D branch (bordi)
    x_boundary = torch.sigmoid(x_extra_d)
    loss_bce = F.binary_cross_entropy(x_boundary, boundary)

    # L2: main CE loss finale
    loss_main = criterion(x_main, target, ignore_index=255)

    # L3: CE loss focalizzata sui bordi
    boundary_mask = (boundary.squeeze(1) > 0.5)
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

# VALIDATION LOOP
def validate_pidnet(epoch, new_model, val_loader, criterion, num_classes):


    # 1. Obtain the pretrained model 
    model = new_model
    print("Validating the model...")
    
    # 2. Initialize the metrics variables    
    print("Initializing the metrics variables...")
    running_loss_total = 0.0
    running_loss_aux = 0.0
    running_loss_bce = 0.0
    running_loss_main = 0.0
    running_loss_boundary_ce = 0.0
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)

    # 3. Start the validation of the model
    print("Starting the validation of the model...")
    model.eval()

    print(f"Validating on {len(val_loader)} batches") 
    

    # 4. Loop on the batches of the dataset
    with torch.no_grad(): # NOT compute the gradient (we already computed in the previous step)
        for batch_idx, (inputs, targets, file_names) in enumerate(val_loader): 
            if batch_idx % 100 == 0: # Print every 100 batches
                print(f"Batch {batch_idx}/{len(val_loader)}")

            inputs, targets = inputs.cuda(), targets.cuda() # GPU
            x_p, x_final, x_d = model(inputs)
            x_p_up = F.interpolate(x_p, size=targets.shape[1:], mode='bilinear', align_corners=False)
            x_final_up = F.interpolate(x_final, size=targets.shape[1:], mode='bilinear', align_corners=False)
            x_d_up = F.interpolate(x_d, size=targets.shape[1:], mode='bilinear', align_corners=False)

            boundaries = get_boundary_map(targets)

            loss, loss_dict = compute_pidnet_loss(criterion,x_p_up, x_final_up, x_d_up, targets, boundaries)
            #print(f"Loss: {loss.item():.4f} | Aux Loss: {loss_dict['loss_aux']:.4f} | BCE Loss: {loss_dict['loss_bce']:.4f} | Main Loss: {loss_dict['loss_main']:.4f} | Boundary CE Loss: {loss_dict['loss_boundary_ce']:.4f}")

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



    # 5. Compute the metrics for the validation set 
    # 5.a Compute the accuracy metrics, i.e. mIoU and mean loss
    print("Computing the metrics for the validation set...")

    iou_per_class = (total_intersections / (total_unions + 1e-10)) * 100
    iou_non_zero = np.array(iou_per_class)
    iou_non_zero = iou_non_zero[np.nonzero(iou_non_zero)]
    
    # Compute the mean without considering NaN value
    mean_iou = np.nanmean(iou_non_zero) 
    mean_loss_total = running_loss_total / len(val_loader)  
    mean_loss_aux = running_loss_aux / len(val_loader)
    mean_loss_bce = running_loss_bce / len(val_loader)
    mean_loss_main = running_loss_main / len(val_loader)
    mean_loss_boundary_ce = running_loss_boundary_ce / len(val_loader)  
    
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

    # 6. Return all the metrics
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

    return metrics