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
    # target: (B, H, W) con valori interi [0, num_classes-1] oppure 255 per ignorare

    # Creiamo una maschera che esclude i pixel 255
    valid_mask = (target != 255).float()  # 1 dove valido, 0 dove 255

    target = target.clone().float()  # per non modificare l'originale
    target[target == 255] = 0  # sostituisci 255 con 0 (classe qualsiasi)

    target = target.unsqueeze(1)  # (B,1,H,W)

    laplace_kernel = torch.tensor([[[[0, 1, 0],
                                     [1,-4, 1],
                                     [0, 1, 0]]]], device=target.device).float()

    boundary = F.conv2d(target, laplace_kernel, padding=1).abs()

    # Applichiamo la maschera di validità anche al risultato per azzerare bordi in pixel ignorati
    boundary = boundary * valid_mask.unsqueeze(1)  # broadcast su canale

    boundary = (boundary > 0).float()

    return boundary
def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss



def compute_pidnet_loss(criterion, x_extra_p, x_main, x_extra_d, target, boundary,
                        lambda_0=0.4, lambda_1=20.0, lambda_2=1.0, lambda_3=1.0):
    """
    Calcola la loss totale di PIDNet composta da:
    - CE aux branch
    - BCE pesata sui bordi
    - CE sulla main branch
    - CE focalizzata sui bordi

    Args:
        criterion: funzione CE standard (es. nn.CrossEntropyLoss(ignore_index=255))
        x_extra_p: output dalla branch P (B, C, H, W)
        x_main: output dalla main branch (B, C, H, W)
        x_extra_d: output dalla branch D (B, 1, H, W)
        target: ground truth segmentazione (B, H, W)
        boundary: mappa binaria bordi (B, 1, H, W)

    Returns:
        total_loss: somma pesata delle quattro componenti
        losses_dict: dizionario con le singole componenti
    """

    # L0: CE ausiliaria sulla branch P
    loss_aux = criterion(x_extra_p, target)

    # L1: BCE pesata sulla branch D (bordi)
    loss_bce = weighted_bce(x_extra_d, boundary)

    # L2: CE principale sulla branch main
    loss_main = criterion(x_main, target)

    # L3: CE focalizzata solo sui pixel al contorno
    boundary_mask = (torch.sigmoid(x_extra_d).squeeze(1) > 0.8)
    masked_target = target[boundary_mask]
    valid_mask = (masked_target != 255)

    if valid_mask.any():
        masked_output = x_main.permute(0, 2, 3, 1)[boundary_mask][valid_mask]
        masked_target = masked_target[valid_mask]
        loss_boundary_ce = criterion(masked_output, masked_target)
    else:
        loss_boundary_ce = torch.tensor(0.0, device=target.device)

    # Combinazione pesata finale
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
    lambda_1 = 20* (0.9 ** (epoch / 10))  # exponential decay lambda_1


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

            loss, loss_dict = compute_pidnet_loss(criterion,x_p_up, x_final_up, x_d_up, targets, boundaries, lambda_1=lambda_1)
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