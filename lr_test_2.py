import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_boundary_map
from tqdm import tqdm

import os
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# Necessary functions and classes (moved here to make the script self-contained)
# We assume that the import paths are correct in your environment.
from models.pidnet.PIDNET import get_seg_model
from datasets.cityscapes import CityScapes
from data.dataloader import dataloader
from datasets.transform_datasets import transform_cityscapes, transform_cityscapes_mask

def get_boundary_map(target, kernel_size=3):
    """
    Generates a binary boundary map from the segmentation mask.
    """
    target_float = target.unsqueeze(1).float()
    laplace_kernel = torch.tensor(
        [[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], 
        device=target.device, dtype=torch.float32
    )
    boundary = F.conv2d(target_float, laplace_kernel, padding=1).abs()
    boundary = (boundary > 0).float()
    return boundary

def compute_pidnet_loss(criterion_ce, x_extra_p, x_main, x_extra_d, target, boundary,
                        lambda_0=0.4, lambda_1=20.0, lambda_2=1.0, lambda_3=1.0):
    """
    Calculates the combined loss for PIDNet as defined in the paper and your training loop.
    """
    # L0: Auxiliary CrossEntropy loss on the P branch
    loss_aux = criterion_ce(x_extra_p, target)

    # L1: Binary Cross Entropy on the D branch (for boundaries)
    loss_bce = F.binary_cross_entropy_with_logits(x_extra_d, boundary)

    # L2: Main CrossEntropy loss on the final output
    loss_main = criterion_ce(x_main, target)

    # L3: Boundary-focused CrossEntropy loss
    boundary_mask = (boundary.squeeze(1) > 0.8)
    masked_target = target[boundary_mask]
    valid_mask = (masked_target != 255) # ignore_index
    
    if valid_mask.any():
        # Apply loss only on valid pixels of the boundary region
        loss_boundary_ce = criterion_ce(
            x_main.permute(0, 2, 3, 1)[boundary_mask][valid_mask],
            masked_target[valid_mask]
        )
    else:
        loss_boundary_ce = torch.tensor(0.0, device=target.device)

    # Weighted total loss
    total_loss = (
        lambda_0 * loss_aux +
        lambda_1 * loss_bce +
        lambda_2 * loss_main +
        lambda_3 * loss_boundary_ce
    )
    # For the range test, we only need the total loss
    return total_loss

# ==============================================================================
# UPDATED LR RANGE TEST FUNCTION
# ==============================================================================


def lr_range_test(
    model,
    optimizer,
    dataloader,
    criterion,
    compute_pidnet_loss,
    init_lr=1e-6,
    final_lr=1,
    num_iter=800,
    device='cuda',
    smooth_beta=0.98
):
    model.train()
    model.to(device)

    # Inizializza log
    lrs = []
    losses = []
    loss_auxes = []
    loss_bces = []
    loss_mains = []
    loss_boundary_ces = []

    best_loss = float('inf')
    avg_loss = 0.0
    lr_mult = (final_lr / init_lr) ** (1 / num_iter)
    lr = init_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    iter_count = 0
    pbar = tqdm(dataloader, desc="LR Range Test")

    for batch in pbar:
        if iter_count >= num_iter:
            break

        inputs, targets, _ = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        x_p, x_final, x_d = model(inputs)

        # Interpola le predizioni come nel train
        x_p_up = F.interpolate(x_p, size=targets.shape[1:], mode='bilinear', align_corners=False)
        x_final_up = F.interpolate(x_final, size=targets.shape[1:], mode='bilinear', align_corners=False)
        x_d_up = F.interpolate(x_d, size=targets.shape[1:], mode='bilinear', align_corners=False)

        # Bordo
        boundary = get_boundary_map(targets)

        # Loss
        loss, loss_dict = compute_pidnet_loss(
            criterion, x_p_up, x_final_up, x_d_up, targets, boundary
        )

        # Calcola smooth loss (media esponenziale)
        avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss.item()
        smoothed = avg_loss / (1 - smooth_beta ** (iter_count + 1))

        lrs.append(lr)
        losses.append(smoothed)
        loss_auxes.append(loss_dict['loss_aux'])
        loss_bces.append(loss_dict['loss_bce'])
        loss_mains.append(loss_dict['loss_main'])
        loss_boundary_ces.append(loss_dict['loss_boundary_ce'])

        # Backprop
        loss.backward()
        optimizer.step()

        # Aumenta il LR
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        iter_count += 1
        pbar.set_postfix(lr=lr, loss=smoothed)

        # Early stop se la loss esplode
        if iter_count > 10 and smoothed > 4 * best_loss:
            print("Loss esplosa, fermo il test.")
            break

        if smoothed < best_loss or iter_count == 1:
            best_loss = smoothed

    # Plot finale
    plt.figure(figsize=(12, 7))
    plt.plot(lrs, losses, label='Total Loss')
    plt.plot(lrs, loss_auxes, label='Aux Loss')
    plt.plot(lrs, loss_bces, label='BCE Loss')
    plt.plot(lrs, loss_mains, label='Main CE Loss')
    plt.plot(lrs, loss_boundary_ces, label='Boundary CE Loss')
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Range Test (PIDNet Loss)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return lrs, losses

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

if __name__ == "__main__":
    print("Running LR Range Test for PIDNet on Cityscapes...")

    var_model = os.environ.get('MODEL', 'PIDNet')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset preparation
    transform = transform_cityscapes()
    target_transform = transform_cityscapes_mask()
    # Change the path to your actual path
    DATASET_PATH = '/kaggle/input/cityscapes-dataset/Cityscapes' # <-- CHANGE IF NEEDED
    cs_train = CityScapes(DATASET_PATH, 'train', transform, target_transform)
    dataloader_cs_train, _ = dataloader(cs_train, None, batch_size=4, shuffle_train=True, shuffle_val=True, drop_last_bach=True, num_workers=2)
   
    # Configuration to load the PIDNet model
    class CFG: pass
    cfg = CFG()
    cfg.MODEL = type('', (), {})()
    cfg.DATASET = type('', (), {})()
    cfg.MODEL.NAME = 'pidnet_s' # or 'pidnet_s', 'pidnet_l'
    # Change the path to your actual path
    cfg.MODEL.PRETRAINED = '/kaggle/input/pidnet-s/PIDNet_S_ImageNet.pth.tar' # <-- CHANGE IF NEEDED
    cfg.DATASET.NUM_CLASSES = 19
    
    # Instantiate the model
    model = get_seg_model(cfg, imgnet_pretrained=True) # Set True if you use pretrained weights
    model = model.to(device)

    # --- CORRECT CRITERION ---
    # The `compute_pidnet_loss` function expects a base CE loss.
    # A custom combined loss was not correct for this logic.
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=5e-4)

    # Execute the LR Range Test

    lr_range_test(
    model=model,
    optimizer=optimizer,
    dataloader=dataloader_cs_train,
    criterion=criterion,
    compute_pidnet_loss=compute_pidnet_loss,
    init_lr=5e-5,
    final_lr=5e-2,
    num_iter=800,
    device=device
)

    print("\nLR Range Test completed. Analyze the plot 'lr_range_test_pidnet.png' to choose the optimal LR!")