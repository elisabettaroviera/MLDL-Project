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

# ==============================================================================
# HELPER FUNCTIONS FOR PIDNET LOSS
# (Taken from your training script for consistency)
# ==============================================================================

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
    dataloader_train,
    optimizer,
    criterion, # This will be the base CrossEntropyLoss
    lr_start=1e-6,
    lr_end=0.1,
    num_iters=None,
    device='cuda',
):
    model.train()
    lrs = []
    losses = []

    if num_iters is None:
        num_iters = len(dataloader_train) * 2

    lr_mult = (lr_end / lr_start) ** (1 / num_iters)
    lr = lr_start
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    iter_count = 0
    avg_loss = 0.
    best_loss = float('inf')
    dataloader_iter = iter(dataloader_train)

    while iter_count < num_iters:
        try:
            inputs, targets, _ = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader_train)
            inputs, targets, _ = next(dataloader_iter)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # --- LOSS CALCULATION SECTION MODIFIED FOR PIDNET ---
        x_p, x_final, x_d = model(inputs)
        
        # Interpolate outputs to the target's size
        x_p_up = F.interpolate(x_p, size=targets.shape[1:], mode='bilinear', align_corners=False)
        x_final_up = F.interpolate(x_final, size=targets.shape[1:], mode='bilinear', align_corners=False)
        x_d_up = F.interpolate(x_d, size=targets.shape[1:], mode='bilinear', align_corners=False)
        
        # Create the boundary map
        boundaries = get_boundary_map(targets)
        
        # Calculate the total loss using the specific PIDNet function
        total_loss = compute_pidnet_loss(criterion, x_p_up, x_final_up, x_d_up, targets, boundaries)
        # --- END OF MODIFIED SECTION ---
        
        total_loss.backward()
        optimizer.step()

        # Calculate smoothed moving average for the loss curve
        avg_loss = 0.98 * avg_loss + 0.02 * total_loss.item() if iter_count > 0 else total_loss.item()
        smoothed_loss = avg_loss / (1 - 0.98 ** (iter_count + 1))

        lrs.append(lr)
        losses.append(smoothed_loss)

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        
        if smoothed_loss > 2.5 * best_loss and iter_count > 20:
            print(f"Loss exploded at iter {iter_count}, stopping the test.")
            break
            
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter_count % 50 == 0:
            print(f"Iter {iter_count}/{num_iters} | lr={lr:.6f} | loss={smoothed_loss:.4f}")

        iter_count += 1
        
    # --- Analysis and Plotting (unchanged) ---
    print("\nAnalyzing and plotting results...")
    lrs_plot = lrs[10:-5]
    losses_plot = losses[10:-5]
    
    if not losses_plot:
        print("Test stopped too early, cannot generate plot.")
        return lrs, losses
        
    min_loss_idx = np.argmin(losses_plot)
    lr_min_loss = lrs_plot[min_loss_idx]

    grads = np.gradient(losses_plot)
    steepest_descent_idx = np.argmin(grads[:min_loss_idx]) if min_loss_idx > 0 else np.argmin(grads)
    lr_steepest = lrs_plot[steepest_descent_idx]

    lr_best, interval_a, interval_b = lr_steepest, lr_steepest, lr_min_loss

    print("\n--- LR Analysis Results ---")
    print(f"Suggested Best LR (steepest descent): {lr_best:.6f}")
    print(f"Suggested Golden Range (a, b): ({interval_a:.6f}, {interval_b:.6f})")
    print("---------------------------\n")

    plt.figure(figsize=(12, 6))
    plt.plot(lrs, losses)
    plt.plot(lr_best, losses_plot[steepest_descent_idx], 'rX', markersize=12, label=f'Best LR: {lr_best:.6f}')
    plt.axvline(x=interval_a, color='r', linestyle='--', label=f'Golden Range ({interval_a:.6f}, {interval_b:.6f})')
    plt.axvline(x=interval_b, color='r', linestyle='--')
    plt.xscale('log')
    plt.xlabel('Learning Rate (Log Scale)')
    plt.ylabel('Loss (Smoothed)')
    plt.title('LR Range Test with Suggestions for PIDNet')
    plt.grid(True, which='both', linestyle='-')
    plt.legend()
    plt.savefig('lr_range_test_pidnet.png')
    plt.show()

    np.savez('lr_range_data_pidnet.npz', lrs=lrs, losses=losses)
    print("Saved plot and data to: lr_range_test_pidnet.png and lr_range_data_pidnet.npz")

    return lrs, losses


# lr_range_test_pidnet.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.nn.functional import cross_entropy
from collections import deque

def lr_range_test_2(model, dataloader, device='cuda', lr_start=5e-5, lr_end=5e-2, num_iters=200, smoothing=0.98):
    model.train()
    model.to(device)

    optimizer = SGD(model.parameters(), lr=lr_start, momentum=0.9, weight_decay=5e-4)
    lr_lambda = lambda x: (lr_end / lr_start) ** (x / num_iters)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    losses = []
    smoothed_losses = []
    lrs = []

    avg_loss = 0.0
    best_loss = float('inf')

    iterator = iter(dataloader)
    for iteration in range(num_iters):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            images, targets = next(iterator)

        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Adjust this if your model has multiple outputs
        if isinstance(outputs, dict):
            main_loss = cross_entropy(outputs['main'], targets)
            aux_loss = cross_entropy(outputs['aux'], targets) if 'aux' in outputs else 0
            detail_loss = cross_entropy(outputs['detail'], targets) if 'detail' in outputs else 0
            loss = main_loss + 0.4 * aux_loss + 0.4 * detail_loss
        else:
            loss = cross_entropy(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        lrs.append(optimizer.param_groups[0]["lr"])
        losses.append(loss.item())

        # Smooth loss
        avg_loss = smoothing * avg_loss + (1 - smoothing) * loss.item()
        debiased_loss = avg_loss / (1 - smoothing ** (iteration + 1))
        smoothed_losses.append(debiased_loss)

        if debiased_loss < best_loss or iteration == 0:
            best_loss = debiased_loss
            best_lr = optimizer.param_groups[0]["lr"]

    # Save data
    np.savez("lr_range_data_pidnet.npz", lrs=np.array(lrs), losses=np.array(losses), smoothed_losses=np.array(smoothed_losses))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(lrs, smoothed_losses, label="Smoothed Loss", color='blue')
    plt.plot(lrs, losses, label="Raw Loss", color='gray', alpha=0.4)
    plt.scatter(best_lr, best_loss, color='red', marker='x', s=100, label=f"Best LR: {best_lr:.6f}")
    plt.axvline(best_lr, color='red', linestyle='--')
    plt.title("LR Range Test with Suggestions for PIDNet")
    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Loss")
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nSuggested Learning Rate: {best_lr:.6f}")
    return best_lr

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
    cfg.MODEL.NAME = 'pidnet_l' # or 'pidnet_s', 'pidnet_l'
    # Change the path to your actual path
    cfg.MODEL.PRETRAINED = '/kaggle/input/pidnet-l/PIDNet_L_ImageNet.pth.tar' # <-- CHANGE IF NEEDED
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
    lrs, losses = lr_range_test_2(
        model, 
        dataloader_cs_train, 
        optimizer, 
        criterion, 
        lr_start=5e-5,
        lr_end=5e-2,
        device=device
    )

    print("\nLR Range Test completed. Analyze the plot 'lr_range_test_pidnet.png' to choose the optimal LR!")