# lr_range_test_pidnet.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.nn.functional import cross_entropy
from collections import deque

def lr_range_test(model, dataloader, device='cuda', lr_start=1e-5, lr_end=5e-2, num_iters=200, smoothing=0.98):
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
