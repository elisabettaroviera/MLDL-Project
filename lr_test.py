import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from itertools import cycle

from models.pidnet.PIDNET import get_seg_model
from datasets.cityscapes import CityScapes
from data.dataloader import dataloader
from datasets.transform_datasets import transform_cityscapes, transform_cityscapes_mask


def get_boundary_map(target, kernel_size=3):
    target_float = target.unsqueeze(1).float()
    laplace_kernel = torch.tensor(
        [[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
        device=target.device, dtype=torch.float32
    )
    boundary = F.conv2d(target_float, laplace_kernel, padding=1).abs()
    return (boundary > 0).float()


def compute_pidnet_loss(criterion_ce, x_extra_p, x_main, x_extra_d, target, boundary,
                        lambda_0=0.4, lambda_1=20.0, lambda_2=1.0, lambda_3=1.0):
    loss_aux = criterion_ce(x_extra_p, target)
    loss_bce = F.binary_cross_entropy_with_logits(x_extra_d, boundary)
    loss_main = criterion_ce(x_main, target)

    boundary_mask = boundary.squeeze(1).bool()
    masked_target = target[boundary_mask]
    valid = masked_target != 255
    if valid.any():
        pred_flat = x_main.permute(0,2,3,1)[boundary_mask][valid]
        gt_flat = masked_target[valid]
        loss_boundary_ce = criterion_ce(pred_flat, gt_flat)
    else:
        loss_boundary_ce = torch.tensor(0., device=target.device)

    total = lambda_0 * loss_aux + lambda_1 * loss_bce + lambda_2 * loss_main + lambda_3 * loss_boundary_ce
    return total, {
        'loss_aux': loss_aux.item(),
        'loss_bce': loss_bce.item(),
        'loss_main': loss_main.item(),
        'loss_boundary_ce': loss_boundary_ce.item()
    }


def lr_range_test(model, optimizer, dataloader, criterion, compute_pidnet_loss,
                  init_lr=1e-6, final_lr=1, num_iter=100, device='cuda', smooth_beta=0.98):
    model.train().to(device)
    lr = init_lr
    lr_mult = (final_lr / init_lr) ** (1 / num_iter)

    avg_loss = 0.
    best_loss = float('inf')

    lrs, losses = [], []

    # init lr
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    data_iter = cycle(dataloader)
    pbar = tqdm(range(num_iter), desc="LR range test")

    for iteration in pbar:
        x, y, _ = next(data_iter)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        xp, xf, xd = model(x)
        xp = F.interpolate(xp, size=y.shape[1:], mode='bilinear', align_corners=False)
        xf = F.interpolate(xf, size=y.shape[1:], mode='bilinear', align_corners=False)
        xd = F.interpolate(xd, size=y.shape[1:], mode='bilinear', align_corners=False)

        boundary = get_boundary_map(y)
        loss, ld = compute_pidnet_loss(criterion, xp, xf, xd, y, boundary)

        avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss.item()
        smoothed = avg_loss / (1 - smooth_beta ** (iteration + 1))

        lrs.append(lr)
        losses.append(smoothed)

        loss.backward()
        optimizer.step()

        lr *= lr_mult
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if smoothed < best_loss:
            best_loss = smoothed
        elif iteration > 10 and smoothed > 4 * best_loss:
            print("Loss diverged, stopping early.")
            break

        pbar.set_postfix({'lr': f"{lr:.2e}", 'loss': f"{smoothed:.2f}"})

    lrs = np.array(lrs)
    losses = np.array(losses)
    min_idx = losses.argmin()
    rise_idx = min_idx + 1
    while rise_idx < len(losses) and losses[rise_idx] <= losses[rise_idx - 1]:
        rise_idx += 1
    if rise_idx >= len(losses):
        rise_idx = len(losses) - 1

    a_lr = lrs[min_idx - 1] if min_idx > 0 else lrs[0]
    b_lr = lrs[rise_idx]
    print(f"Golden interval: [{a_lr:.2e}, {b_lr:.2e}]")

    plt.figure(figsize=(10,6))
    plt.plot(lrs, losses, label='total loss')
    plt.xscale('log')
    plt.axvline(lrs[min_idx], color='green', linestyle='--', label='min loss')
    plt.axvspan(a_lr, b_lr, color='grey', alpha=0.3, label='golden interval')
    plt.xlabel('LR (log)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_range_test_pidnet.png', dpi=300)
    plt.close()

    return lrs, losses, (a_lr, b_lr)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_PATH = '/kaggle/input/cityscapes-dataset/Cityscapes'

    transform = transform_cityscapes()
    target_transform = transform_cityscapes_mask()
    ds = CityScapes(DATA_PATH, 'train', transform, target_transform)
    loader, _ = dataloader(ds, None, batch_size=4, shuffle_train=True, shuffle_val=False, drop_last_bach=True, num_workers=2)

    class CFG: pass
    cfg = CFG(); cfg.MODEL=type('',(),{})(); cfg.DATASET=type('',(),{})()
    cfg.MODEL.NAME = 'pidnet_m'
    cfg.MODEL.PRETRAINED = '/kaggle/input/pidnet-m/PIDNet_M_ImageNet.pth.tar'
    cfg.DATASET.NUM_CLASSES = 19
    model = get_seg_model(cfg, imgnet_pretrained=True).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=5e-4)

    lrs, losses, golden = lr_range_test(
        model, optimizer, loader, criterion, compute_pidnet_loss,
        init_lr=2e-5, final_lr=1e-1, num_iter=5000, device=device
    )
    print(f"Golden interval: [{golden[0]:.2e}, {golden[1]:.2e}]")
