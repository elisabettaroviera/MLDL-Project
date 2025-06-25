import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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

    # Boundary CE on main predictions
    boundary_mask = boundary.squeeze(1).bool()
    masked_target = target[boundary_mask]
    valid = masked_target != 255
    if valid.any():
        pred_flat = x_main.permute(0,2,3,1)[boundary_mask][valid]
        gt_flat   = masked_target[valid]
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
    aux_losses, bce_losses, main_losses, boundary_losses = [], [], [], []

    # init lr
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    iterator = tqdm(dataloader, total=num_iter, desc="LR range test")
    for iteration, (x, y, _) in enumerate(iterator):
        if iteration >= num_iter:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        xp, xf, xd = model(x)
        # upsample
        xp = F.interpolate(xp, size=y.shape[1:], mode='bilinear', align_corners=False)
        xf = F.interpolate(xf, size=y.shape[1:], mode='bilinear', align_corners=False)
        xd = F.interpolate(xd, size=y.shape[1:], mode='bilinear', align_corners=False)

        boundary = get_boundary_map(y)
        loss, ld = compute_pidnet_loss(criterion, xp, xf, xd, y, boundary)

        # smoothing
        avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss.item()
        smoothed = avg_loss / (1 - smooth_beta ** (iteration + 1))

        # record
        lrs.append(lr)
        losses.append(smoothed)
        aux_losses.append(ld['loss_aux'])
        bce_losses.append(ld['loss_bce'])
        main_losses.append(ld['loss_main'])
        boundary_losses.append(ld['loss_boundary_ce'])

        # update
        loss.backward()
        optimizer.step()

        lr *= lr_mult
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # track
        if smoothed < best_loss:
            best_loss = smoothed
        if iteration > 0 and smoothed > 4 * best_loss:
            print("Loss diverged, stopping early.")
            break

        iterator.set_postfix({'lr': lr, 'loss': smoothed})

    # analyze
    lrs = np.array(lrs)
    losses = np.array(losses)
    min_idx = losses.argmin()
    # find first rise after min: where loss[i] > loss[i-1]
    rise_idx = min_idx + 1
    while rise_idx < len(losses) and losses[rise_idx] <= losses[rise_idx - 1]:
        rise_idx += 1
    # clip
    rise_idx = rise_idx if rise_idx < len(losses) else len(losses)-1

    a_lr = lrs[0] if min_idx == 0 else lrs[min_idx-1]
    b_lr = lrs[rise_idx]
    print(f"Min loss at lr={lrs[min_idx]:.2e}, golden interval ~ [{a_lr:.2e}, {b_lr:.2e}]")

    # plot
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
    # setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_PATH = '/kaggle/input/cityscapes-dataset/Cityscapes'  # <-- adjust

    # dataset & dataloader
    transform = transform_cityscapes()
    target_transform = transform_cityscapes_mask()
    ds = CityScapes(DATA_PATH, 'train', transform, target_transform)
    loader, _ = dataloader(ds, None, batch_size=4, shuffle_train=True, shuffle_val=False, drop_last_bach=True, num_workers=2)

    # model
    class CFG: pass
    cfg = CFG(); cfg.MODEL=type('',(),{})(); cfg.DATASET=type('',(),{})()
    cfg.MODEL.NAME = 'pidnet_m'
    cfg.MODEL.PRETRAINED = '/kaggle/input/pidnet-m/PIDNet_M_ImageNet.pth.tar'
    cfg.DATASET.NUM_CLASSES = 19
    model = get_seg_model(cfg, imgnet_pretrained=True).to(device)

    # criterion & optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=5e-4)

    # run
    lrs, losses, golden = lr_range_test(
        model, optimizer, loader, criterion, compute_pidnet_loss,
        init_lr=2e-5, final_lr=1e-1, num_iter=5000, device=device
    )
    print(f"Golden interval: {golden}")
