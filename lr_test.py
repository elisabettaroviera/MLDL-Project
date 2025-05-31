import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from models.bisenet.build_bisenet import BiSeNet
from datasets.cityscapes import CityScapes
from data.dataloader import dataloader
from datasets.transform_datasets import transform_cityscapes, transform_cityscapes_mask
from torch import nn


def lr_range_test(
    model,
    dataloader_train,
    optimizer,
    criterion,
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

        outputs = model(inputs)
        loss_main = criterion(outputs[0], targets)
        loss_aux1 = criterion(outputs[1], targets)
        loss_aux2 = criterion(outputs[2], targets)
        total_loss = loss_main + (loss_aux1 + loss_aux2)

        total_loss.backward()
        optimizer.step()

        avg_loss = 0.98 * avg_loss + 0.02 * total_loss.item() if iter_count > 0 else total_loss.item()

        lrs.append(lr)
        losses.append(avg_loss)

        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if avg_loss > 4 * best_loss:
            print(f"Loss esplosa a iter {iter_count}, fermo il test.")
            break
        if avg_loss < best_loss or iter_count == 0:
            best_loss = avg_loss

        if iter_count % 50 == 0:
            print(f"Iter {iter_count}/{num_iters} | lr={lr:.6f} | loss={avg_loss:.4f}")

        iter_count += 1

    plt.figure()
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('LR Range Test')
    plt.grid(True)
    plt.savefig('lr_range_test.png')
    plt.show()

    np.savez('lr_range_data.npz', lrs=lrs, losses=losses)
    print("Salvato plot e dati in: lr_range_test.png e lr_range_data.npz")

    return lrs, losses


if __name__ == "__main__":
    print("Eseguendo LR Range Test per BiSeNet su Cityscapes...")

    # Ambient variable
    var_model = os.environ['MODEL'] #'DeepLabV2' OR 'BiSeNet' # CHOOSE the model to train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preparazione dataset
    transform = transform_cityscapes()
    target_transform = transform_cityscapes_mask()
    cs_train = CityScapes('./datasets/Cityscapes', 'train', transform, target_transform)
    dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, None, 4, True, True, False)

    if var_model == 'BiSeNet':
        print("MODEL BISENET")
        batch_size = 4 # Bach size
        learning_rate = 0.00625 # Learning rate for the optimizer - 1e-4
        momentum = 0.9 # Momentum for the optimizer
        weight_decay = 1e-4 # Weight decay for the optimizer


    # Istanzia modello BiSeNet
    model = BiSeNet(num_classes=19, context_path='resnet18').to(device)

    # Definisci loss e optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=1e-4)

    # Esegui il LR Range Test
    lrs, losses = lr_range_test(model, dataloader_cs_train, optimizer, criterion, device=device)

    print("LR Range Test completato. Guarda il plot 'lr_range_test.png' e scegli il LR ottimale!")
