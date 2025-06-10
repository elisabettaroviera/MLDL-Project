import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from models.pidnet.PIDNET import PIDNet, get_seg_model
from datasets.cityscapes import CityScapes
from data.dataloader import dataloader
from datasets.transform_datasets import transform_cityscapes, transform_cityscapes_mask
from torch import nn
from utils.utils import CombinedLoss_All


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
    # Usiamo una media mobile per smussare la curva della loss
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

        # Calcola la media mobile smussata (smoothed moving average)
        # Il fattore 0.98 dà più peso alle loss precedenti, smussando le fluttuazioni
        avg_loss = 0.98 * avg_loss + 0.02 * total_loss.item() if iter_count > 0 else total_loss.item()
        smoothed_loss = avg_loss / (1 - 0.98 ** (iter_count + 1)) # Correzione per i primi valori

        lrs.append(lr)
        losses.append(smoothed_loss)

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        
        # Interrompi se la loss diventa troppo alta rispetto alla migliore finora
        if smoothed_loss > 4 * best_loss and iter_count > 10:
            print(f"Loss esplosa a iter {iter_count}, fermo il test.")
            break
            
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter_count % 50 == 0:
            print(f"Iter {iter_count}/{num_iters} | lr={lr:.6f} | loss={smoothed_loss:.4f}")

        iter_count += 1
        
    # --- NUOVA SEZIONE: CALCOLO DEI PUNTI DI INTERESSE ---
    
    # Rimuoviamo i primi e ultimi valori per evitare artefatti ai bordi
    lrs_plot = lrs[10:-5]
    losses_plot = losses[10:-5]
    
    # 1. Trova il LR con la loss minima
    min_loss_idx = np.argmin(losses_plot)
    lr_min_loss = lrs_plot[min_loss_idx]

    # 2. Trova il LR con la discesa più rapida (gradiente massimo)
    # Calcoliamo il gradiente della loss rispetto agli step
    grads = np.gradient(losses_plot)
    # Troviamo l'indice del gradiente più negativo (discesa più ripida)
    # Cerchiamo solo fino al punto di loss minima per evitare la parte in risalita
    steepest_descent_idx = np.argmin(grads[:min_loss_idx]) if min_loss_idx > 0 else np.argmin(grads)
    lr_steepest = lrs_plot[steepest_descent_idx]

    # Definiamo i punti per il plot
    lr_best = lr_steepest  # Il nostro candidato come LR migliore
    interval_a = lr_best
    interval_b = lr_min_loss

    print("\n--- Risultati Analisi LR ---")
    print(f"LR Migliore suggerito (discesa più rapida): {lr_best:.6f}")
    print(f"Intervallo d'oro suggerito (a, b): ({interval_a:.6f}, {interval_b:.6f})")
    print("-----------------------------\n")

    # --- SEZIONE PLOT MODIFICATA ---
    plt.figure(figsize=(12, 6))
    plt.plot(lrs, losses)
    
    # Plotta la X per il LR migliore
    plt.plot(lr_best, losses_plot[steepest_descent_idx], 'rX', markersize=12, label=f'LR Migliore: {lr_best:.6f}')
    
    # Plotta le linee verticali per l'intervallo
    plt.axvline(x=interval_a, color='r', linestyle='--', label=f'Intervallo d\'oro ({interval_a:.6f}, {interval_b:.6f})')
    plt.axvline(x=interval_b, color='r', linestyle='--')

    plt.xscale('log')
    plt.xlabel('Learning Rate (Scala Logaritmica)')
    plt.ylabel('Loss (Smussata)')
    plt.title('LR Range Test con Suggerimenti')
    plt.grid(True, which='both', linestyle='-')
    plt.legend()
    plt.savefig('lr_range_test.png')
    plt.show()

    np.savez('lr_range_data.npz', lrs=lrs, losses=losses)
    print("Salvato plot e dati in: lr_range_test.png e lr_range_data.npz")

    return lrs, losses


if __name__ == "__main__":
    print("Eseguendo LR Range Test per PIDNet su Cityscapes...")

    # Assicurati che la variabile d'ambiente sia impostata, altrimenti usa un default
    var_model = os.environ.get('MODEL', 'PIDNet')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in uso: {device}")

    # Preparazione dataset
    transform = transform_cityscapes()
    target_transform = transform_cityscapes_mask()
    cs_train = CityScapes('/kaggle/input/cityscapes-dataset/Cityscapes', 'train', transform, target_transform)
    # NOTA: Per un test LR veloce, potresti voler usare un sottoinsieme del dataset
    dataloader_cs_train, _ = dataloader(cs_train, None, 4, True, True, False)

    class CFG:
        pass

    cfg = CFG()
    cfg.MODEL = type('', (), {})()
    cfg.DATASET = type('', (), {})()

    cfg.MODEL.NAME = 'pidnet_m'
    cfg.MODEL.PRETRAINED = '/kaggle/input/pidnet-m-imagenet-pretrained-tar/PIDNet_M_ImageNet.pth.tar'
    cfg.DATASET.NUM_CLASSES = 19
    # Serve cosi chiamo pesi preaddestrati su ImageNet
    model = get_seg_model(cfg, imgnet_pretrained=True)
    model = model.to(device)

    num_classes = 19

    if var_model == 'PIDNet':
        print("MODELLO: PIDNet")
        # I valori specifici di lr, momentum etc. qui non sono usati per il test,
        # ma sono utili per la configurazione finale del training.
    
    # Istanzia modello PIDNet
    #model = PIDNet(num_classes=19, context_path='resnet18').to(device)

    # Definisci loss e optimizer
    # L'LR iniziale dell'optimizer (1e-6) sarà sovrascritto dalla funzione di test
    criterion = CombinedLoss_All(num_classes=num_classes, alpha=1.0, beta=0, gamma=0, theta=0, ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=5e-4)

    # Esegui il LR Range Test
    lrs, losses = lr_range_test(
        model, 
        dataloader_cs_train, 
        optimizer, 
        criterion, 
        lr_start=1e-5, # Un lr_start leggermente più alto potrebbe essere utile
        lr_end=1.0,     # Anche un lr_end più alto per vedere bene l'esplosione
        device=device
    )

    print("\nLR Range Test completato. Analizza il plot 'lr_range_test.png' per scegliere il LR ottimale!")