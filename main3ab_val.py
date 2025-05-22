# main_val.py

import os
import random
import torch
import wandb
import time
import numpy as np
from datasets.cityscapes import CityScapes
from models.bisenet.build_bisenet import BiSeNet
from utils.utils import CombinedLoss_All, save_metrics_on_file, save_metrics_on_wandb
from datasets.transform_datasets import transform_cityscapes, transform_cityscapes_mask
from data.dataloader import dataloader
from validation import validate

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def print_metrics(title, metrics):
    print(f"{title} Metrics")
    print(f"Loss: {metrics['mean_loss']:.4f}")
    print(f"Latency: {metrics['mean_latency']:.2f} ms")
    print(f"FLOPs: {metrics['num_flops']:.2f} GFLOPs")
    print(f"Parameters: {metrics['trainable_params']:.2f} M")
    print(f"Mean IoU (mIoU): {metrics['mean_iou']:.2f} %\n")
    print("Class-wise IoU (%):")
    print(f"{'Class':<20} {'IoU':>6}")
    print("-" * 28)
    for cls, val in enumerate(metrics['iou_per_class']):
        print(f"{cls:<20} {val:>6.2f}")

def to_obtain_id(project=""):
    # Configurazione del tuo progetto wandb
    entity = "s325951-politecnico-di-torino-mldl" # nuovo team Lucia
    # entity = "s328422-politecnico-di-torino"

    api = wandb.Api()

    # Recupera tutte le run del progetto
    runs = api.runs(f"{entity}/{project}")

    # Funzione per estrarre il numero dell'epoca dal nome della run
    def extract_epoch_number(run):
        try:
            name = run.name
            if name.startswith("epoch_"):
                return int(name.split("_")[1])
        except:
            return float("inf")
        return float("inf")

    # Filtra e ordina le run per numero di epoca
    sorted_runs = sorted(
        [run for run in runs if run.name and run.name.startswith("epoch_")],
        key=extract_epoch_number
    )

    # Crea la lista degli ID delle run ordinate
    run_ids = [run.id for run in sorted_runs]

    # Ora puoi usare run_ids come vuoi, ad esempio:
    print("Ho caricato", len(run_ids), "run ID.")
    # Esempio: passare run_ids a una funzione
    return run_ids

if __name__ == "__main__":
    set_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("************ VALIDATION ON CITYSCAPES ***************")

    transform_cityscapes_dataset = transform_cityscapes()
    target_transform_cityscapes = transform_cityscapes_mask()

    cs_val = CityScapes('./datasets/Cityscapes', 'val', transform_cityscapes_dataset, target_transform_cityscapes)

    batch_size = 4
    num_epochs = 15 #CHECK BEFORE RUNNING
    num_classes = 19
    ignore_index = 255
    start_epoch = 1
    loss = CombinedLoss_All(num_classes=num_classes, alpha=0.7, beta=0, gamma=0.3, theta=0, ignore_index=255) #CHECK BEFORE RUNNING
    """
    alpha   # CrossEntropy
    beta    # Lovász
    gamma   # Tversky
    theta   # Dice
    """

    _, dataloader_cs_val = dataloader(None, cs_val, batch_size, shuffle_train=False, shuffle_val=False)

    model = BiSeNet(num_classes=num_classes, context_path='resnet18').to(device)

    project_name = "3b_GTA5_to_CITY_aug_color_a_c_25percent" #CHECK BEFORE RUNNING

    # Inserisci qui la lista degli id dei run, in ordine (epoch_1, epoch_2, ..., epoch_50)
    run_ids = to_obtain_id(project_name)

    for epoch in range(start_epoch, num_epochs + 1):
        run = wandb.init(
            project=project_name,
            entity = "s325951-politecnico-di-torino-mldl", # nuovo team Lucia
            # entity="s328422-politecnico-di-torino",
            name=f"epoch_{epoch}",
            id=run_ids[epoch - 1],  # <-- INDICE CORRETTO!
            resume="allow"
        )
        artifact = wandb.use_artifact(f"{project_name}/model_epoch_{epoch}:latest", type="model")
        artifact_path = artifact.download()
        checkpoint_path = os.path.join(artifact_path, f"model_epoch_{epoch}.pt")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Evaluating model from epoch {epoch}...")
        start_val = time.time()
        metrics_val = validate(epoch, model, dataloader_cs_val, loss, num_classes)
        end_val = time.time()
        print(f"Validation time: {(end_val - start_val)/60:.2f} min")

        print_metrics("Validation", metrics_val)
        save_metrics_on_wandb(epoch, metrics_train=None, metrics_val=metrics_val)
        
        wandb.finish()