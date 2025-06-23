# main_val.py

import os
import random
import torch
import wandb
import time
import numpy as np
from datasets.cityscapes import CityScapes
from models.bisenet.build_bisenet import BiSeNet
from utils.utils import CombinedLoss_All,  save_metrics_on_wandb
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
    entity = "s281401-politecnico-di-torino" # New new entity Auro
    #entity = "s325951-politecnico-di-torino-mldl" # nuovo team Lucia
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

def to_obtain_artifact_names(project="", run_name=""):
    """
    Returns a sorted list of unique artifact names (e.g., model_epoch_1, model_epoch_2, ...)
    from a specific run in a wandb project, keeping only the latest version for each epoch.
    """
    entity = "s281401-politecnico-di-torino"
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    print("Available runs in project:")
    run_id = None
    for run in runs:
        print(f"Run name: '{run.name}', Run id: '{run.id}'")
        if run.name == run_name:
            run_id = run.id
    if run_id is None:
        raise ValueError(f"Run with name '{run_name}' not found in project '{project}'")
    # Now get the run by ID
    run = api.run(f"{entity}/{project}/{run_id}")
    artifacts = list(run.logged_artifacts())

    # Keep only the latest version for each epoch
    epoch_to_artifact = {}
    for artifact in artifacts:
        name = artifact.name
        if name.startswith("model_epoch_"):
            try:
                epoch = int(name.split("_")[2].split(":")[0])
                # If multiple versions, keep the one with the highest version number
                if (epoch not in epoch_to_artifact) or (artifact.version > epoch_to_artifact[epoch].version):
                    epoch_to_artifact[epoch] = artifact
            except Exception:
                continue
    # Sort by epoch
    sorted_artifacts = [epoch_to_artifact[e] for e in sorted(epoch_to_artifact)]
    artifact_names = [a.name for a in sorted_artifacts]
    print("Found", len(artifact_names), "unique artifacts.")
    return artifact_names


if __name__ == "__main__":
    set_seed(23)
    wandb.login(key="2bc32b7d4d8f8601d9a93be55631ae9e18f78690")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("************ VALIDATION ON CITYSCAPES ***************")

    transform_cityscapes_dataset = transform_cityscapes()
    target_transform_cityscapes = transform_cityscapes_mask()
    
    #if datsets are not saved on kaggle
    cs_val = CityScapes('./datasets/Cityscapes', 'val', transform_cityscapes_dataset, target_transform_cityscapes)
    # if datasets are saved on kaggle --> actually it doesn't work since its saving the images
    #cs_val = CityScapes('/kaggle/input/cityscapes-dataset/Cityscapes', 'val', transform_cityscapes_dataset, target_transform_cityscapes)


    batch_size = 4
    num_epochs = 25
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
    project_name = "4_Adversarial_Domain_Adaptation_bce_fixed_0002" #CHECK BEFORE RUNNING

    # Inserisci qui la lista degli id dei run, in ordine (epoch_1, epoch_2, ..., epoch_50)
    #run_ids = to_obtain_id(project_name)
    run_name = "epoch_1"
    entity = "s281401-politecnico-di-torino" # New new entity Auro
    api = wandb.Api()

    # Get sorted artifact names from the run
    artifact_names = to_obtain_artifact_names(project=project_name, run_name=run_name)

    
    for artifact_name in artifact_names:
        # Extract epoch number from artifact name
        try:
            epoch = int(artifact_name.split("_")[2].split(":")[0])
        except Exception:
            print(f"Could not extract epoch from artifact name: {artifact_name}")
            continue

        if epoch < start_epoch:
            continue  # Skip epochs before start_epoch

        print(f"\n[INFO] Downloading artifact: {project_name}/{artifact_name}")
        artifact = api.artifact(f"{entity}/{project_name}/{artifact_name}", type="model")
        artifact_dir = artifact.download()
        checkpoint_path = os.path.join(artifact_dir, f"model_epoch_{epoch}.pt")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Evaluating model from epoch {epoch}...")
        start_val = time.time()
        metrics_val = validate(epoch, model, dataloader_cs_val, loss, num_classes)
        end_val = time.time()
        print(f"Validation time: {(end_val - start_val)/60:.2f} min")

        print_metrics("Validation", metrics_val)
        with wandb.init(project=project_name, entity=entity, name=f"validation_epoch_{epoch}", resume="allow"):
            save_metrics_on_wandb(epoch, metrics_train=None, metrics_val=metrics_val)
            wandb.finish()