# main_val.py

import os
import torch
import wandb
import time
import numpy as np
from datasets.cityscapes import CityScapes
from models.bisenet.build_bisenet import BiSeNet
from utils.utils import CombinedLoss_All, save_metrics_on_file, save_metrics_on_wandb, set_seed, print_metrics
from datasets.transform_datasets import transform_cityscapes, transform_cityscapes_mask
from data.dataloader import dataloader
from validation import validate

if __name__ == "__main__":
    set_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("************ VALIDATION ON CITYSCAPES ***************")

    transform_cityscapes_dataset = transform_cityscapes()
    target_transform_cityscapes = transform_cityscapes_mask()

    cs_val = CityScapes('./datasets/Cityscapes', 'val', transform_cityscapes_dataset, target_transform_cityscapes)

    batch_size = 4
    num_classes = 19
    ignore_index = 255
    loss = CombinedLoss_All(num_classes=num_classes, alpha=0.7, beta=0, gamma=0.3, theta=0, ignore_index=255)

    _, dataloader_cs_val = dataloader(None, cs_val, batch_size, shuffle_train=False, shuffle_val=False)

    model = BiSeNet(num_classes=num_classes, context_path='resnet18').to(device)

    project_name = "3b_GTA5_to_CITY_augmented_geometric_cv07_tv_03"
    run = wandb.init(project=project_name, entity="s328422-politecnico-di-torino", name="BiSeNet_GTA5_Validation", reinit=True)

    # Epoch to validate
    epoch = 50  # PUT HERE THE EPOCH YOU WANT TO VALIDATE - THE EXACT NUMBER OF THE EPOCHE THAT YOU WANT TO VALIDATE
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
    save_metrics_on_file(epoch, metrics_train=None, metrics_val=metrics_val)

    wandb.finish()
