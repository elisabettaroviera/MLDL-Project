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

# Function to set the seed for reproducibility
# This function sets the seed for various libraries to ensure that the results are reproducible.
def set_seed(seed):
    torch.manual_seed(seed) # Set the seed for CPU
    torch.cuda.manual_seed(seed) # Set the seed for CPU
    torch.cuda.manual_seed_all(seed) # Set the seed for all GPUs
    np.random.seed(seed) # Set the seed for NumPy
    random.seed(seed) # Set the seed for random
    torch.backends.cudnn.benchmark = True # Enable auto-tuning for max performance
    torch.backends.cudnn.deterministic = False # Allow non-deterministic algorithms for better performance
    #A.set_seed(seed) # ATTENZIONE serve anche se abbiamo messo come seed gli id delle foto???

# Function to print the metrics
# This function print various metrics such as latency, FPS, FLOPs, parameters, and mIoU for a given model and dataset
def print_metrics(title, metrics):
    # NB: this is how the metrics dictionary returned in train is defined
    # metrics = {
    #    'mean_loss': mean_loss,
    #    'mean_iou': mean_iou,
    #    'iou_per_class': iou_per_class,
    #    'mean_latency' : mean_latency,
    #    'num_flops' : num_flops,
    #    'trainable_params': trainable_params}
    
    print(f"{title} Metrics")
    print(f"Loss: {metrics['mean_loss']:.4f}")
    print(f"Latency: {metrics['mean_latency']:.2f} ms")
    #print(f"FPS: {metrics['fps']:.2f} frames/sec")
    print(f"FLOPs: {metrics['num_flops']:.2f} GFLOPs")
    print(f"Parameters: {metrics['trainable_params']:.2f} M")
    print(f"Mean IoU (mIoU): {metrics['mean_iou']:.2f} %")

    print("\nClass-wise IoU (%):")
    print(f"{'Class':<20} {'IoU':>6}")
    print("-" * 28)
    for cls, val in enumerate(metrics['iou_per_class']):
        print(f"{cls:<20} {val:>6.2f}")

if __name__ == "__main__":
    set_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("************ VALIDATION ON CITYSCAPES ***************")

    transform_cityscapes_dataset = transform_cityscapes()
    target_transform_cityscapes = transform_cityscapes_mask()

    cs_val = CityScapes('./datasets/Cityscapes', 'val', transform_cityscapes_dataset, target_transform_cityscapes)

    batch_size = 4
    learning_rate = 0.00625
    momentum = 0.9
    weight_decay = 1e-4
    num_epochs = 50
    num_classes = 19
    ignore_index = 255
    start_epoch = 1
    loss = CombinedLoss_All(num_classes=num_classes, alpha=0.7, beta=0, gamma=0.3, theta=0, ignore_index=255)

    _, dataloader_cs_val = dataloader(None, cs_val, batch_size, shuffle_train=False, shuffle_val=False)

    model = BiSeNet(num_classes=num_classes, context_path='resnet18').to(device)

    project_name = "3b_GTA5_to_CITY_augmented_geometric_cv07_tv_03"
    for epoch in range(start_epoch, num_epochs + 1):
        run = wandb.init(project=project_name, entity="s328422-politecnico-di-torino", name=f"epoch_{epoch}", reinit=True)
        artifact = wandb.use_artifact(f"{project_name}/model_epoch_{epoch}:latest", type="model")
        artifact_path = artifact.download()
        checkpoint_path = os.path.join(artifact_path, f"model_epoch_{epoch}.pt")

        checkpoint = torch.load(checkpoint_path, map_location=device)
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
