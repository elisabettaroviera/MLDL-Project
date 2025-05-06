# Import necessary libraries
import os
from models.deeplabv2.deeplabv2 import lr_policy
import torch
from torchvision.datasets import ImageFolder
from datasets.transform_datasets import *
from data.dataloader import dataloader
import numpy as np
import time
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torchvision.transforms.functional as TF
from datasets.cityscapes import CityScapes
import random
from train import train
from validation import validate
from utils.metrics import compute_miou


# Function to set the seed for reproducibility
# This function sets the seed for various libraries to ensure that the results are reproducible.
def set_seed(seed):
    torch.manual_seed(seed) # Set the seed for CPU
    torch.cuda.manual_seed_all(seed) # Set the seed for all GPUs
    np.random.seed(seed) # Set the seed for NumPy
    random.seed(seed) # Set the seed for random
    torch.backends.cudnn.benchmark = True # Enable auto-tuning for max performance
    torch.backends.cudnn.deterministic = False # Allow non-deterministic algorithms for better performance

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
    for cls, val in metrics['iou_per_class'].items():
        print(f"{cls:<20} {val:>6.2f}")


if __name__ == "__main__":

    set_seed(23)  # Set a seed for reproducibility

    ############################################################################################################
    ################################################# STEP 2.a #################################################

    # Define transformations
    transform = transform_cityscapes()
    target_transform = transform_cityscapes_mask()

    # Load the datasets (Cityspaces)
    cs_train = CityScapes('./datasets/Cityscapes', 'train', transform, target_transform)
    cs_val = CityScapes('./datasets/Cityscapes', 'val', transform, target_transform)

    # DataLoader
    # also saving filenames for the images, when doing train i should not need them
    # each batch is a nuple: images, masks, filenames 
    # I modify the value of the batch size because it has to match with the one of the model
    batch_size = 2 # 3 or the number the we will use in the model
    dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, cs_val, batch_size, True, True)

    # Definition of the parameters
    # Search on the pdf!! REVIEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    num_epochs = 50 # Number of epochs
    num_classes = 19 # Number of classes in the dataset (Cityscapes)
    
    learning_rate = 0.001 # Learning rate for the optimizer
    momentum = 0.9 # Momentum for the optimizer
    weight_decay = 0.0005 # Weight decay for the optimizer
    batch_size = 2 # Batch size for the DataLoader

    # FOR LOOP ON THE EPOCHS
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}")

        print("Load the model")
        # 1. Obtain the pretrained model
        model = None # Put the model with wandb

        loss = torch.nn.CrossEntropyLoss() # Loss function (CrossEntropyLoss for segmentation tasks)
        
        # MODIFY THE LEARNING RATE: we have to update the lr at each batch, not only at the beginning of the epoch
        lr = lr_policy(optimizer, init_lr=learning_rate, iter=current_iter, max_iter=total_iters)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay) # Optimizer (Stochastic Gradient Descent)

        # 2. Training step
        print("Training step")
        metrics_train, new_model = train(epoch, model, dataloader_cs_train, loss, optimizer)
        print("Training step done")

        # PRINT all the metrics!
        print_metrics("Training", metrics_train)

        # 3. Validation step
        print("Validation step")
        metrics_val = validate(new_model, dataloader_cs_val, loss) # Compute the accuracy on the validation set
        print("Validation step done")

        # PRINT all the metrics!
        print_metrics("Validation", metrics_val)