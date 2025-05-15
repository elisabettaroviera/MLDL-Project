# Import necessary libraries
import os
from datasets.gta5 import GTA5
from models.deeplabv2.deeplabv2 import get_deeplab_v2
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
from utils.utils import poly_lr_scheduler, save_metrics_on_file, save_metrics_on_wandb
from validation import validate
from utils.metrics import compute_miou
from torch import nn
import wandb
import gdown
import albumentations as A
from models.bisenet.build_bisenet import BiSeNet
from torch.utils.data import ConcatDataset

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
    A.set_seed(seed)

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
    var_model = os.environ['MODEL'] 
    set_seed(23)  # Set a seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################################################################################
    ################################################# STEP 3.ab #################################################

    print(f"************STEP 3 : TRAINING BISENET ON GTA5***************")
    # Define transformations
    print("Define transformations")
    transform_gta_dataset = transform_gta()
    transform_cityscapes_dataset = transform_cityscapes()
    target_transform_cityscapes = transform_cityscapes_mask()
    target_transform_gta = transform_gta_mask()

    # Load the datasets (Cityspaces)
    print("Load the Cityscapes dataset")
    cs_val = CityScapes('./datasets/Cityscapes', 'val', transform_cityscapes_dataset, target_transform_cityscapes)  
    
    # Load the datasets (GTA5)
    print("Load the GTA5 dataset")
    gta_train = GTA5('./datasets/GTA5', transform_gta_dataset, target_transform_gta, augemntation = False)
    gta_augmentation = GTA5('./datasets/GTA5', transform_gta_dataset, target_transform_gta, augmentation = True)

    # Union of the dataset
    gta_combined = ConcatDataset([gta_train, gta_augmentation]) # To obtain the final dataset = train + augment

    print("MODEL BISENET")
    batch_size = 4 # chatgpt also suggested to try with 8 (nel paper usano 16)
    learning_rate = 0.00625 # Learning rate for the optimizer - 1e-4
    momentum = 0.9 # Momentum for the optimizer
    weight_decay = 1e-4 # Weight decay for the optimizer

    # Define the dataloaders
    print("Create the dataloaders")
    dataloader_gta_train, dataloader_cs_val = dataloader(gta_train, cs_val, batch_size, True, True)

    # Definition of the parameters for GTA5
    # Search on the pdf!! 
    print("Define the parameters")
    num_epochs = 50 # Number of epochs
    # The void label is not considered for evaluation (paper 2)
    # Hence the class are 0-18 (19 classes in total) without the void label
    num_classes = 19 # Number of classes in the dataset (Cityscapes)
    ignore_index = 255 # Ignore index for the loss function (void label in Cityscapes)
    iter_curr = 0 # Initialize the iteration counter
    max_iter = num_epochs * len(dataloader_gta_train) # Maximum number of iterations (epochs * batches per epoch)

    model = BiSeNet(num_classes=num_classes, context_path='resnet18')
    # number of epoch that we want to start from
    start_epoch = 1

    # Load the model on the device    
    model = model.to(device)

    # Definition of the optimizer for the first epoch
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay) # Optimizer (Stochastic Gradient Descent)
    print("Optimizer loaded")
    
    # Defintion of the loss function
    ##########################################################################################
    # CHANGE THE LOSS FUNCTION WRT THE RESULTS OF 2.B
    #########################################################################################
    loss = nn.CrossEntropyLoss(ignore_index=ignore_index) # Loss function (CrossEntropyLoss for segmentation tasks)
    print("loss loaded")


    for epoch in range(start_epoch, num_epochs + 1):
        iter_curr = len(dataloader_gta_train) * (epoch - 1) # Update the iteration counter
        # To save the model we need to initialize wandb 
        # Change the name of the project before the final run of 50 epochs
        project_name = "3a_GTA5_to_CITY_" # Change here!
        wandb.init(project=f"{project_name}", entity="s328422-politecnico-di-torino", name=f"epoch_{epoch}", reinit=True) # Replace with your wandb entity name
        print("Wandb initialized")

        print(f"Epoch {epoch}")

        print("Load the model")
        # 1. Obtain the pretrained model
        if epoch != 1:
            # Load the model from the previous epoch using wandb artifact
            artifact = wandb.use_artifact(f"s328422-politecnico-di-torino/{project_name}/model_epoch_{epoch-1}:latest", type="model")
            
            # Get the local path where the artifact is saved
            artifact_dir = artifact.download()

            # Load the model checkpoint from the artifact
            checkpoint_path = os.path.join(artifact_dir, f"model_epoch_{epoch-1}.pt")
            checkpoint = torch.load(checkpoint_path)  # Carica il checkpoint

            # Carica il modello e lo stato dell'ottimizzatore
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    
        # 2. Training step
        print("Training step")
        start_train = time.time()
        metrics_train, iter_curr = train(epoch, model, dataloader_gta_train, loss, optimizer, iter_curr, learning_rate, num_classes, max_iter)
        end_train = time.time()
        print(f"Time taken for training step: {(end_train - start_train)/60:.2f} minutes")
        print("Training step done")

        # PRINT all the metrics!
        print_metrics("Training", metrics_train)

        # 3. Validation step
        print("Validation step")
        start_val = time.time()
        metrics_val = validate(epoch, model, dataloader_cs_val, loss, num_classes) # Compute the accuracy on the validation set
        end_val = time.time()
        print(f"Time taken for validation step: {(end_val - start_val)/60:.2f} minutes")
        print("Validation step done")


        # PRINT all the metrics!
        print_metrics("Validation", metrics_val)
        # Compute the total time taken for the epoch
        # (training + validation)
        tot_time = end_val - start_train
        print(f"Total time taken for epoch {epoch}: {(tot_time)/60:.2f} minutes")
        save_metrics_on_wandb(epoch, metrics_train, metrics_val)
        save_metrics_on_file(epoch, metrics_train, metrics_val)
        wandb.finish()