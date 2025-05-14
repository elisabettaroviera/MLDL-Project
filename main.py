# Import necessary libraries
import os
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
from utils.utils import CombinedLoss_All, CombinedLoss_Tversky, MaskedDiceLoss, poly_lr_scheduler, save_metrics_on_file, save_metrics_on_wandb, CombinedLoss_Lovasz
from validation import validate
from utils.metrics import compute_miou
from torch import nn
import wandb
import gdown
from models.bisenet.build_bisenet import BiSeNet
from torch.utils.data import Subset
from monai.losses import DiceLoss


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
    for cls, val in enumerate(metrics['iou_per_class']):
        print(f"{cls:<20} {val:>6.2f}")



if __name__ == "__main__":
    # ambient variable
    var_model = os.environ['MODEL'] #'DeepLabV2' #BiSeNet # Choose the model to train

    set_seed(23)  # Set a seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################################################################################
    ################################################# STEP 2.a #################################################

    print(f"************STEP 2 : TRAINING {var_model} ON CITYSCAPES***************")
    # Define transformations
    print("Define transformations")
    transform = transform_cityscapes()
    target_transform = transform_cityscapes_mask()

    # Load the datasets (Cityspaces)
    print("Load the datasets")
    cs_train = CityScapes('./datasets/Cityscapes', 'train', transform, target_transform)
    cs_val = CityScapes('./datasets/Cityscapes', 'val', transform, target_transform)


    # Choose the model's parameters
    if var_model == 'DeepLabV2':
        print("MODEL DEEPLABV2")
        # I modify the value of the batch size because it has to match with the one of the model
        batch_size = 3 # 3 or the number the we will use in the model
        learning_rate = 0.0001 # Learning rate for the optimizer - 1e-4
        momentum = 0.9 # Momentum for the optimizer
        weight_decay = 0.0005 # Weight decay for the optimizer
        
    elif var_model == 'BiSeNet':
        print("MODEL BISENET")
        """SCEGLI I PARAMETRIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"""
        batch_size = 4 # chatgpt also suggested to try with 8 (nel paper usano 16)
        learning_rate = 0.00625 # Learning rate for the optimizer - 1e-4
        momentum = 0.9 # Momentum for the optimizer
        weight_decay = 1e-4 # Weight decay for the optimizer

    # TO SPEED UP THE TRAINING WE CAN USE A SUBSET OF THE DATASET
    # Define the dataloaders
    print("Create the dataloaders")
    """# -- Subset TRAIN --
    total_len_train = len(cs_train)
    indices_train = list(range(total_len_train))
    random.shuffle(indices_train)
    subset_len_train = (total_len_train // 4) -1 
    subset_indices_train = indices_train[:subset_len_train]
    train_subset = Subset(cs_train, subset_indices_train)

    # -- Subset VAL --
    total_len_val = len(cs_val)
    indices_val = list(range(total_len_val))
    random.shuffle(indices_val)
    subset_len_val = (total_len_val // 4) +1 
    subset_indices_val = indices_val[:subset_len_val]
    val_subset = Subset(cs_val, subset_indices_val)"""

    # -- DataLoader --
    dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, cs_val, batch_size, True, True)



    # Definition of the parameters for CITYSCAPES
    # Search on the pdf!! 
    print("Define the parameters")
    num_epochs = 50 # Number of epochs

    # The void label is not considered for evaluation (paper 2)
    # Hence the class are 0-18 (19 classes in total) without the void label
    num_classes = 19 # Number of classes in the dataset (Cityscapes)
    ignore_index = 255 # Ignore index for the loss function (void label in Cityscapes)
    iter_curr = 0 # Initialize the iteration counter
    max_iter = num_epochs * len(dataloader_cs_train) # Maximum number of iterations (epochs * batches per epoch)

    if var_model == 'DeepLabV2':
        # Pretrained model path 
        print("Pretrained model path")
        pretrain_model_path = "./pretrained/deeplabv2_cityscapes.pth"
        if not os.path.exists(pretrain_model_path):
            os.makedirs(os.path.dirname(pretrain_model_path), exist_ok=True)
            print("Scarico i pesi pre-addestrati da Google Drive...")
            url = "https://drive.google.com/uc?id=1HZV8-OeMZ9vrWL0LR92D9816NSyOO8Nx"
            gdown.download(url, pretrain_model_path, quiet=False)

        print("Load the model")
        model = get_deeplab_v2(num_classes=num_classes, pretrain=True, pretrain_model_path=pretrain_model_path)
        # number of epoch that we want to start from
        start_epoch = 37 
        

    elif var_model == 'BiSeNet':
        model = BiSeNet(num_classes=num_classes, context_path='resnet18')
        # number of epoch that we want to start from
        start_epoch = 11

    # Load the model on the device    
    model = model.to(device)

    # Definition of the optimizer for the first epoch
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay) # Optimizer (Stochastic Gradient Descent)
    print("Optimizer loaded")
    
    # Defintion of the loss function
    #loss = nn.CrossEntropyLoss(ignore_index=ignore_index) # Loss function (CrossEntropyLoss for segmentation tasks)
    #loss = MaskedDiceLoss(num_classes=num_classes)
    #loss = CombinedLoss_Lovasz(alpha=0.7, beta=0.3, ignore_index=255) # alpha = cross entropy, beta = lovasz
    #loss = CombinedLoss_Tversky(num_classes=num_classes, alpha=0.7, beta=0.3, ignore_index=255) 
    # alpha = 0.7  # CrossEntropy
    # gamma = 0.3  # Tversky
    #loss = CombinedLoss_All(num_classes=num_classes, alpha=0.4, beta=0.1, gamma=0.4, theta=0.1, ignore_index=255)
    loss = CombinedLoss_All(num_classes=num_classes, alpha=0.7, beta=0, gamma=0.3, theta=0, ignore_index=255)

    print(loss.__class__.__name__)
    print("loss loaded")


    
    for epoch in range(start_epoch, num_epochs + 1):
        iter_curr = len(dataloader_cs_train) * (epoch - 1) # Update the iteration counter
        # To save the model we need to initialize wandb 
        # Change the name of the project before the final run of 50 epochs
        wandb.init(project=f"{var_model}_lr_0.00625_cr07_tv03_total_dataset", entity="s328422-politecnico-di-torino", name=f"epoch_{epoch}", reinit=True) # Replace with your wandb entity name
        print("Wandb initialized")

        print(f"Epoch {epoch}")

        print("Load the model")
        # 1. Obtain the pretrained model
        if epoch != 1:
            # Load the model from the previous epoch using wandb artifact
            artifact = wandb.use_artifact(f"s328422-politecnico-di-torino/{var_model}_lr_0.00625_cr07_tv03_total_dataset/model_epoch_{epoch-1}:latest", type="model")
            
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
        metrics_train, iter_curr = train(epoch, model, dataloader_cs_train, loss, optimizer, iter_curr, learning_rate, num_classes, max_iter)
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

        # File for the mIoU for each epoch
            # Training phase
            # Epoch - 1
            # value mIoU
            # value mIoU per class
            # Validation phase
            # Epoch - 1 
            # value mIoU
            # value mIoU per class

        # File for the Loss for each epoch
            # Training phase
            # Epoch - 1
            # value loss
            # Validation phase
            # Epoch - 1
            # value loss


        # File for all the final metrics (only the 50th epoch)
            # Training phase
            # Epoch - 50
            # value mIoU
            # value mIoU per class
            # value loss
            # value latency
            # value FPS
            # value FLOPs
            # value parameters

            # Validation phase
            # Epoch - 50
            # value mIoU
            # value mIoU per class
            # value loss
            # value latency
            # value FPS
            # value FLOPs
            # value parameters
        save_metrics_on_wandb(epoch, metrics_train, metrics_val)
        save_metrics_on_file(epoch, metrics_train, metrics_val)
        wandb.finish()

        """
        ROBE DI AURO 
        
        from lovasz_losses import lovasz_softmax  # assicurati di avere questo file o installato da GitHub

        class CombinedLoss(nn.Module):
            def __init__(self, alpha=0.5, beta=0.5, ignore_index=255):
                super(CombinedLoss, self).__init__()
                self.alpha = alpha
                self.beta = beta
                self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
                self.ignore_index = ignore_index

            def forward(self, outputs, targets):
                ce = self.ce_loss(outputs, targets)
                probs = torch.softmax(outputs, dim=1)
                lovasz = lovasz_softmax(probs, targets, ignore=self.ignore_index)
                return self.alpha * ce + self.beta * lovasz
                
        criterion = CombinedLoss(alpha=0.5, beta=0.5, ignore_index=255)"""
