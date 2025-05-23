import os
import torch
from torch import nn
import wandb
import gdown
from torchvision.datasets import ImageFolder
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from models.bisenet.build_bisenet import BiSeNet
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
from utils.utils import *
from validation import validate
from utils.metrics import compute_miou
from torch.utils.data import Subset

# This function sets the seed for various libraries to ensure that the results are reproducible.
def set_seed(seed):
    torch.manual_seed(seed) # Set the seed for CPU
    torch.cuda.manual_seed_all(seed) # Set the seed for all GPUs
    np.random.seed(seed) # Set the seed for NumPy
    random.seed(seed) # Set the seed for random
    torch.backends.cudnn.benchmark = True # Enable auto-tuning for max performance
    torch.backends.cudnn.deterministic = False # Allow non-deterministic algorithms for better performance

def select_random_fraction_of_dataset(full_dataloader, fraction=1.0, batch_size=4):
    assert 0 < fraction <= 1.0, "La frazione deve essere tra 0 e 1."

    dataset = full_dataloader.dataset
    total_samples = len(dataset)
    num_samples = int(total_samples * fraction)

    # Seleziona indici casuali senza ripetizioni
    indices = np.random.choice(total_samples, num_samples, replace=False)

    # Crea un subset e un nuovo dataloader
    subset = Subset(dataset, indices)
    subset_dataloader, _ = dataloader(subset, None, batch_size, True, True, True) # Drop of the last batch

    return subset_dataloader

if __name__ == "__main__":
    # Ambient variable
    var_model = os.environ['MODEL'] #'DeepLabV2' OR 'BiSeNet' # CHOOSE the model to train

    set_seed(23)  # Set a seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################################################################################
    ################################################## STEP 2 ##################################################
    ############################################################################################################

    print(f"************ STEP 2 : TRAINING {var_model} ON CITYSCAPES ***************")
    
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
        batch_size = 3 # Bach size
        learning_rate = 0.0003 # Learning rate for the optimizer - CHANGE HERE!
        momentum = 0.9 # Momentum for the optimizer
        weight_decay = 0.0005 # Weight decay for the optimizer
        
    elif var_model == 'BiSeNet':
        print("MODEL BISENET")
        batch_size = 4 # Bach size
        learning_rate = 0.0025 # Learning rate for the optimizer - 1e-4
        momentum = 0.9 # Momentum for the optimizer
        weight_decay = 1e-4 # Weight decay for the optimizer

    # Define the dataloaders
    print("Create the dataloaders")
    dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, cs_val, batch_size, True, True)

    # Take a subset of the dataloader
    # dataloader_cs_train = select_random_fraction_of_dataset(dataloader_cs_train, fraction=0.25, batch_size=batch_size)

    # Definition of the parameters for CITYSCAPES 
    print("Define the parameters")
    num_epochs = 50 # Number of epochs
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
            print("Load the pre-trained weight from Google Drive")
            url = "https://drive.google.com/uc?id=1HZV8-OeMZ9vrWL0LR92D9816NSyOO8Nx"
            gdown.download(url, pretrain_model_path, quiet=False)

        print("Load the model")
        model = get_deeplab_v2(num_classes=num_classes, pretrain=True, pretrain_model_path=pretrain_model_path)
       
        start_epoch = 16 # CHANGE HERE THE STARTING EPOCH
        

    elif var_model == 'BiSeNet':
        model = BiSeNet(num_classes=num_classes, context_path='resnet18')
        start_epoch = 45 # CHANGE HERE THE STARTING EPOCH

    # Load the model on the device    
    model = model.to(device)

    # Definition of the optimizer for the first epoch
    print("Definition of the optimizer")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay) # CHANGE HERE THE OPTIMIZER
    
    
    # Defintion of the loss function CombinedLoss_All
    print("Definition of the loss") 
    loss = CombinedLoss_All(num_classes=num_classes, alpha=0.5, beta=0, gamma=0, theta=0, delta=0.5, focal_gamma=2, ignore_index=255) # CHANGE HERE THE LOSS
    # alpha   - CrossEntropy
    # beta    - Lovász
    # gamma   - Tversky
    # theta   - Dice
    # delta   - Focal
    
    # Iteration loop on EPOCHS
    for epoch in range(start_epoch, num_epochs + 1):
        iter_curr = len(dataloader_cs_train) * (epoch - 1) # Update the iteration counter

        # To save the model we need to initialize wandb 
        # entity="s328422-politecnico-di-torino" # Old entity Betta
        entity = "s325951-politecnico-di-torino-mldl" # New entity Lucia
        # _ce06_l0.2_fo0.2_no_warnup_lr_0.0001
        # _ce05_f05_warnup_lr_0.0003
        project_name = f"{var_model}_ce05_f05_warnup_lr_0.0003"
        wandb.init(project=project_name, entity=entity, name=f"epoch_{epoch}", reinit=True) 
        print("Wandb initialized")

        print(f"Epoch {epoch}")

        print("Load the model")
        # 1. Obtain the pretrained model
        if epoch != 1:
            # Load the model from the previous epoch using wandb artifact
            artifact = wandb.use_artifact(f"{entity}/{project_name}/model_epoch_{epoch-1}:latest", type="model")
            
            # Get the local path where the artifact is saved
            artifact_dir = artifact.download()

            # Load the model checkpoint from the artifact
            checkpoint_path = os.path.join(artifact_dir, f"model_epoch_{epoch-1}.pt")
            checkpoint = torch.load(checkpoint_path)  

            # Load the model and the ottimizator state
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
         
    
        # 2. Training step
        print("Training step")

        start_train = time.time()
        metrics_train, iter_curr = train(epoch, model, dataloader_cs_train, loss, optimizer, iter_curr, learning_rate, num_classes, max_iter)
        end_train = time.time()

        print(f"Time taken for training step: {(end_train - start_train)/60:.2f} minutes")

        print("Training step done")

        # 3. Validation step
        print("Validation step")

        start_val = time.time()
        metrics_val = validate(epoch, model, dataloader_cs_val, loss, num_classes) 
        end_val = time.time()

        print(f"Time taken for validation step: {(end_val - start_val)/60:.2f} minutes")

        print("Validation step done")


        # Compute the total time taken for the epoch (training + validation)
        tot_time = end_val - start_train
        print(f"Total time taken for epoch {epoch}: {(tot_time)/60:.2f} minutes")

        save_metrics_on_wandb(epoch, metrics_train, metrics_val)

        wandb.finish()
   