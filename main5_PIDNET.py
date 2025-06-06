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
from train import train_pidnet
from utils.utils import CombinedLoss_All, poly_lr_scheduler, save_metrics_on_wandb
from validation import validate_pidnet
from utils.metrics import compute_miou
from PIDNET import PIDNet, get_seg_model
from torch.utils.data import ConcatDataset, Subset
import torch.nn.functional as F

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


# This function sets the seed for various libraries to ensure that the results are reproducible.
def set_seed(seed):
    torch.manual_seed(seed) # Set the seed for CPU
    torch.cuda.manual_seed_all(seed) # Set the seed for all GPUs
    np.random.seed(seed) # Set the seed for NumPy
    random.seed(seed) # Set the seed for random
    torch.backends.cudnn.benchmark = True # Enable auto-tuning for max performance
    torch.backends.cudnn.deterministic = False # Allow non-deterministic algorithms for better performance

if __name__ == "__main__":


    set_seed(23)  # Set a seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################################################################################
    ################################################## STEP 2 ##################################################
    ############################################################################################################

    print(f"************ STEP 2 : TRAINING PIDNET ON CITYSCAPES ***************")
    
    # Define transformations
    print("Define transformations")
    transform = transform_cityscapes()
    target_transform = transform_cityscapes_mask()

    # Load the datasets (Cityspaces)
    print("Load the datasets")
    cs_train = CityScapes('/kaggle/input/cityscapes-dataset/Cityscapes', 'train', transform, target_transform)
    cs_val = CityScapes('/kaggle/input/cityscapes-dataset/Cityscapes', 'val', transform, target_transform)

    class CFG:
        pass

    cfg = CFG()
    cfg.MODEL = type('', (), {})()
    cfg.DATASET = type('', (), {})()

    cfg.MODEL.NAME = 'pidnet_s'
    cfg.MODEL.PRETRAINED = '/kaggle/input/pidnet-s-imagenet-pretrained-tar/PIDNet_S_ImageNet.pth.tar'
    cfg.DATASET.NUM_CLASSES = 19

    model = get_seg_model(cfg, imgnet_pretrained=True)

    model = model.to(device)


    # Define the dataloaders
    batch_size = 4
    print("Create the dataloaders")
    dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, cs_val, batch_size, True, True)
    # Select a random fraction of the training dataset (25% of the original dataset)
    #dataloader_cs_train = select_random_fraction_of_dataset(dataloader_cs_train, fraction=0.5, batch_size=batch_size)

    # Definition of the parameters for CITYSCAPES 
        # Constant value
    learning_rate = 0.00625
    #learning_rate = 0.00025
    #learning_rate = 0.0005 # Changed to 0.00025 for PIDNet
    momentum = 0.9
    weight_decay = 5e-4 #sul paper usa questo batch size 12
    num_epochs = 50#changed bc doing smaller runs
    num_classes = 19
    ignore_index = 255
    start_epoch = 1
     #CHECK BEFORE RUNNING
    iter_curr = 0 # Initialize the iteration counter
    max_iter = num_epochs * len(dataloader_cs_train) # Maximum number of iterations (epochs * batches per epoch)



    # Definition of the optimizer for the first epoch
    print("Definition of the optimizer")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay) # CHANGE HERE THE OPTIMIZER
    
    
    # Defintion of the loss function: usano cross entropy nel apper
    print("Definition of the loss")
    loss = CombinedLoss_All(num_classes=num_classes, alpha=1.0, beta=0, gamma=0, theta=0, ignore_index=255) # CHANGE HERE THE LOSS
    # in realta in train e validation la ridefinisco 
    # alpha   - CrossEntropy
    # beta    - Lovász
    # gamma   - Tversky
    # theta   - Dice
    
    # Iteration loop on EPOCHS
    for epoch in range(start_epoch, num_epochs + 1):
        iter_curr = len(dataloader_cs_train) * (epoch - 1) # Update the iteration counter

        # To save the model we need to initialize wandb 
        # entity="s328422-politecnico-di-torino" # Old entity Betta
        entity = "s281401-politecnico-di-torino" # New entity  Auro
        project_name = f"5_PIDNET_1ce_0.00625_totloss_100_percent"
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
        metrics_train, iter_curr = train_pidnet(epoch, model, dataloader_cs_train, loss, optimizer, iter_curr, learning_rate, num_classes, max_iter)
        end_train = time.time()

        print(f"Time taken for training step: {(end_train - start_train)/60:.2f} minutes")

        print("Training step done")

        # 3. Validation step
        print("Validation step")

        start_val = time.time()
        metrics_val = validate_pidnet(epoch, model, dataloader_cs_val, loss, num_classes) 
        end_val = time.time()

        print(f"Time taken for validation step: {(end_val - start_val)/60:.2f} minutes")

        print("Validation step done")


        # Compute the total time taken for the epoch (training + validation)
        tot_time = end_val - start_train
        print(f"Total time taken for epoch {epoch}: {(tot_time)/60:.2f} minutes")

        save_metrics_on_wandb(epoch, metrics_train, metrics_val)

        wandb.finish()
   