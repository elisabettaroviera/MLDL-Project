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
from PIDNET import PIDNet
from torch.utils.data import ConcatDataset, Subset

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



    # 1. Definisci il modello: pidnet s
    model = PIDNet(m=2, n=3, num_classes=19, augment=False) #pretrained è false anceh perche non abbiamo i pesi pre-addestrati
    model = model.to(device)

    """inutili<. mi amnca boundary mask
    # 3. I pesi lambda e soglia (come da paper)
    loss_weights = {
        'lambda0': 0.4,
        'lambda1': 20,
        'lambda2': 1,
        'lambda3': 1,
        'threshold': 0.8,
    }

    # 4. Funzione loss definita prima
    def pidnet_loss(outputs, target_semantic, target_boundary, weight_boundary, weights):
        x_extra_p, x_main, x_extra_d = outputs
        lambda0 = weights['lambda0']
        lambda1 = weights['lambda1']
        lambda2 = weights['lambda2']
        lambda3 = weights['lambda3']
        threshold = weights['threshold']

        # l0: semantic loss sul ramo P
        l0 = torch.nn.functional.cross_entropy(x_extra_p, target_semantic)

        # l1: weighted binary cross entropy per boundary
        l1 = torch.nn.functional.binary_cross_entropy_with_logits(x_extra_d, target_boundary.float(), weight=weight_boundary)

        # l2: semantic loss sul ramo principale
        l2 = torch.nn.functional.cross_entropy(x_main, target_semantic)

        # l3: boundary-aware CE loss, calcolata solo sui pixel di confine
        mask = (torch.sigmoid(x_extra_d) > threshold).float()
        ce_per_pixel = torch.nn.functional.cross_entropy(x_main, target_semantic, reduction='none')
        l3 = (ce_per_pixel * mask).sum() / (mask.sum() + 1e-6)  # evita divisione per zero

        loss = lambda0 * l0 + lambda1 * l1 + lambda2 * l2 + lambda3 * l3
        return loss
    """

    # Define the dataloaders
    batch_size = 4
    print("Create the dataloaders")
    dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, cs_val, batch_size, True, True)
    # Select a random fraction of the training dataset (25% of the original dataset)
    #dataloader_cs_train = select_random_fraction_of_dataset(dataloader_cs_train, fraction=0.25, batch_size=batch_size)

    # Definition of the parameters for CITYSCAPES 
        # Constant value
    learning_rate = 0.00625
    momentum = 0.9
    weight_decay = 1e-4
    num_epochs = 50#changed bc doing smaller runs
    num_classes = 19
    ignore_index = 255
    start_epoch = 4 #CHECK BEFORE RUNNING
    iter_curr = 0 # Initialize the iteration counter
    max_iter = num_epochs * len(dataloader_cs_train) # Maximum number of iterations (epochs * batches per epoch)



    # Definition of the optimizer for the first epoch
    print("Definition of the optimizer")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay) # CHANGE HERE THE OPTIMIZER
    
    
    # Defintion of the loss function: usano cross entropy nel apper
    print("Definition of the loss")
    loss = CombinedLoss_All(num_classes=num_classes, alpha=1.0, beta=0, gamma=0, theta=0, ignore_index=255) # CHANGE HERE THE LOSS
    # alpha   - CrossEntropy
    # beta    - Lovász
    # gamma   - Tversky
    # theta   - Dice
    
    # Iteration loop on EPOCHS
    for epoch in range(start_epoch, num_epochs + 1):
        iter_curr = len(dataloader_cs_train) * (epoch - 1) # Update the iteration counter

        # To save the model we need to initialize wandb 
        # entity="s328422-politecnico-di-torino" # Old entity Betta
        entity = "s325951-politecnico-di-torino-mldl" # New entity Lucia
        project_name = f"5_PIDNET_ce_100_percent"
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
   