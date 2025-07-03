import argparse
import os
from datasets.gta5 import GTA5
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
from models.pidnet.PIDNET import PIDNet, get_seg_model
#from models.pidnet.DROPOUT_PIDNET  import get_seg_model #<- SE APPLICO DROPOUT PIDNET
from torch.utils.data import ConcatDataset, Subset
import torch.nn.functional as F

# This function sets the seed for various libraries to ensure that the results are reproducible.
def set_seed(seed):
    torch.manual_seed(seed) # Set the seed for CPU
    torch.cuda.manual_seed_all(seed) # Set the seed for all GPUs
    np.random.seed(seed) # Set the seed for NumPy
    random.seed(seed) # Set the seed for random
    torch.backends.cudnn.benchmark = True # Enable auto-tuning for max performance
    torch.backends.cudnn.deterministic = False # Allow non-deterministic algorithms for better performance

#mai1: city to city
def main1():
    print("Executing main1: PIDNet City to City")
    name_main = 'city_to_city'
    
    # Tranformations for Cityscapes Train and Val
    transform_cityscapes_dataset = transform_cityscapes()
    target_transform_cityscapes = transform_cityscapes_mask()

    # Load the Cityscapes dataset + make dataloader
    print("Load the datasets and create the datalaoders")
    cs_train = CityScapes('/kaggle/input/cityscapes-dataset/Cityscapes', 'train', transform_cityscapes_dataset, target_transform_cityscapes)  
    cs_val = CityScapes('/kaggle/input/cityscapes-dataset/Cityscapes', 'val', transform_cityscapes_dataset, target_transform_cityscapes)  
    dataloader_train, dataloader_val = dataloader(cs_train, cs_val, batch_size, True, True)

#main2: gta5 to city no augmentations
def main2():
    print("Executing main2: PIDNet GTA5 (no augmentations) to City")
    name_main = 'gta_to_city_no_aug'
    
    # Tranformations for Cityscapes and GTA5 
    transform_cityscapes_dataset = transform_cityscapes()
    target_transform_cityscapes = transform_cityscapes_mask()
    transform_gta_dataset = transform_gta()
    target_transform_gta = transform_gta_mask()

    # Load the Cityscapes dataset + make dataloader
    print("Load the datasets and create the datalaoders")
   
    cs_val = CityScapes('/kaggle/input/cityscapes-dataset/Cityscapes', 'val', transform_cityscapes_dataset, target_transform_cityscapes)  
    gta_train_nonaug = GTA5('/kaggle/input/gta5-dataset/GTA5', transform_gta_dataset, target_transform_gta, augmentation=False, type_aug={})  # No type_aug 

    dataloader_train, _ = dataloader(gta_train_nonaug, None, batch_size, True, True, False, 4)
    _, dataloader_val = dataloader(None, cs_val, batch_size, True, True, False, 4)

# main3: gta5 to city with augmentation aug_1
def main3():
    print("Executing main3: PIDNet GTA5 (with augmentation aug_1) to City")
    name_main = 'gta_to_city_aug_1'
    
    # Tranformations for Cityscapes and GTA5 
    transform_cityscapes_dataset = transform_cityscapes()
    target_transform_cityscapes = transform_cityscapes_mask()
    transform_gta_dataset = transform_gta()
    target_transform_gta = transform_gta_mask()

    # Load datasets + make dataloader
    print("Load the datasets and create the datalaoders")
   
    cs_val = CityScapes('/kaggle/input/cityscapes-dataset/Cityscapes', 'val', transform_cityscapes_dataset, target_transform_cityscapes)  
    gta_train_nonaug = GTA5('/kaggle/input/gta5-dataset/GTA5', transform_gta_dataset, target_transform_gta, augmentation=False, type_aug={})  # No type_aug 
    #if aug_1 -> augmentation_transform
    type_aug = {'color': ['HueSaturationValue','CLAHE', 'GaussNoise', 'RGBShift', 'RandomBrightnessContrast']} 
    #if aug_2 -> augmentation_transform_oneof_col3_wea
    #type_aug = None
    gta_train_aug = GTA5('/kaggle/input/gta5-dataset/GTA5', transform_gta_dataset, target_transform_gta, augmentation=True, type_aug=type_aug)  # Change the augm that you want

    # Choose with probability 0.5 the augmented images
    num_augmented = int(0.5 * len(gta_train_aug))
    indices = random.sample(range(len(gta_train_aug)), num_augmented)
    gta_train_aug = Subset(gta_train_aug, indices)

    # Union of the dataset
    gta_train = ConcatDataset([gta_train_nonaug, gta_train_aug])  # To obtain the final dataset = train + augment

    dataloader_train, _ = dataloader(gta_train, None, batch_size, True, True, False, 4)
    _, dataloader_val = dataloader(None, cs_val, batch_size, True, True, False, 4)


# main4: gta5 to city with augmentation aug_2
def main4():
    print("Executing main3: PIDNet GTA5 (with augmentation aug_2) to City")
    name_main = 'gta_to_city_aug_2'
    
    # Tranformations for Cityscapes and GTA5 
    transform_cityscapes_dataset = transform_cityscapes()
    target_transform_cityscapes = transform_cityscapes_mask()
    transform_gta_dataset = transform_gta()
    target_transform_gta = transform_gta_mask()

    # Load the Cityscapes dataset + make dataloader
    print("Load the datasets and create the datalaoders")
   
    cs_val = CityScapes('/kaggle/input/cityscapes-dataset/Cityscapes', 'val', transform_cityscapes_dataset, target_transform_cityscapes)  
    gta_train_nonaug = GTA5('/kaggle/input/gta5-dataset/GTA5', transform_gta_dataset, target_transform_gta, augmentation=False, type_aug={})  # No type_aug 

    #if aug_2 -> augmentation_transform_oneof_col3_wea
    type_aug = None
    gta_train_aug = GTA5('/kaggle/input/gta5-dataset/GTA5', transform_gta_dataset, target_transform_gta, augmentation=True, type_aug=type_aug)  # Change the augm that you want

    # Choose with probability 0.5 the augmented images
    num_augmented = int(0.5 * len(gta_train_aug))
    indices = random.sample(range(len(gta_train_aug)), num_augmented)
    gta_train_aug = Subset(gta_train_aug, indices)

    # Union of the dataset
    gta_train = ConcatDataset([gta_train_nonaug, gta_train_aug])  # To obtain the final dataset = train + augment

    dataloader_train, _ = dataloader(gta_train, None, batch_size, True, True, False, 4)
    _, dataloader_val = dataloader(None, cs_val, batch_size, True, True, False, 4)

# RUN WITH: !python python main5_PIDNET.py --mode main1 or 2 or 3 or 4
if __name__ == "__main__":
    
    print(f"************ MAIN5: PIDNET ***************")
    
    # Define transformations
    print("Define transformations")

    set_seed(23)  # Set a seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Definition of the hyperparameters 
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4 
    num_epochs = 50 #changed bc doing smaller runs
    num_classes = 19
    ignore_index = 255
    start_epoch = 1
    batch_size = 4

    # Defintion of the loss function: usano cross entropy nel apper
    print("Definition of the loss")
    loss = CombinedLoss_All(num_classes=num_classes, alpha=1.0, beta=0, gamma=0, theta=0, ignore_index=255) 
    # alpha   - CrossEntropy
    # beta    - Lovász
    # gamma   - Tversky
    # theta   - Dice

    # Parse arguments
    parser = argparse.ArgumentParser(description="Choose main")
    parser.add_argument("--mode", type=str, choices=["main1", "main2", "main3", "main4"], required=True)
    args = parser.parse_args()

    if args.mode == "main1":
        main1()
    elif args.mode == "main2":
        main2()
    elif args.mode == "main3":
        main3()

    elif args.mode == "main4":
        main4()

    # Define the model (PIDNet M)
    class CFG:
        pass
    cfg = CFG()
    cfg.MODEL = type('', (), {})()
    cfg.DATASET = type('', (), {})()

    cfg.MODEL.NAME = 'pidnet_m'
    #PRETRAINED WEIGHTS ON IMAGENET
    cfg.MODEL.PRETRAINED = '/kaggle/input/pidnet-m/PIDNet_M_ImageNet.pth.tar'
    cfg.DATASET.NUM_CLASSES = 19
    # Serve cosi chiamo pesi preaddestrati su ImageNet
    model = get_seg_model(cfg, imgnet_pretrained=True)
    model = model.to(device)


    #CHECK BEFORE RUNNING
    iter_curr = 0 # Initialize the iteration counter
    max_iter = num_epochs * len(dataloader_train) # Maximum number of iterations (epochs * batches per epoch)

    # Definition of the optimizer
    print("Definition of the optimizer")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay) # CHANGE HERE THE OPTIMIZER
    
    ############################################################################################################
    
    # Iteration loop on EPOCHS
    for epoch in range(start_epoch, num_epochs + 1):
        iter_curr = len(dataloader_train) * (epoch - 1) # Update the iteration counter

        # To save the model we need to initialize wandb 
        # entity="s328422-politecnico-di-torino" # entity Betta
        entity = "s281401-politecnico-di-torino" # entity  Auro
        project_name = f"PIDNet_M_{name_main}" 

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
        metrics_train, iter_curr = train_pidnet(epoch, model, dataloader_train, loss, optimizer, iter_curr, learning_rate, num_classes, max_iter)
        end_train = time.time()

        print(f"Time taken for training step: {(end_train - start_train)/60:.2f} minutes")

        print("Training step done")

        # 3. Validation step
        print("Validation step")

        start_val = time.time()
        metrics_val = validate_pidnet(epoch, model, dataloader_val, loss, num_classes) 
        end_val = time.time()

        print(f"Time taken for validation step: {(end_val - start_val)/60:.2f} minutes")

        print("Validation step done")


        # Compute the total time taken for the epoch (training + validation)
        tot_time = end_val - start_train
        print(f"Total time taken for epoch {epoch}: {(tot_time)/60:.2f} minutes")

        save_metrics_on_wandb(epoch, metrics_train, metrics_val)

        wandb.finish()
