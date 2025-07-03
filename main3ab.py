# main_train.py

import os
import torch
import wandb
import time
import random
import numpy as np
from torch.utils.data import Subset
import random
from datasets.gta5 import GTA5
from datasets.cityscapes import CityScapes
from models.bisenet.build_bisenet import BiSeNet
from utils.utils import CombinedLoss_All, save_metrics_on_file, save_metrics_on_wandb
from datasets.transform_datasets import transform_gta, transform_gta_mask, transform_cityscapes, transform_cityscapes_mask
from data.dataloader import dataloader
from torch.utils.data import ConcatDataset, Subset
from train import train

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


def select_random_fraction_of_dataset(full_dataloader, fraction=1.0, batch_size=4):
    assert 0 < fraction <= 1.0, "La frazione deve essere tra 0 e 1."

    dataset = full_dataloader.dataset
    total_samples = len(dataset)
    num_samples = int(total_samples * fraction)

    # selection of random indices
    indices = np.random.choice(total_samples, num_samples, replace=False)

    # create a subset of the dataset using the selected indices
    subset = Subset(dataset, indices)
    subset_dataloader, _ = dataloader(subset, None, batch_size, True, True, True) # Drop of the last batch

    return subset_dataloader

def to_obtain_id(project=""):
    # project configuration on wandb
    entity = "s325951-politecnico-di-torino-mldl" 
    # entity = "s328422-politecnico-di-torino"

    api = wandb.Api()

    # take project runs
    runs = api.runs(f"{entity}/{project}")

    # Function to extract the epoch number from the run name
    def extract_epoch_number(run):
        try:
            name = run.name
            if name.startswith("epoch_"):
                return int(name.split("_")[1])
        except:
            return float("inf")
        return float("inf")

    # Filter and sort the runs by epoch number
    sorted_runs = sorted(
        [run for run in runs if run.name and run.name.startswith("epoch_")],
        key=extract_epoch_number
    )

    # Create the list of ordered run IDs
    run_ids = [run.id for run in sorted_runs]

    print("Ho caricato", len(run_ids), "run ID.")
    
    return run_ids

if __name__ == "__main__":
    set_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("************ TRAINING BiSeNet ON GTA5 ***************")

    # Constant values
    batch_size = 4
    learning_rate = 0.00625
    momentum = 0.9
    weight_decay = 1e-4
    num_epochs = 50 
    num_classes = 19
    ignore_index = 255
    start_epoch = 1 

    # Transformation
    transform_gta_dataset = transform_gta()
    target_transform_gta = transform_gta_mask()

    print("Loading datasets")

    #select if you want augmentations (3b) or not (3a):
    a_or_b = 'b' # 'a' if 3a , 'b' if 3b

    if a_or_b == 'b':
        # Define the type of augmentation to apply
        """
        for aug_1:
        type_aug == {'color': ['HueSaturationValue','CLAHE', 'GaussNoise', 'RGBShift', 'RandomBrightnessContrast']}

        for aug_2:
        type_aug = None
        """
        type_aug = None

        # to run with local Drive : 
        gta_train_nonaug = GTA5('./datasets/GTA5', transform_gta_dataset, target_transform_gta, augmentation=False, type_aug={}) 
        gta_train_aug = GTA5('./datasets/GTA5', transform_gta_dataset, target_transform_gta, augmentation=True, type_aug=type_aug) 
        # OR to run on kaggle uncomment :
        #gta_train_nonaug = GTA5('/kaggle/input/gta5-dataset/GTA5', transform_gta_dataset, target_transform_gta, augmentation=False, type_aug={}) 
        #gta_train_aug = GTA5('/kaggle/input/gta5-dataset/GTA5', transform_gta_dataset, target_transform_gta, augmentation=True, type_aug=type_aug) 

        # Choose with probability 0.5 the augmented images
        num_augmented = int(0.5 * len(gta_train_aug))
        indices = random.sample(range(len(gta_train_aug)), num_augmented)
        gta_train_aug = Subset(gta_train_aug, indices)

        # Union of the dataset
        gta_train = ConcatDataset([gta_train_nonaug, gta_train_aug]) # To obtain the final dataset = train + augment
    elif a_or_b == 'a':
        # to run with local Drive : 
        gta_train = GTA5('./datasets/GTA5', transform_gta_dataset, target_transform_gta, augmentation=False, type_aug={}) 
        # OR to run on kaggle uncomment :
        #gta_train = GTA5('/kaggle/input/gta5-dataset/GTA5', transform_gta_dataset, target_transform_gta, augmentation=False, type_aug={}) 

    
    # Create dataloader
    full_dataloader_gta_train, _ = dataloader(gta_train, None, batch_size, True, True)
    # Take a subset of the dataloader
    dataloader_gta_train = select_random_fraction_of_dataset(full_dataloader_gta_train, fraction=1.0, batch_size=batch_size)
    
    # Definition of the model
    model = BiSeNet(num_classes=num_classes, context_path='resnet18').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    loss = CombinedLoss_All(num_classes=num_classes, alpha=0.7, beta=0, gamma=0.3, theta=0, ignore_index=255) #CHECK BEFORE RUNNING
    """
    alpha   # CrossEntropy
    beta    # Lovász
    gamma   # Tversky
    theta   # Dice
    """

    max_iter = num_epochs * len(full_dataloader_gta_train)
    iter_curr = 0

    for epoch in range(start_epoch, num_epochs + 1):
        project_name = "3b_GTA5_to_CITY_aug_color_weather_oneof_3_comb_100_percent"
        entity = "s325951-politecnico-di-torino-mldl" 
        # entity="s328422-politecnico-di-torino" 
        run = wandb.init(project=project_name, entity=entity, name=f"epoch_{epoch}", reinit=True)
        wandb.config.update({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "num_classes": num_classes
        })

        if epoch > 1:
            artifact = wandb.use_artifact(f"{project_name}/model_epoch_{epoch-1}:latest", type="model")
            checkpoint_path = artifact.download()
            checkpoint = torch.load(os.path.join(checkpoint_path, f"model_epoch_{epoch-1}.pt"))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"\nEpoch {epoch}")
        start_train = time.time()

        metrics_train, iter_curr = train(epoch, model, dataloader_gta_train, loss, optimizer, iter_curr,
                                         learning_rate, num_classes, max_iter)
        end_train = time.time()
        print(f"Time for training: {(end_train - start_train)/60:.2f} min")

        save_metrics_on_wandb(epoch, metrics_train, metrics_val=None)

        # Save model checkpoint as wandb artifact
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        save_path = f"model_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)

        artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
        artifact.add_file(save_path)
        run.log_artifact(artifact)
        os.remove(save_path)

    wandb.finish()

    print("************ VALIDTING BiSeNet ON GTA5 ***************")

    transform_cityscapes_dataset = transform_cityscapes()
    target_transform_cityscapes = transform_cityscapes_mask()

    # to run with local Drive:
    cs_val = CityScapes('./datasets/Cityscapes', 'val', transform_cityscapes_dataset, target_transform_cityscapes)
    #OR to run on kaggle : 
    #cs_val = CityScapes('/kaggle/input/cityscapes-dataset/Cityscapes', 'val', transform_cityscapes_dataset, target_transform_cityscapes)
    
    _, dataloader_cs_val = dataloader(None, cs_val, batch_size, shuffle_train=False, shuffle_val=False)

    model = BiSeNet(num_classes=num_classes, context_path='resnet18').to(device)

    # take run ids from wandb
    run_ids = to_obtain_id(project_name)

    for epoch in range(start_epoch, num_epochs + 1):
        run = wandb.init(
            project=project_name,
            entity = "s325951-politecnico-di-torino-mldl", 
            # entity="s328422-politecnico-di-torino",
            name=f"epoch_{epoch}",
            id=run_ids[epoch - 1],  
            resume="allow"
        )
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
        
        wandb.finish()
