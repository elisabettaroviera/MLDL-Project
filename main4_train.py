import os
import torch
import wandb
import time
import random
import numpy as np
from datasets.gta5 import GTA5
from datasets.cityscapes import CityScapes
from models.bisenet.build_bisenet import BiSeNet
from utils.utils import CombinedLoss_All, save_metrics_on_wandb
from datasets.transform_datasets import transform_gta, transform_gta_mask, transform_cityscapes, transform_cityscapes_mask
from data.dataloader import dataloader
from torch.utils.data import ConcatDataset, Subset
from train import train_with_adversary
from models.discriminator.build_discriminator import FCDiscriminator

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

def generate_discriminators(num, num_classes, device='CPU'):
    """
    Generates a list of discriminators based on the number of classes.
    Each discriminator is an instance of FCDiscriminator.
    
    Args:
        num (int): Number of discriminators to generate.
        num_classes (int): Number of classes for the discriminators.
        
    Returns:
        list: A list of discriminator instances.
    """
    return [FCDiscriminator(num_classes=num_classes).to_device(device) for _ in range(num)], [torch.optim.Adam(FCDiscriminator(num_classes=num_classes).parameters(), lr=0.0001, betas=(0.9, 0.99)).to_device(device) for _ in range(num)]


if __name__ == "__main__":
    set_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("************ TRAINING BiSeNet ON GTA5 ***************")

    # Constant value
    batch_size = 4
    learning_rate = 0.00625
    momentum = 0.9
    weight_decay = 1e-4
    num_epochs = 50 
    num_classes = 19
    ignore_index = 255
    start_epoch = 1 #CHECK BEFORE RUNNING
    compute_mIoU = False # If True, compute mIoU at the end of each epoch

    # Transformation
    transform_gta_dataset = transform_gta()
    target_transform_gta = transform_gta_mask()

    print("Loading datasets")
    """
    type_aug_dict = {
    'color': ['HueSaturationValue', 'CLAHE', 'GaussNoise', 'RGBShift', 'RandomBrightnessContrast'],
    'weather': ['RandomShadow', 'RandomRain', 'RandomFog', 'ISONoise', 'GaussianBlur'],
    'geometric': ['RandomCrop', 'Affine', 'Perspective']
    }
    """

    type_aug = {} # CHANGE HERE!!!
    gta_train_nonaug = GTA5('./datasets/GTA5', transform_gta_dataset, target_transform_gta, augmentation=False, type_aug={}) # No type_aug 
    # Contains all pictures bc they are all augmented
    gta_train_aug = GTA5('./datasets/GTA5', transform_gta_dataset, target_transform_gta, augmentation=True, type_aug=type_aug) # Change the augm that you want

    # Choose with probability 0.5 the augmented images
    num_augmented = int(0.5 * len(gta_train_aug))
    indices = random.sample(range(len(gta_train_aug)), num_augmented)
    gta_train_aug = Subset(gta_train_aug, indices)

    # Union of the dataset
    gta_train = ConcatDataset([gta_train_nonaug, gta_train_aug]) # To obtain the final dataset = train + augment
    
    # Create dataloader
    full_dataloader_gta_train, _ = dataloader(gta_train, None, batch_size, True, True, False, 4)
    full_dataloader_cityscapes_train, _ = dataloader(CityScapes('./datasets/Cityscapes', transform=transform_cityscapes(), target_transform=transform_cityscapes_mask()), None, batch_size, True, True)
    # Take a subset of the dataloader
    #dataloader_gta_train = select_random_fraction_of_dataset(full_dataloader_gta_train, fraction=0.25, batch_size=batch_size)
    
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

    discriminators, discriminators_optimizers = generate_discriminators(1, num_classes, device) # Generate 1 discriminator

    lambdas = [0.001, 0.001]  # Lambda values for the adversarial loss

    project_name = "4_Adversarial_Domain_Adaptation_base" #CHECK BEFORE RUNNING
    entity = "s281401-politecnico-di-torino" # New new entity Auro
    # entity = "s325951-politecnico-di-torino-mldl" # new team Lucia
    # entity="s328422-politecnico-di-torino" # old team Betta
    run = wandb.init(project=project_name, entity=entity, name=f"epoch_{start_epoch}", reinit=True)
    wandb.config.update({
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "num_classes": num_classes
    })
    
    
    if start_epoch > 1:
        artifact = wandb.use_artifact(f"{project_name}/model_epoch_{start_epoch-1}:latest", type="model")
        checkpoint_path = artifact.download()
        checkpoint = torch.load(os.path.join(checkpoint_path, f"model_epoch_{start_epoch-1}.pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for i, discriminator in enumerate(discriminators):
            artifact = wandb.use_artifact(f"{project_name}/discriminator_{i+1}_epoch_{start_epoch-1}:latest", type="model")
            checkpoint_path = artifact.download()
            checkpoint = torch.load(os.path.join(checkpoint_path, f"discriminator_{i+1}_epoch_{start_epoch-1}.pt"))
            discriminator.load_state_dict(checkpoint['model_state_dict'])
            discriminators_optimizers[i].load_state_dict(checkpoint['optimizer_state_dict'])


    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}")
        start_train = time.time()

        if epoch % 10 == 0:
            compute_mIoU = True
        else:
            compute_mIoU = False

        metrics_train, iter_curr = train_with_adversary(epoch, model, discriminators, full_dataloader_gta_train, full_dataloader_cityscapes_train, loss, optimizer, discriminators_optimizers, iter_curr,
                                                        learning_rate, num_classes, max_iter, lambdas, compute_mIoU)
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

        for i, (discriminator, optimizer_d) in enumerate(zip(discriminators, discriminators_optimizers)):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer_d.state_dict()
            }
            save_path_model = f"discriminator_{i+1}_epoch_{epoch}.pt"
            torch.save(checkpoint, save_path_model)
            artifact = wandb.Artifact(f"discriminator_{i+1}_epoch_{epoch}", type="model")
            artifact.add_file(save_path_model)
            run.log_artifact(artifact)
            os.remove(save_path_model)

        wandb.finish()
        run = wandb.init(project=project_name, entity=entity, name=f"epoch_{epoch}", reinit=True)

        
