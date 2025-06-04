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

if __name__ == "__main__":
    set_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("************ TRAINING BiSeNet ON GTA5 ***************")

    # Constant value
    batch_size = 4
    learning_rate = 0.00625
    momentum = 0.9
    weight_decay = 1e-4
    num_epochs = 50 #changed bc doing smaller runs
    num_classes = 19
    ignore_index = 255
    start_epoch = 46 #CHECK BEFORE RUNNING

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
    ## 1 TRASFORMAZIONE
    #type_aug = { 'color': ['HueSaturationValue']} #a) 3b_GTA5_to_CITY_aug_color_a_25percent --> parte con 20% poi scende a 15%
    #type_aug = {'color': ['CLAHE']} # b) 3b_GTA5_to_CITY_aug_color_b_25percent --> parte con 20% poi scende a 11%
    #type_aug = {'color': ['GaussNoise']} # c) 3b_GTA5_to_CITY_aug_color_c_25percent --> stabile sul 17%
    #type_aug = {'color': ['RGBShift']} # d) 3b_GTA5_to_CITY_aug_color_d_25percent --> parte con 30%(!!) poi scende a 20%
    #type_aug = {'color': ['RandomBrightnessContrast']} # e) 3b_GTA5_to_CITY_aug_color_e_25percent -->

    ## 2 TRASFORMAZIONI : 
    # a + cose
    #type_aug = { 'color': ['HueSaturationValue', 'CLAHE']} # a+b) 3b_GTA5_to_CITY_aug_color_a_b_25percent OKK
    #type_aug = { 'color': ['HueSaturationValue', 'GaussNoise']} # a+c) 3b_GTA5_to_CITY_aug_color_a_c_25percent OKK
    #type_aug = { 'color': ['HueSaturationValue', 'RGBShift']} # a+d) 3b_GTA5_to_CITY_aug_color_a_d_25percent OKK
    #type_aug = { 'color': ['HueSaturationValue', 'RandomBrightnessContrast']} # a+e) 3b_GTA5_to_CITY_aug_color_a_e_25percent OKK

    # b + cose
    #type_aug = { 'color': ['CLAHE', 'GaussNoise']} # b+c) 3b_GTA5_to_CITY_aug_color_b_c_25percent OKK
    #type_aug = { 'color': ['CLAHE', 'RGBShift']} # b+d) 3b_GTA5_to_CITY_aug_color_b_d_25percent OKK
    #type_aug = { 'color': ['CLAHE', 'RandomBrightnessContrast']} # b+e) 3b_GTA5_to_CITY_aug_color_b_e_25percent OKK

    # c + cose
    #type_aug = { 'color': ['GaussNoise', 'RGBShift']} # c+d) 3b_GTA5_to_CITY_aug_color_c_d_25percent OKK
    #type_aug = { 'color': ['GaussNoise', 'RandomBrightnessContrast']} # c+e) 3b_GTA5_to_CITY_aug_color_c_e_25percent OKK

    # d + cose
    #type_aug = { 'color': ['RGBShift', 'RandomBrightnessContrast']} # d+e) 3b_GTA5_to_CITY_aug_color_d_e_25percent OKK

    ## 3 TRASFORMAZIONI
    # a + b + cose
    #type_aug = { 'color': ['HueSaturationValue', 'CLAHE', 'GaussNoise']} # a+b+c) 3b_GTA5_to_CITY_aug_color_a_b_c_25percent OKK
    #type_aug = { 'color': ['HueSaturationValue', 'CLAHE', 'RGBShift']} # a+b+d) 3b_GTA5_to_CITY_aug_color_a_b_d_25percent  OKK
    #type_aug = { 'color': ['HueSaturationValue', 'CLAHE', 'RandomBrightnessContrast']} # a+b+e) 3b_GTA5_to_CITY_aug_color_a_b_e_25percent OKK
    # a + c + cose
    #type_aug = { 'color': ['HueSaturationValue', 'GaussNoise', 'RGBShift']} # a+c+d) 3b_GTA5_to_CITY_aug_color_a_c_d_25percent OKK
    #type_aug = { 'color': ['HueSaturationValue', 'GaussNoise', 'RandomBrightnessContrast']} # a+c+e) 3b_GTA5_to_CITY_aug_color_a_c_e_25percent OKK
    # a + d + e
    #type_aug = { 'color': ['HueSaturationValue', 'RGBShift', 'RandomBrightnessContrast']} # a+d+e) 3b_GTA5_to_CITY_aug_color_a_d_e_25percent OKK
    # b + c +cose 
    #type_aug = { 'color': ['CLAHE', 'GaussNoise']} # b+c) 3b_GTA5_to_CITY_aug_color_b_c_d_25percent OKK
    #type_aug = { 'color': ['CLAHE', 'RGBShift']} # b+d) 3b_GTA5_to_CITY_aug_color_b_c_e_25percent OKK
    # b + d + cose
    #type_aug = { 'color': ['CLAHE', 'RandomBrightnessContrast']} # b+e) 3b_GTA5_to_CITY_aug_color_b_d_e_25percent OKK
    # c + d +e
    #type_aug = { 'color': ['GaussNoise', 'RGBShift','RandomBrightnessContrast' ]} # c+d+e) 3b_GTA5_to_CITY_aug_color_c_d_e_25percent OKK

    ## 4 TRASFORMAZIONI
    #type_aug = { 'color': ['HueSaturationValue', 'CLAHE', 'GaussNoise', 'RGBShift']} # a+b+c+d) 3b_GTA5_to_CITY_aug_color_a_b_c_d_25percent OKK
    #type_aug = { 'color': ['HueSaturationValue', 'CLAHE', 'GaussNoise', 'RandomBrightnessContrast']} # a+b+c+e) 3b_GTA5_to_CITY_aug_color_a_b_c_e_25percent OKK
    #type_aug = { 'color': ['RandomBrightnessContrast', 'CLAHE', 'GaussNoise', 'RGBShift']} # b+c+d+e) 3b_GTA5_to_CITY_aug_color_b_c_d_e_25percent OKK
    #type_aug = { 'color': ['HueSaturationValue', 'GaussNoise', 'RGBShift', 'RandomBrightnessContrast']} # a+c+d+e) 3b_GTA5_to_CITY_aug_color_a_c_d_e_25percent OKK
    #type_aug = { 'color': ['HueSaturationValue', 'CLAHE', 'RGBShift', 'RandomBrightnessContrast']} # a+b+d+e) 3b_GTA5_to_CITY_aug_color_a_b_d_e_25percent OKK
  
    ## 5 TRASFORMAZIONI
    #type_aug = { 'color': ['HueSaturationValue', 'CLAHE', 'GaussNoise', 'RGBShift', 'RandomBrightnessContrast']} # a+b+c+d+e) 3b_GTA5_to_CITY_aug_color_a_b_c_d_e_25percent OKK

    # TRASFORMAZIONI  SU TUTTO IL DATASET
    # 1) hue + RGB + RB  (a + d + e)
    #type_aug = { 'color': ['HueSaturationValue', 'RGBShift', 'RandomBrightnessContrast']} # 3b_GTA5_to_CITY_aug_color_a_d_e_100_percent OKK
    # 2) RGB + RB (d + e)
    #type_aug = { 'color': ['RGBShift', 'RandomBrightnessContrast']} # a+b+c+d+e) 3b_GTA5_to_CITY_aug_color_d_e_100_percent OKK
    # 3) hue + clahe + RGB (a + b + d)
    #type_aug = { 'color': ['HueSaturationValue', 'CLAHE', 'RGBShift']} # a+b+c+d+e) 3b_GTA5_to_CITY_aug_color_a_b_d_100_percent OKK
    # 4) Gn + RGB + RB (c + d + e)
    #type_aug = { 'color': ['GaussNoise', 'RGBShift', 'RandomBrightnessContrast']} # a+b+c+d+e) 3b_GTA5_to_CITY_aug_color_c_d_e_100_percent OKK
    # 5) one of 4 best comb  of color 
    #type_aug = None # 3b_GTA5_to_CITY_aug_color_oneof_4_comb_100_percent OKK
    # 6) one of 2 best comb of color + best comb of weather
    #type_aug = None # 3b_GTA5_to_CITY_aug_color_weather_oneof_3_comb_100_percent OKK
    # 7) one of 3(!) best comb of color + best comb of weather
    #type_aug = None # 3b_GTA5_to_CITY_aug_color_weather_oneof_4_comb_100_percent OK val
    # 8) one of 4(!) best comb of color + best comb of weather
    type_aug = None # 3b_GTA5_to_CITY_aug_color_weather_oneof_5_comb_100_percent to val

    #type_aug = None # 3b_GTA5_to_CITY_aug_color_weather_rc_oneof_3_comb_100_percent forse non ha senso farla...

    # ALTRE CON 25 % DATASET MIX TRASFORMAZIONI
    #type_aug = None # 3b_GTA5_to_CITY_aug_color_weather_oneof_3_comb_25_percent OKK 
    #type_aug = None # 3b_GTA5_to_CITY_aug_color_geo_oneof_3_comb_25_percent OKK 
    #type_aug = None # 3b_GTA5_to_CITY_aug_color_weather_rc_oneof_3_comb_25_percent OKK

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
    wandb.login(key="2bc32b7d4d8f8601d9a93be55631ae9e18f78690")
    ################################ IN THIS BRANCH USE AURO TEAM : API 2bc32b7d4d8f8601d9a93be55631ae9e18f78690 ###################################
    for epoch in range(start_epoch, num_epochs + 1):
        project_name = "3b_GTA5_to_CITY_aug_color_weather_oneof_5_comb_100_percent" #CHECK BEFORE RUNNING________________________________________HERE
        entity = "s281401-politecnico-di-torino" #new team auro
        # "s325951-politecnico-di-torino-mldl" # new team Lucia
        # entity="s328422-politecnico-di-torino" # old team Betta
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
