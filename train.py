# TODO: Define here your training and validation loops.

# Import necessary libraries
import os
import torch
from torchvision.datasets import ImageFolder
from datasets.transform_datasets import *

if __name__ == "__main__":

    #### STEP 2.a
    # Load the dataset CITYSCAPES
    cs_train = ImageFolder(root='./datasets/Cityscapes/Cityspaces/images/train', transform=transform_cityscapes())
    cs_val = ImageFolder(root='./datasets/Cityscapes/Cityspaces/images/val', transform=transform_cityscapes())

    # DataLoader
    dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, cs_val, 64, True, True)

    # Determine the number of classes and samples
    num_classes = len(cityscapes_train.classes)
    num_samples = len(cityscapes_train)

    print(f'Number of classes: {num_classes}')
    print(f'Number of samples: {num_samples}')
