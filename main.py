# Import necessary libraries
import os
from models.deeplabv2.deeplabv2 import lr_policy
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
from validation import validate
from utils.metrics import compute_miou


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
    print(f"{title} Metrics")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Latency: {metrics['latency']:.2f} ms")
    print(f"FPS: {metrics['fps']:.2f} frames/sec")
    print(f"FLOPs: {metrics['flops']:.2f} GFLOPs")
    print(f"Parameters: {metrics['params']:.2f} M")
    print(f"Mean IoU (mIoU): {metrics['miou']:.2f} %")

    print("\nClass-wise mIoU (%):")
    print(f"{'Class':<20} {'mIoU':>6}")
    print("-" * 28)
    for cls, val in metrics['miou_per_class'].items():
        print(f"{cls:<20} {val:>6.2f}")


if __name__ == "__main__":

    set_seed(23)  # Set a seed for reproducibility

    ############################################################################################################
    ################################################# STEP 2.a #################################################

    # Define transformations
    transform = transform_cityscapes()
    target_transform = transform_cityscapes_mask()

    # Load the datasets (Cityspaces)
    cs_train = CityScapes('./datasets/Cityscapes', 'train', transform, target_transform)
    cs_val = CityScapes('./datasets/Cityscapes', 'val', transform, target_transform)

    # DataLoader
    # also saving filenames for the images, when doing train i should not need them
    # each batch is a nuple: images, masks, filenames 
    # I modify the value of the batch size because it has to match with the one of the model
    batch_size = 2 # 3 or the number the we will use in the model
    dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, cs_val, batch_size, True, True)

    # Create output dir if needed
    os.makedirs('./outputs', exist_ok=True)

    # Get first batch
    first_batch = next(iter(dataloader_cs_train))
    # I need filenames to save the images, but i don't think we need it when doing training
    # The idea is to save some images and masks for the report (maybe we can understand if the image is so much worse at the first
    # epochs is so much worse that at the 50 and discuss some comparison in the report)
    images, masks, filenames = first_batch

    # Number of samples you want to save from the batch
    num_to_save = min(5, len(images))  # e.g., save 5 or fewer

    for i in range(num_to_save):
        img_tensor = images[i]
        mask_tensor = masks[i]

        # Check the pixel values of the first mask in the batch, each value should be in the range [0, 18]? (bc 19 classes) + 255 for void
        mask = mask_tensor.cpu().numpy()  # Convert mask tensor to NumPy array

        # Show the unique class values in the mask
        print(f"Unique class values in the mask: {np.unique(mask)}")

        img_pil = TF.to_pil_image(img_tensor.cpu())
        # Convert mask tensor to PIL image, i am using long int64 to keep the class labels but image fromarray doen't support them
        mask_pil = Image.fromarray(mask_tensor.byte().cpu().numpy())  # Convert to uint8 before Image.fromarray

        base_filename = filenames[i].replace("leftImg8bit", "")
        img_path = f'./outputs/{base_filename}_image.png'
        mask_path = f'./outputs/{base_filename}_mask.png'

        img_pil.save(img_path)
        mask_pil.save(mask_path)

        print(f"Saved image to {img_path}")
        print(f"Saved mask to {mask_path}")
        
    print("************trying out compute_miou:***************")
    # Dummy ground truth and prediction with 3 classes 
    gt_images = [
    np.array([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]
    ]),
    np.array([
        [2, 2, 2],
        [1, 1, 1],
        [0, 0, 0]
    ])]
    pred_images = [
    np.array([
        [0, 1, 2],   # correct
        [0, 0, 2],   # 1 → 0 (mistake)
        [0, 1, 1]    # 2 → 1 (mistake)
    ]),
    np.array([
        [2, 1, 2],   # 2 → 1 (mistake)
        [1, 1, 1],   # correct
        [0, 1, 0]    # 1 → 0 (mistake)
    ])]

    mean_iou_dummy, iou_per_class_dummy = compute_miou(gt_images, pred_images, num_classes=3)
    print("__________dummy try_________")
    print("Mean IoU:", mean_iou_dummy)
    print("IoU per class:", iou_per_class_dummy)

    print("_________try with saved masks________")
    gt_mask = np.array(Image.open("outputs/monchengladbach_000000_019500_.png_mask.png").convert("L"))
    #gt_mask = np.array(Image.open("outputs/hanover_000000_006922_.png_mask.png").convert("L"))
    print("GT labels:", np.unique(gt_mask))
    valid_mask = gt_mask != 255
    num_classes = int(np.max(gt_mask[valid_mask]) + 1)
    mean_iou, iou_per_class = compute_miou(gt_mask, gt_mask, num_classes)
    print(f"mean iou = {mean_iou}")
    print(f"iou per class= {iou_per_class}")

    '''
    # Definition of the parameters
    # Search on the pdf!! REVIEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    num_epochs = 50 # Number of epochs
    num_classes = 19 # Number of classes in the dataset (Cityscapes)
    
    learning_rate = 0.001 # Learning rate for the optimizer
    momentum = 0.9 # Momentum for the optimizer
    weight_decay = 0.0005 # Weight decay for the optimizer
    batch_size = 2 # Batch size for the DataLoader

    # FOR LOOP ON THE EPOCHS
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}")

        print("Load the model")
        # 1. Obtain the pretrained model
        model = None # Put the model with wandb

        loss = torch.nn.CrossEntropyLoss() # Loss function (CrossEntropyLoss for segmentation tasks)
        
        # MODIFY THE LEARNING RATE: we have to update the lr at each batch, not only at the beginning of the epoch
        lr = lr_policy(optimizer, init_lr=learning_rate, iter=current_iter, max_iter=total_iters)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay) # Optimizer (Stochastic Gradient Descent)

        # 2. Training step
        print("Training step")
        metrics_train, new_model = train(epoch, model, dataloader_cs_train, loss, optimizer)
        print("Training step done")

        # PRINT all the metrics!
        print_metrics("Training", metrics_train)

        # 3. Validation step
        print("Validation step")
        metrics_val = validate(new_model, dataloader_cs_val, loss) # Compute the accuracy on the validation set
        print("Validation step done")

        # PRINT all the metrics!
        print_metrics("Validation", metrics_val)
    '''
