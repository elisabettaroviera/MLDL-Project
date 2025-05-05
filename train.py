# TODO: Define here your training and validation loops.

# Import necessary libraries
import os
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

"""   

    DEFINION OF THE METRICS
    1. mIoU% --> compute_miou
    2. Latency & FPS --> compute_latency_and_fps
    3. FLOPs --> compute_flops

"""

# 1. mIoU% 
# Function to compute the mean Intersection over Union (mIoU) for a given set of predictions and targets
# gt_images := all the ground truth images (all the images of the dataset)
# pred_images := all the predicted images (all the images of the dataset)
# num_classes := number of classes in the dataset
def compute_miou(gt_images, pred_images, num_classes):
    iou_per_class = np.zeros(num_classes)
    eps = 1e-10  # To avoid division by 0

    for class_id in range(num_classes):
        total_intersection = 0
        total_union = 0

        # gt_im := ground truth image (only one image)
        # pred_im := predicted image (only one image)
        for gt_im, pred_im in zip(gt_images, pred_images):
            target = (gt_im == class_id)
            prediction = (pred_im == class_id)

            intersection = np.logical_and(target, prediction).sum()
            union = np.logical_or(target, prediction).sum()

            total_intersection += intersection
            total_union += union

        if total_union > 0:
            iou_per_class[class_id] = (total_intersection / (total_union + eps)) * 100  # Convert to percentage
        else:
            iou_per_class[class_id] = np.nan  # If the class is not present in the image, set to NaN

    mean_iou = np.nanmean(iou_per_class) * 100  # Convert to percentage
    return mean_iou, iou_per_class


# 2. Latency & FPS
# Function to compute the latency and FPS of a model on a given input size
# model := the model to be evaluated
# height := the height of the input image
# width := the width of the input image
# iterations := number of iterations to compute the latency and FPS
def compute_latency_and_fps(model, height=512, width=1024, iterations=1000):
    # Generates a random image with shape (3, height, width)
    image = np.random.rand(3, height, width).astype(np.float32)

    latencies = []
    fps_values = []

    for _ in range(iterations):
        start = time.time()

        output = model(image) # Execution of the model on the image

        end = time.time()

        # Latency is the time taken to process the image in seconds
        latency = end - start
        latencies.append(latency)

        # FPS is the number of frames processed per second
        fps = 1.0 / latency if latency > 0 else 0
        fps_values.append(fps)

    # Conversion of latency in milliseconds
    mean_latency = np.mean(latencies) * 1000
    std_latency = np.std(latencies) * 1000

    mean_fps = np.mean(fps_values)
    std_fps = np.std(fps_values)

    return mean_latency, std_latency, mean_fps, std_fps

# 3. FLOPs
# Function to compute the FLOPs of a model on a given input size
# model := the model to be evaluated
# height := the height of the input image
# width := the width of the input image
# iterations := number of iterations to compute the FLOPs
def compute_flops(model, height=512, width=1024):
    image = torch.zeros((3, height, width)) # Generesate a random image with shape (3, height, width) with all zeros

    flops = FlopCountAnalysis(model, image) # Compute the FLOPs of the model on the image

    return flop_count_table(flops)

# 4. Parameters
def compute_parameters(model):
    # Compute the number of parameters in the model
    tot_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return tot_params, trainable_params




# TRAIN LOOP
def train(epoch, old_model, dataloader_train, criterion, optimizer): # criterion = loss
    # 1. Obtain the pretrained model
    model = old_model 

    # 2. Set the hyperparameters of the model i.e. learning rate, optimizer, etc.
    running_loss = 0.0 
    correct = 0
    total = 0
    iou_per_class = {}

    # 3. Start the training of the model
    model.train() 
    
    # 4. Loop on the batches of the dataset
    for batch_idx, (inputs, targets) in enumerate(dataloader_train): #(X,y)
        inputs, targets = inputs.cuda(), targets.cuda() # GPU

        # Compute outout of the train
        outputs = model(inputs) 
        
        # Compute the loss
        loss = criterion(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the accuracy metrics, i.e. mIoU 
        running_loss += loss.item() # Update of the loss = contain the total loss of the epoch
        for target, output in zip(targets, outputs):
            iou_per_class = compute_miou(target, output, num_classes) 


        

    train_loss = running_loss / len(dataloader_train) # Mean loss of the all dataset
    
    # 5. Compute the computation metrics, i.e. FLOPs
    compute_latency_and_fps(model, height=512, width=1024, iterations=1000)
    compute_flops(model, height=512, width=1024)
    compute_parameters(model)

    # 6. SAVE THE PARAMETERS OF THE MODEL 
    # 6. SAVE MODEL!

    # 7. Return all the metrics
    metrics = {}
    return metrics, new_model

# VALIDATION LOOP
def validate(new_model, val_loader, criterion):
    # 1. Obtai the pretrained model 
    model = new_model

    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad(): # NOT compute the gradient (we already computed in the previous step)
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs) # Predicted
            
            # Compute the accuracy metrics, i.e. mIoU 
            loss = criterion(outputs, targets) # Computation of the loss

            # As computed in the train part
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 2. Compute the computation metrics, i.e. FLOPs
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    # 3. Return all the metrics
    metrics = {}
    return metrics




if __name__ == "__main__":

    #### STEP 2.a
    # Load the dataset CITYSCAPES
    # Define transformations
    transform = transform_cityscapes()
    target_transform = transform_cityscapes_mask()

    # Load the datasets (Cityspaces)
    cs_train = CityScapes('./datasets/Cityscapes', 'train', transform, target_transform)
    cs_val = CityScapes('./datasets/Cityscapes', 'val', transform, target_transform)

    # DataLoader
    dataloader_cs_train, dataloader_cs_val = dataloader(cs_train, cs_val, 64, True, True)

    # Create output dir if needed
    os.makedirs('./outputs', exist_ok=True)

    # Get first batch
    first_batch = next(iter(dataloader_cs_train))
    images, masks, filenames = first_batch

    # Number of samples you want to save from the batch
    num_to_save = min(5, len(images))  # e.g., save 5 or fewer

    for i in range(num_to_save):
        img_tensor = images[i]
        mask_tensor = masks[i]

        # Check the pixel values of the first mask in the batch
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
    # Definition of the parameters
    # Search on the pdf!!
    num_epochs = 50 # Number of epochs

    # FOR LOOP ON THE EPOCHS
    for epoch in range(1, num_epochs + 1):
        # 1. Obtain the pretrained model
        model = None # Put the model with wandb

        # Training
        metrics_train, new_model = train(epoch, model, dataloader_cs_train, criterion, optimizer)
        # PRINT all the metrics!

        # Validation
        metrics_val = validate(new_model, val_loader, criterion) # Compute the accuracy on the validation set
        
        # PRINT all the metrics!

