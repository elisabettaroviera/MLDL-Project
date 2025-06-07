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
from utils.metrics import compute_miou, compute_latency_and_fps, compute_flops, compute_parameters
from PIL import Image
import torch.nn.functional as F

# Function to save sample images,ground truth color masks, prediction color masks
def save_images(flag_save, save_dir,inputs, file_names, preds,file_name_1, file_name_2):
    resize_transform = transforms.Resize((512, 1024))  # Resize da applicare
    # color map       
    CITYSCAPES_COLORMAP = np.array([
        [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
        [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
        [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
        [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
    ], dtype=np.uint8)
    
    for input, file_name, pred in zip(inputs, file_names, preds):
        if file_name in [file_name_1, file_name_2]:
            flag_save += 1

            # Store the original image from 'inputs' in tensor form
            original_img_path = os.path.join("/kaggle/input/cityscapes-dataset/Cityscapes/Cityspaces/images/val/frankfurt", file_name)
            original_img = Image.open(original_img_path).convert('RGB')

            # Resize the image
            resized_img = resize_transform(original_img)
            resized_img.save(f"{save_dir}/{file_name}_image_original.png")

            # Save the predicted colored mask
            color_mask = CITYSCAPES_COLORMAP[pred]
            color_mask_img = Image.fromarray(color_mask)  
            color_mask_img.save(f"{save_dir}/{file_name}_pred_color.png")

            # Store the colored target mask
            gt_file_name = file_name.replace("leftImg8bit", "gtFine_color")
            gt_path = os.path.join("/kaggle/input/cityscapes-dataset/Cityscapes/Cityspaces/gtFine/val/frankfurt", gt_file_name)
            # non funziona con kaggle
            color_target_img = Image.open(gt_path).convert('RGB')
            resized_target = resize_transform(color_target_img)
            resized_target.save(f"{save_dir}/{file_name}_color_target.png")

    return flag_save

# VALIDATION LOOP
def validate(epoch, new_model, val_loader, criterion, num_classes):
    var_model = os.environ['MODEL']

    # 1. Obtain the pretrained model 
    model = new_model
    print("Validating the model...")
    
    # 2. Initialize the metrics variables    
    print("Initializing the metrics variables...")
    mean_loss = 0
    running_loss = 0.0
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)

    # 3. Start the validation of the model
    print("Starting the validation of the model...")
    model.eval()

    print(f"Validating on {len(val_loader)} batches") 
    
    # Make sure the cartella outputs exists
    save_dir = f'./outputs/{var_model}_outputs'
    os.makedirs(save_dir, exist_ok=True)
    flag_save = 0

    # Image which we want to save the predicted masks of
    # frankfurt_000001_054640_gtFine_color.png
    file_name_1 = "frankfurt_000001_054640_leftImg8bit.png"
    # frankfurt_000001_062016_gtFine_color.png
    file_name_2 = "frankfurt_000001_062016_leftImg8bit.png"
    

    # 4. Loop on the batches of the dataset
    with torch.no_grad(): # NOT compute the gradient (we already computed in the previous step)
        for batch_idx, (inputs, targets, file_names) in enumerate(val_loader):
            if batch_idx % 100 == 0: # Print every 100 batches
                print(f"Batch {batch_idx}/{len(val_loader)}")
            inputs, targets = inputs.cuda(), targets.cuda()

            # Compute output of the model
            outputs = model(inputs) # Predicted
            
            # Compute the loss
            loss = criterion(outputs, targets)

            # Update the running loss
            running_loss += loss.item() 

            # Convert model outputs to predicted class labels
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            gts = targets.detach().cpu().numpy()
            
            # Accumulate intersections and unions per class
            _, _, inters, unions = compute_miou(gts, preds, num_classes)
            total_intersections += inters
            total_unions += unions

            # Only enter the loop if we haven't saved both images    
            #if flag_save < 2:
            #   flag_save = save_images(flag_save,save_dir,inputs, file_names, preds, file_name_1, file_name_2)


    # 5. Compute the metrics for the validation set 
    # 5.a Compute the accuracy metrics, i.e. mIoU and mean loss
    print("Computing the metrics for the validation set...")

    iou_per_class = (total_intersections / (total_unions + 1e-10)) * 100
    iou_non_zero = np.array(iou_per_class)
    iou_non_zero = iou_non_zero[np.nonzero(iou_non_zero)]
    
    mean_iou = np.nanmean(iou_non_zero) # Compute mIoU without considering the NaN value       
    mean_loss = running_loss / len(val_loader)
    
    # 5.b Compute the computation metrics, i.e. FLOPs, latency, number of parameters (only at the last epoch)
    if epoch == 50:
        print("Computing the computation metrics...")

        mean_latency, std_latency, mean_fps, std_fps = compute_latency_and_fps(model, height=512, width=1024, iterations=1000)
        print(f"Latency: {mean_latency:.2f} ± {std_latency:.2f} ms | FPS: {mean_fps:.2f} ± {std_fps:.2f}")

        num_flops = compute_flops(model, height=512, width=1024)
        print(f"Total numer of FLOPS: {num_flops} GigaFLOPs")

        tot_params, trainable_params = compute_parameters(model)
        print(f"Total Params: {tot_params}, Trainable: {trainable_params}")
    else:
        # NB: metric = -1 means we have not computed it (we compute only at the last epoch)
        mean_latency = -1
        std_latency = -1
        num_flops = -1
        trainable_params = -1
        mean_fps = -1
        std_fps = -1

    # 6. Return all the metrics
    metrics = {
        'mean_loss': mean_loss,
        'mean_iou': mean_iou,
        'iou_per_class': iou_per_class,
        'mean_latency' : mean_latency,
        'std_latency' : std_latency,
        'mean_fps' : mean_fps,
        'std_fps' : std_fps,
        'num_flops' : num_flops,
        'trainable_params': trainable_params
    }

    return metrics

# VALIDATION LOOP
def validate_pidnet(epoch, new_model, val_loader, criterion, num_classes):


    # 1. Obtain the pretrained model 
    model = new_model
    print("Validating the model...")
    
    # 2. Initialize the metrics variables    
    print("Initializing the metrics variables...")
    mean_loss = 0
    running_loss = 0.0
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)

    # 3. Start the validation of the model
    print("Starting the validation of the model...")
    model.eval()

    print(f"Validating on {len(val_loader)} batches") 
    

    # 4. Loop on the batches of the dataset
    with torch.no_grad(): # NOT compute the gradient (we already computed in the previous step)
        for batch_idx, (inputs, targets, file_names) in enumerate(val_loader):
            if batch_idx % 100 == 0: # Print every 100 batches
                print(f"Batch {batch_idx}/{len(val_loader)}")
            inputs, targets = inputs.cuda(), targets.cuda()

            # Compute output of the model
            outputs = model(inputs) # Predicted
            outputs_up = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
            loss = criterion(outputs_up, targets)

            # Update the running loss
            running_loss += loss.item()

            # Convert model outputs to predicted class labels
            preds = outputs_up.argmax(dim=1).detach().cpu().numpy()
            gts = targets.detach().cpu().numpy()
            
            # Accumulate intersections and unions per class
            _, _, inters, unions = compute_miou(gts, preds, num_classes)
            total_intersections += inters
            total_unions += unions



    # 5. Compute the metrics for the validation set 
    # 5.a Compute the accuracy metrics, i.e. mIoU and mean loss
    print("Computing the metrics for the validation set...")

    iou_per_class = (total_intersections / (total_unions + 1e-10)) * 100
    iou_non_zero = np.array(iou_per_class)
    iou_non_zero = iou_non_zero[np.nonzero(iou_non_zero)]
    
    mean_iou = np.nanmean(iou_non_zero) # Compute mIoU without considering the NaN value       
    mean_loss = running_loss / len(val_loader)
    
    # 5.b Compute the computation metrics, i.e. FLOPs, latency, number of parameters (only at the last epoch)
    if epoch == 50:
        print("Computing the computation metrics...")

        mean_latency, std_latency, mean_fps, std_fps = compute_latency_and_fps(model, height=512, width=1024, iterations=1000)
        print(f"Latency: {mean_latency:.2f} ± {std_latency:.2f} ms | FPS: {mean_fps:.2f} ± {std_fps:.2f}")

        num_flops = compute_flops(model, height=512, width=1024)
        print(f"Total numer of FLOPS: {num_flops} GigaFLOPs")

        tot_params, trainable_params = compute_parameters(model)
        print(f"Total Params: {tot_params}, Trainable: {trainable_params}")
    else:
        # NB: metric = -1 means we have not computed it (we compute only at the last epoch)
        mean_latency = -1
        std_latency = -1
        num_flops = -1
        trainable_params = -1
        mean_fps = -1
        std_fps = -1

    # 6. Return all the metrics
    metrics = {
        'mean_loss': mean_loss,
        'mean_iou': mean_iou,
        'iou_per_class': iou_per_class,
        'mean_latency' : mean_latency,
        'std_latency' : std_latency,
        'mean_fps' : mean_fps,
        'std_fps' : std_fps,
        'num_flops' : num_flops,
        'trainable_params': trainable_params
    }

    return metrics

