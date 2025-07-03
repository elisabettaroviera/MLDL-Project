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
            #original_img_path = os.path.join("./datasets/Cityscapes/Cityspaces/images/val/frankfurt", file_name) #colab
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
           # gt_path = os.path.join("./datasets/Cityscapes/Cityspaces/gtFine/val/frankfurt", gt_file_name) #colab
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
            if flag_save < 2:
                flag_save = save_images(flag_save,save_dir,inputs, file_names, preds, file_name_1, file_name_2)


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

##########################################################################################################################################
##################################################   PIDNET    #############################################################################
def get_boundary_map(target, kernel_size=3):
    # target: (B, H, W) with integer values [0, num_classes-1] or 255 for ignore

    # Create a mask that excludes pixels with value 255
    valid_mask = (target != 255).float()  # 1 where valid, 0 where 255

    target = target.clone().float()  # clone to avoid modifying the original
    target[target == 255] = 0  # replace 255 with 0 (any valid class)

    target = target.unsqueeze(1)  # (B, 1, H, W)

    # Define the Laplacian kernel to detect edges
    laplace_kernel = torch.tensor([[[[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]]]], device=target.device).float()

    # Apply convolution to get boundary map
    boundary = F.conv2d(target, laplace_kernel, padding=1).abs()

    # Apply the valid mask to zero out edges in ignored pixels
    boundary = boundary * valid_mask.unsqueeze(1)  # broadcast over channel

    boundary = (boundary > 0).float()  # binarize

    return boundary


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num

    # Assign higher weight to the minority class
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


def compute_pidnet_loss(criterion, x_extra_p, x_main, x_extra_d, target, boundary,
                        lambda_0=0.4, lambda_1=20.0, lambda_2=1.0, lambda_3=1.0):
    """
    Compute the total PIDNet loss composed of:
    - CE on the auxiliary branch
    - Weighted BCE on the boundary
    - CE on the main branch
    - CE focused on the boundary pixels only
    """

    # L0: CE loss on the auxiliary P branch
    loss_aux = criterion(x_extra_p, target)

    # L1: weighted BCE loss on D branch (boundaries)
    loss_bce = weighted_bce(x_extra_d, boundary)

    # L2: CE loss on the main branch
    loss_main = criterion(x_main, target)

    # L3: CE focused only on boundary pixels
    boundary_mask = (torch.sigmoid(x_extra_d).squeeze(1) > 0.8)
    masked_target = target[boundary_mask]
    valid_mask = (masked_target != 255)

    if valid_mask.any():
        masked_output = x_main.permute(0, 2, 3, 1)[boundary_mask][valid_mask]
        masked_target = masked_target[valid_mask]
        loss_boundary_ce = criterion(masked_output, masked_target)
    else:
        loss_boundary_ce = torch.tensor(0.0, device=target.device)

    # Final weighted combination
    total_loss = (
        lambda_0 * loss_aux +
        lambda_1 * loss_bce +
        lambda_2 * loss_main +
        lambda_3 * loss_boundary_ce
    )

    return total_loss, {
        "loss_aux": loss_aux.item(),
        "loss_bce": loss_bce.item(),
        "loss_main": loss_main.item(),
        "loss_boundary_ce": loss_boundary_ce.item()
    }


# VALIDATION LOOP
def validate_pidnet(epoch, new_model, val_loader, criterion, num_classes):
    # 1. Load the model to validate
    model = new_model
    print("Validating the model...")

    # 2. Initialize metrics
    print("Initializing metrics...")
    running_loss_total = 0.0
    running_loss_aux = 0.0
    running_loss_bce = 0.0
    running_loss_main = 0.0
    running_loss_boundary_ce = 0.0
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)

    # 3. Set model to eval mode
    print("Starting validation loop...")
    model.eval()

    print(f"Validating on {len(val_loader)} batches")

    # Directory to save output images
    save_dir = f'./outputs/PIDNET_outputs'
    os.makedirs(save_dir, exist_ok=True)
    flag_save = 0

    # Image names to save predicted masks for
    file_name_1 = "frankfurt_000001_054640_leftImg8bit.png"
    file_name_2 = "frankfurt_000001_062016_leftImg8bit.png"

    # Exponential decay for lambda_1
    lambda_1 = 20 * (0.9 ** (epoch / 10))

    # 4. Validation loop
    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, (inputs, targets, file_names) in enumerate(val_loader):
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(val_loader)}")

            inputs, targets = inputs.cuda(), targets.cuda()
            x_p, x_final, x_d = model(inputs)

            # Resize outputs to match target size
            x_p_up = F.interpolate(x_p, size=targets.shape[1:], mode='bilinear', align_corners=False)
            x_final_up = F.interpolate(x_final, size=targets.shape[1:], mode='bilinear', align_corners=False)
            x_d_up = F.interpolate(x_d, size=targets.shape[1:], mode='bilinear', align_corners=False)

            # Compute boundary map from target
            boundaries = get_boundary_map(targets)

            
            # Uncomment to compute loss during validation
            loss, loss_dict = compute_pidnet_loss(criterion, x_p_up, x_final_up, x_d_up, targets, boundaries, lambda_1=lambda_1)

            # Update loss tracking
            running_loss_total += loss.item()
            running_loss_aux += loss_dict['loss_aux']
            running_loss_bce += loss_dict['loss_bce']
            running_loss_main += loss_dict['loss_main']
            

            # Convert model output to predicted class labels
            preds = x_final_up.argmax(dim=1).detach().cpu().numpy()
            gts = targets.detach().cpu().numpy()

            # Update IoU metrics
            _, _, inters, unions = compute_miou(gts, preds, num_classes)
            total_intersections += inters
            total_unions += unions

            # Save predictions for specific images
            if flag_save < 2:
                flag_save = save_images(flag_save, save_dir, inputs, file_names, preds, file_name_1, file_name_2)

    # 5. Compute final validation metrics
    print("Computing final metrics...")

    iou_per_class = (total_intersections / (total_unions + 1e-10)) * 100
    iou_non_zero = iou_per_class[np.nonzero(iou_per_class)]
    mean_iou = np.nanmean(iou_non_zero)

    # Compute average losses
    mean_loss_total = running_loss_total / len(val_loader)
    mean_loss_aux = running_loss_aux / len(val_loader)
    mean_loss_bce = running_loss_bce / len(val_loader)
    mean_loss_main = running_loss_main / len(val_loader)
    mean_loss_boundary_ce = running_loss_boundary_ce / len(val_loader)

    # 5.b Optional: compute complexity metrics at final epoch
    if epoch == 50:
        print("Computing complexity metrics...")
        mean_latency, std_latency, mean_fps, std_fps = compute_latency_and_fps(model, height=512, width=1024, iterations=1000)
        print(f"Latency: {mean_latency:.2f} ± {std_latency:.2f} ms | FPS: {mean_fps:.2f} ± {std_fps:.2f}")

        num_flops = compute_flops(model, height=512, width=1024)
        print(f"Total FLOPs: {num_flops} GigaFLOPs")

        tot_params, trainable_params = compute_parameters(model)
        print(f"Total Parameters: {tot_params}, Trainable: {trainable_params}")
    else:
        # -1 means metric not computed at this epoch
        mean_latency = -1
        std_latency = -1
        mean_fps = -1
        std_fps = -1
        num_flops = -1
        trainable_params = -1

    # 6. Return all metrics as a dictionary
    metrics = {
        'mean_loss': mean_loss_total,
        'mean_loss_aux': mean_loss_aux,
        'mean_loss_bce': mean_loss_bce,
        'mean_loss_main': mean_loss_main,
        'mean_loss_boundary_ce': mean_loss_boundary_ce,
        'mean_iou': mean_iou,
        'iou_per_class': iou_per_class,
        'mean_latency': mean_latency,
        'std_latency': std_latency,
        'mean_fps': mean_fps,
        'std_fps': std_fps,
        'num_flops': num_flops,
        'trainable_params': trainable_params
    }

    return metrics
