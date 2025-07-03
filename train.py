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
from utils.utils import poly_lr_scheduler
import wandb
import torch.nn.functional as F

##########################################################################################################################################
############################################################### MAIN 2 a b ###############################################################
##########################################################################################################################################
# TRAIN LOOP
def train(epoch, old_model, dataloader_train, criterion, optimizer, iter, learning_rate, num_classes, max_iter): # criterion == loss function
    var_model = os.environ['MODEL'] 

    # 1. Obtain the pretrained model
    model = old_model 
    print("Training the model...")

    # 2. Initialize the metrics variables and hyperparameters
    print("Initializing the metrics variables...")
    
    running_loss = 0.0 
    mean_loss = 0.0
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)

    # 3. Start the training of the model
    print("Starting the training of the model...")
    model.train() 

    print(f"Training on {len(dataloader_train)} batches")
    
    # 4. Loop on the batches of the dataset
    for batch_idx, (inputs, targets, file_names) in enumerate(dataloader_train): 
        if batch_idx % 100 == 0: # Print every 100 batches
            print(f"Batch {batch_idx}/{len(dataloader_train)}")

        iter += 1 # Increment the iteration counter

        inputs, targets = inputs.cuda(), targets.cuda() # GPU

        # Compute output of the train
        outputs = model(inputs)        

        # Compute the loss
        # DeepLabV2 returns for training the output, None, None
        # BiseNet returns the output, aux1, aux2 (aux are predictions from contextpath)
        loss = criterion(outputs[0], targets)
        if var_model == "BiSeNet":
            alpha = 1 # In the paper they use 1
            loss +=  alpha * criterion(outputs[1], targets) + alpha *  criterion(outputs[2], targets)
             
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the learning rate
        lr = poly_lr_scheduler(optimizer, init_lr=learning_rate, iter=iter, lr_decay_iter=1, max_iter=max_iter, power=0.9)

        # Update the running loss
        running_loss += loss.item() # Update of the loss == contain the total loss of the epoch

        # Convert model outputs to predicted class labels
        preds = outputs[0].argmax(dim=1).detach().cpu().numpy()
        gts = targets.detach().cpu().numpy()
        
        # Accumulate intersections and unions per class
        _, _, inters, unions = compute_miou(gts, preds, num_classes)
        total_intersections += inters
        total_unions += unions

    # 5. Compute the metrics for the training set 
    # 5.a Compute the standard metrics for all the epochs
    print("Computing the metrics for the training set...")

    iou_per_class = (total_intersections / (total_unions + 1e-10)) * 100
    iou_non_zero = np.array(iou_per_class)
    iou_non_zero = iou_non_zero[np.nonzero(iou_non_zero)]

    # Compute the mean without considering NaN value
    mean_iou = np.nanmean(iou_non_zero) 
    mean_loss = running_loss / len(dataloader_train)    

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


    # 6. Save the parameter of the model 
    print("Saving the model")

    wandb.log({
        "epoch": epoch,
        "loss": mean_loss,
        "lr": lr
    })

    # Save the model weight at each epoch
    model_save_path = f"model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': mean_loss,
    }, model_save_path)

    # Create a new artefact for the current epoch
    artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
    artifact.add_file(model_save_path) 

    # Store the artefact on wandb
    wandb.log_artifact(artifact)
    print(f"Model saved for epoch {epoch}")

    # 7. Return all the metrics
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

    return metrics, iter

def train_with_adversary(epoch, old_model, discriminators, dataloader_source_train, dataloader_target_train, 
                         criterion, optimizer, discriminator_optimizers, iteration, learning_rate, num_classes, max_iter, lambdas, compute_mIoU = True, trial_type = "bce_fixed"): # criterion == loss function
   
    # --------------------------- BASIC DEFINITIONS -------------------------------------- #
    try:
        var_model = os.environ['MODEL'] 
    except KeyError:
        print("Environment variable 'MODEL' not set. Using default model 'BiSeNet'.")
        var_model = "BiSeNet"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = old_model
    running_loss = 0.0 
    mean_loss = 0.0
    total_intersections = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total_unions = torch.zeros(num_classes, dtype=torch.float64, device=device)
    target_label = 0
    source_label = 1
    target_iter = iter(dataloader_target_train) # Create an iterator for the target dataset
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for discriminator in discriminators:
        discriminator.train()
    bisenet_accumulator = 0.0
    discriminator_accumulator = 0.0
    statistics_accumulator = 0.0

    start_time = time.time()

    # --------------------------- TRAINING LOOP -------------------------------------- #
    for batch_idx, (inputs_src, targets_src, file_names) in enumerate(dataloader_source_train): 
        if batch_idx % 100 == 0: # Print every 100 batches
            print(f"Batch {batch_idx}/{len(dataloader_source_train)}")

        # ------------------- TRAINING BISENET WITH ADVERSARIAL LOSS ------------------- #w
        bisenet_start = time.time()
        locking_start = time.time()
        for discriminator in discriminators:
            lock_model(discriminator) # Lock the discriminator parameters to avoid training them
        locking_end = time.time()
        #print(f"Discriminator locking time: {locking_end - locking_start:.2f} seconds")


        iteration += 1 # Increment the iteration counter
        inputs_src, targets_src = inputs_src.to(device), targets_src.to(device) # GPU

        # Compute output of the train
        strat_compute_loss = time.time()
        with torch.cuda.amp.autocast():
            outputs = model(inputs_src)        
            loss = criterion(outputs[0], targets_src)
        
            alpha = 1 # In the paper they use 1
            loss +=  alpha * criterion(outputs[1], targets_src) + alpha *  criterion(outputs[2], targets_src)
        end_compute_loss = time.time()
        #print(f"Loss computation time: {end_compute_loss - strat_compute_loss:.2f} seconds")

        # Get the next batch from the target dataset
        start_get_target = time.time()
        try:
            inputs_target, _, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(dataloader_target_train)
            inputs_target, _, _ = next(target_iter)
        inputs_target = inputs_target.to(device)
        end_get_target = time.time()
        #print(f"Target dataset batch retrieval time: {end_get_target - start_get_target:.2f} seconds")

        # Compute the output of the target dataset
        start_model = time.time()
        outputs_target = model(inputs_target)
        end_model = time.time()
        #print(f"Model inference time: {end_model - start_model:.2f} seconds")

        # Compute the adversarial loss
        start_adversarial = time.time()
        with torch.cuda.amp.autocast():
            #adv_loss = adversarial_loss(discriminators, outputs_target, target_label, source_label, device, lambdas)
            lambda_dynamic = get_lambda_adv(iteration, max_iter, trial_type)
            if trial_type == "bce_confidence":
                with torch.no_grad():
                    D_out = discriminators[0](torch.softmax(outputs_target[0], dim=1))
                    lambda_dynamic = 0.001 * (1 - torch.sigmoid(D_out).mean().item())
            lambdas = [lambda_dynamic for _ in discriminators]
            adv_loss = adversarial_loss(discriminators, outputs_target, target_label, source_label, device, lambdas, trial_type)
        # Combine the losses
        loss += adv_loss
        end_adversarial = time.time()
        #print(f"Adversarial loss computation time: {end_adversarial - start_adversarial:.2f} seconds")

        start_learning = time.time()
        # Backpropagation
        optimizer = backpropagate(optimizer, loss, scaler)

        # Compute the learning rate
        lr = poly_lr_scheduler(optimizer, init_lr=learning_rate, iter=iteration, lr_decay_iter=1, max_iter=max_iter, power=0.9)
        end_learning = time.time()
        #print(f"Learning rate computation time: {end_learning - start_learning:.2f} seconds")

        # Update the running loss
        running_loss += loss.item() # Update of the loss == contain the total loss of the epoch

        # Convert model outputs to predicted class labels
        bisenet_end = time.time()
        print(f"BiSeNet training time: {bisenet_end - bisenet_start:.2f} seconds")
        bisenet_accumulator += (bisenet_end - bisenet_start)

        # ------------------- TRAINING DISCRIMINATORS ------------------- #
        discriminator_start = time.time()
        lock_model(model) # Lock the model parameters to avoid training them
        for discriminator in discriminators:
            unlock_model(discriminator)
        
        with torch.no_grad():
            outputs_source = model(inputs_src)
            outputs_target = model(inputs_target)
        


        for i, (discriminator, disc_optimizer) in enumerate(zip(discriminators, discriminator_optimizers)):
            disc_optimizer.zero_grad()
            
            # Compute the softmax outputs for source and target
            softmax_src = softmax(outputs_source[i], dim=1).detach()
            softmax_tgt = softmax(outputs_target[i], dim=1).detach()

            # SOURCE: discriminator deve dire "source_label"
            disc_pred_src = discriminator(softmax_src)
            # TARGET: discriminator deve dire "target_label"
            disc_pred_tgt = discriminator(softmax_tgt)
            
            '''
            loss_d_src = torch.nn.functional.binary_cross_entropy_with_logits(
                disc_pred_src,
                torch.full(disc_pred_src.shape, float(source_label), device=device)
            )

            loss_d_tgt = torch.nn.functional.binary_cross_entropy_with_logits(
                disc_pred_tgt,
                torch.full(disc_pred_tgt.shape, float(target_label), device=device)
            )
            '''

            if trial_type.startswith("hinge"):
                loss_d_src = torch.relu(1.0 - disc_pred_src).mean()
                loss_d_tgt = torch.relu(1.0 + disc_pred_tgt).mean()
            elif trial_type.startswith("mse"):
                loss_d_src = F.mse_loss(disc_pred_src, torch.full_like(disc_pred_src, float(source_label), device=device))
                loss_d_tgt = F.mse_loss(disc_pred_tgt, torch.full_like(disc_pred_tgt, float(target_label), device=device))
            else:
                loss_d_src = F.binary_cross_entropy_with_logits(disc_pred_src, torch.full_like(disc_pred_src, float(source_label), device=device))
                loss_d_tgt = F.binary_cross_entropy_with_logits(disc_pred_tgt, torch.full_like(disc_pred_tgt, float(target_label), device=device))
        

            # Media delle due loss
            loss_d = 0.5 * (loss_d_src + loss_d_tgt)
            loss_d.backward()
            disc_optimizer.step()

        unlock_model(model) # Unlock the model parameters to allow training
        discriminator_end = time.time()

        print(f"Discriminator training time: {discriminator_end - discriminator_start:.2f} seconds")
        discriminator_accumulator += (discriminator_end - discriminator_start)

        if compute_mIoU:
            start_statistics = time.time()
            preds = outputs[0].argmax(dim=1)
            gts = targets_src.detach()
            

            # Accumulate intersections and unions per class
            # _, _, inters, unions = compute_miou_torch(gts, preds, num_classes) ## Loops
            _, _, inters, unions = compute_miou_torch_vectorized(gts, preds, num_classes, device) ## Vectorized
            total_intersections += inters
            total_unions += unions
            end_statistics = time.time()

        
            print(f"Statistics computation time: {end_statistics - start_statistics:.2f} seconds")
            statistics_accumulator += (end_statistics - start_statistics)

    del outputs, outputs_target, softmax_src, softmax_tgt, preds
    torch.cuda.empty_cache()
    gc.collect()

        
    # --------------------------- END OF TRAINING LOOP -------------------------------------- #

    # 5. Compute the metrics for the training set 
    # 5.a Compute the standard metrics for all the epochs
    print("Computing the metrics for the training set...")

    if compute_mIoU:
        start_metrics = time.time()

        iou_per_class = (total_intersections / (total_unions + 1e-10)) * 100
        iou_per_class_np = iou_per_class.detach().cpu().numpy()
        iou_non_zero = np.array(iou_per_class_np)
        iou_non_zero = iou_non_zero[np.nonzero(iou_non_zero)]

        # Compute the mean without considering NaN value
        mean_iou = np.nanmean(iou_non_zero) 
        mean_loss = running_loss / len(dataloader_source_train)    

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

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")

        # 6. Save the parameter of the model 
    print("Saving the model")

    wandb.log({
        "epoch": epoch,
        "loss": mean_loss,
        "lr": lr,
        "lambda_adv": lambdas[0]
    })

    # Save the model weight at each epoch
    model_save_path = f"model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': mean_loss,
    }, model_save_path)

    # Create a new artefact for the current epoch
    artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
    artifact.add_file(model_save_path) 

    # Store the artefact on wandb
    wandb.log_artifact(artifact)
    print(f"Model saved for epoch {epoch}")

    if compute_mIoU:
        # 7. Return all the metrics
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

    end_metrics = time.time()
    print(f"Metrics computation time: {end_metrics - start_metrics:.2f} seconds")
    print(f"Total BiSeNet training time: {bisenet_accumulator:.2f} seconds")
    print(f"Total Discriminator training time: {discriminator_accumulator:.2f} seconds")
    print(f"Total Statistics computation time: {statistics_accumulator:.2f} seconds")
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    return metrics, iteration

##########################################################################################################################################
##################################################   PIDNET    ###########################################################################
def get_boundary_map(target, kernel_size=3):
    # target: (B, H, W) with integer values [0, num_classes-1] or 255 for ignore

    # Create a mask to exclude pixels with value 255
    valid_mask = (target != 255).float()  # 1 where valid, 0 where 255

    target = target.clone().float()  # avoid modifying the original tensor
    target[target == 255] = 0  # replace 255 with 0 (any class)

    target = target.unsqueeze(1)  # (B,1,H,W)

    laplace_kernel = torch.tensor([[[[0, 1, 0],
                                     [1,-4, 1],
                                     [0, 1, 0]]]], device=target.device).float()

    boundary = F.conv2d(target, laplace_kernel, padding=1).abs()

    # Apply the validity mask to the result to zero out boundaries in ignored pixels
    boundary = boundary * valid_mask.unsqueeze(1)  # broadcast over channel

    boundary = (boundary > 0).float()

    return boundary

def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


def compute_pidnet_loss(criterion, x_extra_p, x_main, x_extra_d, target, boundary,
                        lambda_0=0.4, lambda_1=20.0, lambda_2=1.0, lambda_3=1.0):
    """
    Computes the total PIDNet loss, consisting of:
    - CE on the auxiliary P branch
    - Weighted BCE on boundaries
    - CE on the main branch
    - CE focused on boundary pixels

    Args:
        criterion: standard CE loss function (e.g., nn.CrossEntropyLoss(ignore_index=255))
        x_extra_p: output from the P branch (B, C, H, W)
        x_main: output from the main branch (B, C, H, W)
        x_extra_d: output from the D branch (B, 1, H, W)
        target: ground truth segmentation (B, H, W)
        boundary: binary edge map (B, 1, H, W)

    Returns:
        total_loss: weighted sum of the four components
        losses_dict: dictionary containing each individual loss
    """

    # L0: Auxiliary CE loss on the P branch
    loss_aux = criterion(x_extra_p, target)

    # L1: Weighted BCE on the D branch (edges)
    loss_bce = weighted_bce(x_extra_d, boundary)

    # L2: Main CE loss on the main branch
    loss_main = criterion(x_main, target)

    # L3: CE focused only on edge pixels
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


# TRAIN LOOP
def train_pidnet(epoch, old_model, dataloader_train, criterion, optimizer, iteration, learning_rate, num_classes, max_iter): # criterion == loss function

    # 1. Obtain the pretrained model
    model = old_model 
    print("Training the model...")

    # 2. Initialize the metrics variables and hyperparameters
    print("Initializing the metrics variables...")
    
    running_loss_total = 0.0
    running_loss_aux = 0.0
    running_loss_bce = 0.0
    running_loss_main = 0.0
    running_loss_boundary_ce = 0.0
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)

    # 3. Start the training of the model
    print("Starting the training of the model...")
    model.train() 

    print(f"Training on {len(dataloader_train)} batches")
    lambda_1 = 20* (0.9 ** (epoch / 10))  # exponential decay lambda_1
    
    # 4. Loop on the batches of the dataset
    for batch_idx, (inputs, targets, file_names) in enumerate(dataloader_train): 
        if batch_idx % 100 == 0: # Print every 100 batches
            print(f"Batch {batch_idx}/{len(dataloader_train)}")

        iteration += 1 # Increment the iteration counter

        inputs, targets = inputs.cuda(), targets.cuda() # GPU
        x_p, x_final, x_d = model(inputs)
        x_p_up = F.interpolate(x_p, size=targets.shape[1:], mode='bilinear', align_corners=False)
        x_final_up = F.interpolate(x_final, size=targets.shape[1:], mode='bilinear', align_corners=False)
        x_d_up = F.interpolate(x_d, size=targets.shape[1:], mode='bilinear', align_corners=False)

        boundaries = get_boundary_map(targets)

        loss, loss_dict = compute_pidnet_loss(criterion,x_p_up, x_final_up, x_d_up, targets, boundaries, lambda_1=lambda_1)
        #print(f"Loss: {loss.item():.4f} | Aux Loss: {loss_dict['loss_aux']:.4f} | BCE Loss: {loss_dict['loss_bce']:.4f} | Main Loss: {loss_dict['loss_main']:.4f} | Boundary CE Loss: {loss_dict['loss_boundary_ce']:.4f}")
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the learning rate

        lr = poly_lr_scheduler(optimizer, init_lr=learning_rate, iter=iteration, lr_decay_iter=1, max_iter=max_iter, power=0.95)


         # Update running losses
        running_loss_total += loss.item()
        running_loss_aux += loss_dict['loss_aux']
        running_loss_bce += loss_dict['loss_bce']
        running_loss_main += loss_dict['loss_main']
        running_loss_boundary_ce += loss_dict['loss_boundary_ce']

        # Convert model outputs to predicted class labels
        preds = x_final_up.argmax(dim=1).detach().cpu().numpy()
        gts = targets.detach().cpu().numpy()
        
        # Accumulate intersections and unions per class
        _, _, inters, unions = compute_miou(gts, preds, num_classes)
        total_intersections += inters
        total_unions += unions

    # 5. Compute the metrics for the training set 
    # 5.a Compute the standard metrics for all the epochs
    print("Computing the metrics for the training set...")

    iou_per_class = (total_intersections / (total_unions + 1e-10)) * 100
    iou_non_zero = np.array(iou_per_class)
    iou_non_zero = iou_non_zero[np.nonzero(iou_non_zero)]

    # Compute the mean without considering NaN value
    mean_iou = np.nanmean(iou_non_zero) 
    mean_loss_total = running_loss_total / len(dataloader_train)  
    mean_loss_aux = running_loss_aux / len(dataloader_train)
    mean_loss_bce = running_loss_bce / len(dataloader_train)
    mean_loss_main = running_loss_main / len(dataloader_train)
    mean_loss_boundary_ce = running_loss_boundary_ce / len(dataloader_train)  

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



    # 6. Save the parameter of the model 
    print("Saving the model")

    wandb.log({
        "epoch": epoch,
        'mean_loss': mean_loss_total,
        'mean_loss_boundary_ce': mean_loss_boundary_ce,
        'mean_loss_aux': mean_loss_aux,
        'mean_loss_bce': mean_loss_bce,
        'mean_loss_main': mean_loss_main,
        "lr": lr
    })

    # Save the model weight at each epoch
    model_save_path = f"model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': mean_loss_total,
    }, model_save_path)

    # Create a new artefact for the current epoch
    artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
    artifact.add_file(model_save_path) 

    # Store the artefact on wandb
    wandb.log_artifact(artifact)
    print(f"Model saved for epoch {epoch}")

    # 7. Return all the metrics
    metrics = {
        'mean_loss': mean_loss_total,
        'mean_loss_aux': mean_loss_aux,
        'mean_loss_bce': mean_loss_bce,
        'mean_loss_main': mean_loss_main,
        'mean_loss_boundary_ce': mean_loss_boundary_ce, 
        'mean_iou': mean_iou,
        'iou_per_class': iou_per_class,
        'mean_latency' : mean_latency,
        'std_latency' : std_latency,
        'mean_fps' : mean_fps,
        'std_fps' : std_fps,
        'num_flops' : num_flops,
        'trainable_params': trainable_params
    }

    return metrics, iteration
