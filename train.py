import os
import torch
from datasets.transform_datasets import *
import numpy as np
import time
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from utils.metrics import compute_miou, compute_latency_and_fps, compute_flops, compute_parameters
from utils.utils import poly_lr_scheduler
import wandb
import gc
import time

bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
softmax = torch.nn.functional.softmax

def lock_model(model):
    """
    Lock the model parameters to avoid training them.
    """
    for param in model.parameters():
        param.requires_grad = False
    return model

def unlock_model(model):
    """
    Unlock the model parameters to allow training.
    """
    for param in model.parameters():
        param.requires_grad = True
    return model

def backpropagate(optimizer, loss):
    """
    Perform backpropagation and optimization step.
    """
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()        # Backpropagate the loss
    optimizer.step()       # Update the model parameters
    return optimizer

def adversarial_loss(discriminators, outputs, target_label, source_label, device, lambdas):
    total_adv_loss = 0.0
    for i, discriminator in enumerate(discriminators):
        disc_pred = discriminator(softmax(outputs[0], dim=1).detach())
        target_tensor = torch.full(disc_pred.shape, float(source_label), device=device, dtype=disc_pred.dtype)
        adv_loss = bce_loss(disc_pred, target_tensor)
        total_adv_loss += lambdas[i]*adv_loss
    return total_adv_loss

# TRAIN LOOP
def train(epoch, old_model, dataloader_train, criterion, optimizer, iteration, learning_rate, num_classes, max_iter): # criterion == loss function
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

        iteration += 1 # Increment the iteration counter

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
        lr = poly_lr_scheduler(optimizer, init_lr=learning_rate, iter=iteration, lr_decay_iter=1, max_iter=max_iter, power=0.9)

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

    return metrics, iteration


def train_with_adversary(epoch, old_model, discriminators, dataloader_source_train, dataloader_target_train, 
                         criterion, optimizer, discriminator_optimizers, iteration, learning_rate, num_classes, max_iter, lambdas): # criterion == loss function
   
    # --------------------------- BASIC DEFINITIONS -------------------------------------- #
    try:
        var_model = os.environ['MODEL'] 
    except KeyError:
        print("Environment variable 'MODEL' not set. Using default model 'BiSeNet'.")
        var_model = "BiSeNet"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = old_model.to(device) 
    running_loss = 0.0 
    mean_loss = 0.0
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)
    target_label = 0
    source_label = 1
    target_iter = iter(dataloader_target_train) # Create an iterator for the target dataset


    model.train() 
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
        for discriminator in discriminators:
            lock_model(discriminator.to(device)) # Lock the discriminator parameters to avoid training them


        iteration += 1 # Increment the iteration counter
        inputs_src, targets_src = inputs_src.to(device), targets_src.to(device) # GPU

        # Compute output of the train
        outputs = model(inputs_src)        
        loss = criterion(outputs[0], targets_src)
        
        alpha = 1 # In the paper they use 1
        loss +=  alpha * criterion(outputs[1], targets_src) + alpha *  criterion(outputs[2], targets_src)

        # Get the next batch from the target dataset   
        try:
            inputs_target, _, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(dataloader_target_train)
            inputs_target, _, _ = next(target_iter)
        inputs_target = inputs_target.to(device)

        # Compute the output of the target dataset
        outputs_target = model(inputs_target)

        # Compute the adversarial loss
        adv_loss = adversarial_loss(discriminators, outputs_target, target_label, source_label, device, lambdas)
        # Combine the losses
        loss += adv_loss

        # Backpropagation
        optimizer = backpropagate(optimizer, loss)

        # Compute the learning rate
        lr = poly_lr_scheduler(optimizer, init_lr=learning_rate, iter=iteration, lr_decay_iter=1, max_iter=max_iter, power=0.9)

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
        
        softmax_src = softmax(outputs_source[0], dim=1).detach()
        softmax_tgt = softmax(outputs_target[0], dim=1).detach()

        for discriminator, disc_optimizer in zip(discriminators, discriminator_optimizers):
            disc_optimizer.zero_grad()

            # SOURCE: discriminator deve dire "source_label"
            disc_pred_src = discriminator(softmax_src)
            loss_d_src = torch.nn.functional.binary_cross_entropy_with_logits(
                disc_pred_src,
                torch.full(disc_pred_src.shape, float(source_label), device=device)
            )

            # TARGET: discriminator deve dire "target_label"
            disc_pred_tgt = discriminator(softmax_tgt)
            loss_d_tgt = torch.nn.functional.binary_cross_entropy_with_logits(
                disc_pred_tgt,
                torch.full(disc_pred_tgt.shape, float(target_label), device=device)
            )

            # Media delle due loss
            loss_d = 0.5 * (loss_d_src + loss_d_tgt)
            loss_d.backward()
            disc_optimizer.step()

        unlock_model(model) # Unlock the model parameters to allow training
        discriminator_end = time.time()

        print(f"Discriminator training time: {discriminator_end - discriminator_start:.2f} seconds")
        discriminator_accumulator += (discriminator_end - discriminator_start)


        start_statistics = time.time()
        preds = outputs[0].argmax(dim=1).detach().cpu().numpy()
        gts = targets_src.detach().cpu().numpy()

        # Accumulate intersections and unions per class
        _, _, inters, unions = compute_miou(gts, preds, num_classes)
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

    start_metrics = time.time()

    iou_per_class = (total_intersections / (total_unions + 1e-10)) * 100
    iou_non_zero = np.array(iou_per_class)
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

    end_metrics = time.time()
    print(f"Metrics computation time: {end_metrics - start_metrics:.2f} seconds")
    print(f"Total BiSeNet training time: {bisenet_accumulator:.2f} seconds")
    print(f"Total Discriminator training time: {discriminator_accumulator:.2f} seconds")
    print(f"Total Statistics computation time: {statistics_accumulator:.2f} seconds")
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    return metrics, iteration