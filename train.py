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
from utils.utils import poly_lr_scheduler, poly_lr_scheduler_warmup
import wandb

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
        #lr = poly_lr_scheduler(optimizer, init_lr=learning_rate, iter=iter, lr_decay_iter=1, max_iter=max_iter, power=1.1)
        lr = poly_lr_scheduler_warmup(optimizer, base_lr=learning_rate, curr_iter=iter, max_iter=max_iter, power=0.9,  warmup_iters=2500, warmup_start_lr=1e-6)

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

    return metrics, iter, lr
