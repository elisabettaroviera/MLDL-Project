import numpy as np
import wandb
from lovasz_losses import lovasz_softmax  # file taken from github
import torch
import torch.nn as nn


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    
    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = float(lr) 
    return float(lr) 

    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer
    # return lr



def fast_hist(a, b, n):
    '''
    a and b are label and prediction respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


# We implement our own function to compute the mean IoU
def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

def save_metrics_on_file(epoch, metrics_train, metrics_val):
    open_mode = "w" if epoch == 1 else "a"

    with open("IMoU.txt", open_mode) as imou_file:
        imou_file.write(f"""Epoch - {epoch}
    ---------------
    Training Phase
    mIoU: {metrics_train['mean_iou']}
    mIoU per Class: {metrics_train['iou_per_class']}
    ---------------
    Validation Phase
    mIoU: {metrics_val['mean_iou']}
    mIoU per Class: {metrics_val['iou_per_class']}
    ===============

    """)

    with open("Loss.txt", open_mode) as loss_file:
        loss_file.write(f"""Epoch - {epoch}
    ---------------
    Training Phase
    Value Loss: {metrics_train['mean_loss']}
    ---------------
    Validation Phase
    Value Loss: {metrics_val['mean_loss']}
    ===============

    """)

    if epoch == 50:
        with open("Final_Metrics.txt", "w") as metrics_file:
            metrics_file.write(f"""Epoch - {epoch}
    ---------------
    Training Phase
    mIoU: {metrics_train['mean_iou']}
    mIoU per Class: {metrics_train['iou_per_class']}
    Loss: {metrics_train['mean_loss']}
    Latency: {metrics_train['mean_latency']} ± {metrics_train['std_latency']}
    FPS: {metrics_train['mean_fps']} ± {metrics_train['std_fps']}
    FLOPs: {metrics_train['num_flops']}
    Trainable Params: {metrics_train['trainable_params']}
    ---------------
    Validation Phase
    mIoU: {metrics_val['mean_iou']}
    mIoU per Class: {metrics_val['iou_per_class']}
    Loss: {metrics_val['mean_loss']}
    Latency: {metrics_val['mean_latency']} ± {metrics_val['std_latency']}
    FPS: {metrics_val['mean_fps']} ± {metrics_val['std_fps']}
    FLOPs: {metrics_val['num_flops']}
    Trainable Params: {metrics_val['trainable_params']}
    ===============

    """)

 #Function to save the metrics on WandB           
def save_metrics_on_wandb(epoch, metrics_train, metrics_val):
    # Log delle metriche di training e validazione su WandB
    wandb.log({
        "epoch": epoch,
        "train_mIoU": metrics_train['mean_iou'],
        "train_mIoU_per_class": metrics_train['iou_per_class'],
        "train_loss": metrics_train['mean_loss'],
        "val_mIoU": metrics_val['mean_iou'],
        "val_mIoU_per_class": metrics_val['iou_per_class'],
        "val_loss": metrics_val['mean_loss']
    })

    # Salvataggio delle metriche finali al 50esimo epoch
    if epoch == 50:
        wandb.log({
            "train_mIoU_final": metrics_train['mean_iou'],
            "train_loss_final": metrics_train['mean_loss'],
            "train_latency": metrics_train['mean_latency'],
            "train_fps": metrics_train['mean_fps'],
            "train_flops": metrics_train['num_flops'],
            "train_params": metrics_train['trainable_params'],
            "val_mIoU_final": metrics_val['mean_iou'],
            "val_loss_final": metrics_val['mean_loss'],
            "val_latency": metrics_val['mean_latency'],
            "val_fps": metrics_val['mean_fps'],
            "val_flops": metrics_val['num_flops'],
            "val_params": metrics_val['trainable_params']
        })

    # Class to compute the combined loss: alpha*cross entropy + beta*lovasz
    class CombinedLoss_Lovasz(nn.Module):
        def __init__(self, alpha=0.5, beta=0.5, ignore_index=255):
            super(CombinedLoss_Lovasz, self).__init__()
            self.alpha = alpha
            self.beta = beta
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self.ignore_index = ignore_index

        def forward(self, outputs, targets):
            ce = self.ce_loss(outputs, targets)
            probs = torch.softmax(outputs, dim=1)
            lovasz = lovasz_softmax(probs, targets, ignore=self.ignore_index)
            return self.alpha * ce + self.beta * lovasz
