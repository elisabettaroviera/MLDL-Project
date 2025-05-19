import numpy as np
from utils.lovasz_losses import lovasz_softmax
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.losses import TverskyLoss


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

# Function to save the metrics on WandB UPDATED 
# In this function, we log the metrics for each epoch
# and also log additional metrics at the end of the training (epoch 50)   
def save_metrics_on_wandb(epoch, metrics_train, metrics_val, final_epoch=50):
    to_serialize = {"epoch": epoch}

    # Log training metrics
    if metrics_train is not None:
        to_serialize.update({
            "train_mIoU": metrics_train['mean_iou'],
            "train_loss": metrics_train['mean_loss']
        })

        for index, iou in enumerate(metrics_train['iou_per_class']):
            to_serialize[f"class_{index}_train"] = iou

        if epoch == final_epoch:
            to_serialize.update({
                "train_latency": metrics_train['mean_latency'],
                "train_fps": metrics_train['mean_fps'],
                "train_flops": metrics_train['num_flops'],
                "train_params": metrics_train['trainable_params']
            })

    # Log validation metrics
    if metrics_val is not None:
        to_serialize.update({
            "val_mIoU": metrics_val['mean_iou'],
            "val_mIoU_per_class": metrics_val['iou_per_class'],
            "val_loss": metrics_val['mean_loss']
        })

        for index, iou in enumerate(metrics_val['iou_per_class']):
            to_serialize[f"class_{index}_val"] = iou

        if epoch == final_epoch:
            to_serialize.update({
                "val_latency": metrics_val['mean_latency'],
                "val_fps": metrics_val['mean_fps'],
                "val_flops": metrics_val['num_flops'],
                "val_params": metrics_val['trainable_params']
            })

    # Logging finale su wandb
    wandb.log(to_serialize)



class MaskedTverskyLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, beta=0.5, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, softmax=True)

    def forward(self, pred, target):
        valid_mask = (target != self.ignore_index)
        target_clean = target.clone()
        target_clean[~valid_mask] = 0

        target_onehot = F.one_hot(target_clean, num_classes=self.num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1).float()
        pred = pred * valid_mask
        target_onehot = target_onehot * valid_mask

        return self.tversky(pred, target_onehot)

# To avoid void class in dice loss
class MaskedDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice = DiceLoss(include_background=False, softmax=True, reduction="mean")

    def forward(self, pred, target):
        # Mask void pixels
        mask = (target != self.ignore_index).float()
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0

        # One-hot encode
        target_one_hot = F.one_hot(target_clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Apply mask
        mask = mask.unsqueeze(1)
        pred_masked = pred * mask
        target_masked = target_one_hot * mask

        return self.dice(pred_masked, target_masked)
    

class CombinedLoss_All(nn.Module):
    def __init__(self, num_classes, 
                 alpha=0.4,   # CrossEntropy
                 beta=0.1,    # Lovász
                 gamma=0.4,   # Tversky
                 theta=0.1,   # Dice
                 ignore_index=255):
        super(CombinedLoss_All, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.tversky_loss = MaskedTverskyLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.dice_loss = MaskedDiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, outputs, targets):
        ce = self.ce_loss(outputs, targets)

        probs = torch.softmax(outputs, dim=1)
        lovasz = lovasz_softmax(probs, targets, ignore=self.ignore_index)

        tversky = self.tversky_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)

        total_loss = (self.alpha * ce +
                      self.beta * lovasz +
                      self.gamma * tversky +
                      self.theta * dice)
        return total_loss

    def __repr__(self):
        return (f"{self.__class__.__name__}(num_classes={self.num_classes}, "
                f"alpha={self.alpha}, beta={self.beta}, "
                f"gamma={self.gamma}, theta={self.theta})")