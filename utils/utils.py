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

def poly_lr_scheduler_warmup(optimizer, base_lr, curr_iter, max_iter, power=0.9, warmup_iters=500, warmup_start_lr=1e-6):
    """Polynomial decay with warmup"""
    if curr_iter < warmup_iters:
        # Linear warmup
        lr = warmup_start_lr + (base_lr - warmup_start_lr) * (curr_iter / warmup_iters)
    else:
        # Poly decay
        decay_iter = curr_iter - warmup_iters
        total_decay = max_iter - warmup_iters
        lr = base_lr * (1 - decay_iter / total_decay) ** power

    optimizer.param_groups[0]['lr'] = lr
    return lr



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

# To avoid void class on TverskyLoss
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

# To avoid void class in DiceLoss
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
    def __init__(self, num_classes, alpha=0.4, beta=0.1, gamma=0.4, theta=0.1, delta=0.0, focal_gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.theta, self.delta = alpha, beta, gamma, theta, delta
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        if alpha != 0:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        if beta != 0:
            from utils.lovasz_losses import lovasz_softmax
            self.lovasz_softmax = lovasz_softmax

        if gamma != 0:
            self.tversky_loss = MaskedTverskyLoss(num_classes, ignore_index)

        if theta != 0:
            self.dice_loss = MaskedDiceLoss(num_classes, ignore_index)

        if delta != 0:
            self.focal_loss = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index)

    def forward(self, outputs, targets):
        total_loss = 0.0

        if self.alpha != 0:
            total_loss += self.alpha * self.ce_loss(outputs, targets)

        if self.beta != 0:
            probs = torch.softmax(outputs, dim=1)
            total_loss += self.beta * self.lovasz_softmax(probs, targets, ignore=self.ignore_index)

        if self.gamma != 0:
            total_loss += self.gamma * self.tversky_loss(outputs, targets)

        if self.theta != 0:
            total_loss += self.theta * self.dice_loss(outputs, targets)

        if self.delta != 0:
            total_loss += self.delta * self.focal_loss(outputs, targets)

        return total_loss

    def __repr__(self):
        return (f"{self.__class__.__name__}(num_classes={self.num_classes}, "
                f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, "
                f"theta={self.theta}, delta={self.delta})")
"""
                
class CombinedLoss_All(nn.Module):
    def __init__(self, num_classes, alpha=0.4, beta=0.1, gamma=0.4, theta=0.1, delta=0.0, focal_gamma=2.0, ignore_index=255, class_weights=None):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.theta, self.delta = alpha, beta, gamma, theta, delta
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.class_weights = class_weights

        if alpha != 0:
            if class_weights is not None:
                self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
            else:
                self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        if beta != 0:
            from utils.lovasz_losses import lovasz_softmax
            self.lovasz_softmax = lovasz_softmax

        if gamma != 0:
            self.tversky_loss = MaskedTverskyLoss(num_classes, ignore_index)

        if theta != 0:
            self.dice_loss = MaskedDiceLoss(num_classes, ignore_index)

        if delta != 0:
            self.focal_loss = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index)

    def forward(self, outputs, targets):
        total_loss = 0.0

        if self.alpha != 0:
            total_loss += self.alpha * self.ce_loss(outputs, targets)

        if self.beta != 0:
            probs = torch.softmax(outputs, dim=1)
            total_loss += self.beta * self.lovasz_softmax(probs, targets, ignore=self.ignore_index)

        if self.gamma != 0:
            total_loss += self.gamma * self.tversky_loss(outputs, targets)

        if self.theta != 0:
            total_loss += self.theta * self.dice_loss(outputs, targets)

        if self.delta != 0:
            total_loss += self.delta * self.focal_loss(outputs, targets)

        return total_loss

    def __repr__(self):
        return (f"{self.__class__.__name__}(num_classes={self.num_classes}, "
                f"alpha={self.alpha}, beta={self.beta}, "
                f"gamma={self.gamma}, theta={self.theta})")
