import os
import torch
import wandb
import gdown
from models.deeplabv2.deeplabv2 import get_deeplab_v2
from models.bisenet.build_bisenet import BiSeNet
from models.pidnet.PIDNET import PIDNet, get_seg_model
from datasets.transform_datasets import *
from data.dataloader import dataloader
from datasets.cityscapes import CityScapes
from utils.utils import CombinedLoss_All, save_metrics_on_wandb
from validation import validate, validate_pidnet
import random
import numpy as np
import time

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

if __name__ == "__main__":
    # MODEL = 'DeepLabV2' or 'BiSeNet'
    var_model = os.environ['MODEL']
    start_epoch = 38 # Cambia l'epoca desiderata qui

    set_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Define transformations")
    transform = transform_cityscapes()
    target_transform = transform_cityscapes_mask()

    print("Load datasets")
    cs_val = CityScapes('/kaggle/input/cityscapes-dataset/Cityscapes', 'val', transform, target_transform)

    # Parametri modello
    num_classes = 19
    ignore_index = 255
    batch_size = 4  # può cambiare se usi DeepLabV2

    print("Dataloader")
    _, dataloader_cs_val = dataloader(None, cs_val, batch_size, False, True)
    
    if var_model == 'DeepLabV2':
        print("MODEL DEEPLABV2")
        pretrain_model_path = "./pretrained/deeplabv2_cityscapes.pth"
        if not os.path.exists(pretrain_model_path):
            os.makedirs(os.path.dirname(pretrain_model_path), exist_ok=True)
            print("Download pretrained model")
            url = "https://drive.google.com/uc?id=1HZV8-OeMZ9vrWL0LR92D9816NSyOO8Nx"
            gdown.download(url, pretrain_model_path, quiet=False)
        model = get_deeplab_v2(num_classes=num_classes, pretrain=True, pretrain_model_path=pretrain_model_path)

    elif var_model == 'BiSeNet':
        print("MODEL BISENET")
        model = BiSeNet(num_classes=num_classes, context_path='resnet18')
        
    """
    class CFG:
        pass

    cfg = CFG()
    cfg.MODEL = type('', (), {})()
    cfg.DATASET = type('', (), {})()

    cfg.MODEL.NAME = 'pidnet_m'
    cfg.MODEL.PRETRAINED = '/kaggle/input/pidnet-m/PIDNet_M_ImageNet.pth.tar'
    cfg.DATASET.NUM_CLASSES = 19

    # PIDNet-M parameters from the paper/repo
    model = PIDNet(
        m=2,                  # num PID branches (usually fixed)
        n=3,                  # number of output feature maps for I, P, D branches
        num_classes=19,       # Cityscapes: 19 classes
        planes=64,            # plane size for PIDNet-M
        ppm_planes=96,        # Pyramid Pooling Module planes
        head_planes=128,      # final head classifier planes
        augment=True        # True during training, False for inference
    )
"""
    model = model.to(device)


    # LOSS
    #anche se non è quella giusta qua non importa tanto le run le avevo gia fatte, mis ervono solo le foto ora
    loss = CombinedLoss_All(num_classes=num_classes, alpha=1.0, beta=0, gamma=0, theta=0, ignore_index=ignore_index)

    # wandb settings
    entity = "s328422-politecnico-di-torino"
    project_name = "3a_GTA5_to_CITY_ce_batch_update"
    wandb.init(project=project_name, entity=entity, name=f"val_epoch_{start_epoch}", reinit=True)
    
    print("Download model checkpoint from wandb")
    artifact = wandb.use_artifact(f"{entity}/{project_name}/model_epoch_{start_epoch}:latest", type="model")
    artifact_dir = artifact.download()
    checkpoint_path = os.path.join(artifact_dir, f"model_epoch_{start_epoch}.pt")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    print("Run validation")
    start_val = time.time()
    metrics_val = validate(start_epoch, model, dataloader_cs_val, loss, num_classes)
    end_val = time.time()

    print(f"Validation time: {(end_val - start_val)/60:.2f} minutes")
    print("Validation results:", metrics_val)

    save_metrics_on_wandb(start_epoch, {}, metrics_val)

    wandb.finish()
