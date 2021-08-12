import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from datetime import datetime
import pickle

from config import TrainConfig, GPU_ID

from torchvision.transforms import Compose
from data_utils import stratified_train_val_split, PotholesDataset

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader


# Ignore pytorch warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

def train():
    params = TrainConfig().config

    wandb.init(project='mycrops_hw',
               entity='tomron27',
               job_type="eval",
               reinit=True,
               config=params,
               notes=params["name"])

    metadata, train_images, val_images = stratified_train_val_split(params['potholes_data_path'],
                                                                    params['pothole_json_path'])
    # Datasets
    train_dataset = PotholesDataset(params['potholes_data_path'], train_images, metadata)
    val_dataset = PotholesDataset(params['potholes_data_path'], val_images, metadata)

    # CUDA
    device = torch.device("cuda" if torch.cuda.is_available() and params["use_gpu"] else "cpu")

    # Dataloaders
    def collate_fn(batch):
        return tuple(zip(*batch))
    train_loader = DataLoader(dataset=train_dataset,
                              num_workers=params["num_workers"],
                              pin_memory=True,
                              batch_size=params["batch_size"],
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset,
                            num_workers=params["num_workers"],
                            pin_memory=True,
                            batch_size=params["batch_size"],
                            shuffle=False,
                            collate_fn=collate_fn)

    # Model
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=params['coco_pretrained'],
                                              progress=True,
                                              pretrained_backbone=params['imagenet_pretrained'])
    # Manually replace the linear layer for our number of classes
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(1024, params['num_classes'])

    # Create log dir
    log_dir = os.path.join(params["log_path"], params["name"], datetime.now().strftime("%Y%m%d_%H:%M:%S"))
    os.makedirs(log_dir)
    print("Log dir: '{}' created".format(log_dir))
    pickle.dump(params, open(os.path.join(log_dir, "params.p"), "wb"))

    model = model.to(device)

    # Training
    best_val_score = 0.0
    save_dir = os.path.join(params["log_path"], "val_results")
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(params["num_epochs"]):
        train_stats, val_stats = {}, {}
        for fold in ['train', 'val']:
            print(f"*** Epoch {epoch + 1} {fold} fold ***")
            if fold == "train":
                model.train()
                for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
                    images, bboxes, labels = sample
                    images = [x.to(device, non_blocking=True) for x in images]
                    bboxes = [x.to(device, non_blocking=True) for x in bboxes]
                    labels = [x.to(device, non_blocking=True) for x in labels]
                    targets = []
                    for i in range(len(images)):
                        targets.append({'boxes': bboxes[i], 'labels': labels[i]})
                    # Faster-RCNN implements it's own forward / backward as it has multiple modules (RPN, R-CNN)
                    output = model(images, targets)
            else:
                model.eval()
                for i, sample in tqdm(enumerate(train_loader), total=len(val_loader)):
                    images, bboxes, labels = sample
                    images = [x.to(device, non_blocking=True) for x in images]
                    bboxes = [x.to(device, non_blocking=True) for x in bboxes]
                    labels = [x.to(device, non_blocking=True) for x in labels]
                    targets = []
                    for i in range(len(images)):
                        targets.append({'boxes': bboxes[i], 'labels': labels[i]})
                    output = model(images, targets)
                    pass
                #         losses = criterion(outputs, targets, marginals)
                #         current_lr = optimizer.param_groups[0]['lr'] if scheduler is not None else params["lr"]
                #         log_stats_classification(val_stats, outputs, targets, losses, batch_size=params["batch_size"],
                #                              lr=current_lr)
                # val_loss, val_score = write_stats_classification(train_stats, val_stats, epoch,
                #                                                  ret_metric=params["save_metric"])

        # # progress LR scheduler
        # if scheduler is not None:
        #     scheduler.step(val_loss)
        #
        # # Save parameters
        # if val_score >= best_val_score and epoch >= params["min_epoch_save"]:
        #     model_file = os.path.join(log_dir,params["name"] + f'__best__epoch={epoch + 1:03d}_score={val_score:.4f}.pt')
        #     print(f'Model improved {params["save_metric"]} from {best_val_score:.4f} to {val_score:.4f}')
        #     print(f'Saving model at \'{model_file}\' ...')
        #     torch.save(model.state_dict(), model_file)
        #     best_val_score = val_score
        #     wandb.run.summary["best_val_score"] = best_val_score
        #
        # if params["chekpoint_save_interval"] > 0:
        #     if epoch % params["chekpoint_save_interval"] == 0 and epoch >= params["min_epoch_save"]:
        #         model_file = os.path.join(log_dir,
        #                                   params["name"] + f'__ckpt__epoch={epoch + 1:03d}_score={val_score:.4f}.pt')
        #         print(f"Saving model at '{model_file}' ...")
        #         torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    train()