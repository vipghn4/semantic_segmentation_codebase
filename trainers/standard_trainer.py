import os
import glob
import importlib
import json
import math
from easydict import EasyDict
from tqdm import tqdm
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from datasets.standard_dataset import StandardDataset
from augmentators import DeeplabV3Augmentator
from optimizers import get_quick_optimizer, SlowStartDeeplabV3Scheduler
from metrics import CELoss, pixel_acc, IoU, DICE
from trainers.utils import AverageMeter, Logger, bcolors, logits_to_onehot
from misc.voc2012_color_map import get_color_map

COLOR_MAP = get_color_map()


class StandardTrainer:
    def __init__(self, train_config):
        r"""Standard Trainer class for training a segmentation model with VOC-2012 dataset.
        
        Args:
            train_config (EasyDict): Contain training configuration of VOC-2012 trainer.
                * model (torch.nn.Module): Segmentation model to train.
                * loss_func (__callable__): Loss function used for training. The loss function must take ypred and ytrue, which are torch.FloatTensors and returns a torch.FloatTensor.
                * metric_funcs (dict): Metric functions used for evaluation. The format should be
                
                ```
                {
                    "metric_name": callable object,
                    ...
                }
                ```
                
                Each callable object must take ypred and ytrue, which are torch.FloatTensors and returns a torch.FloatTensor.

                * base_lr (float): Base learning rate for optimization.
                * slow_start_lr (float): Base learning rate for slow-start training phase.
                * slow_start_step (int): Number of slow-start training steps.
                * n_epochs (int): Number of training epochs.
                * data_config (EasyDict): Contain configuration of VOC-2012 dataset to pass into StandardDataset (see its docstring for more details).
                * batch_size (int): Batch size to be used during training and testing.
                * num_workers (int): Number of workers to be invoked for loading data.
                * device (str): Device dedicated for training, e.g. "cpu", "cuda:0", etc.
                * log_dir (str): Directory to save Tensorboard records.
                * checkpoint_dir (str): Directory to save trained models
        """
        self.config = train_config
        self.model = self.config.model
        self.loss_func = self.config.loss_func
        self.metric_funcs = self.config.metric_funcs
        self.train_dataset, self.train_loader, self.val_dataset, self.val_loader = self.__get_dataloaders()
        self.optimizer, self.scheduler, self.slow_start_scheduler = self.__get_optimizer()
        self.logger = Logger(self.config.log_dir)

    def train(self):
        r"""Train a deep segmentation model"""
        self.best_criteria, self.best_epoch = float('-inf'), -1
        self.last_criteria, self.last_epoch = float('-inf'), -1
        
        self.global_training_step = 0
        for epoch in range(self.config.n_epochs):
            self.train_avg_meters, self.val_avg_meters = self.__get_average_meters()
            self.__train_one_epoch(epoch)
            self.__eval_one_epoch(epoch)
            self.__record_results(epoch)

    def __get_average_meters(self):
        r"""Get AverageMeter(s) for training and evaluation"""
        train_avg_meters = {"loss": AverageMeter()}
        train_avg_meters.update({metric: AverageMeter() for metric in self.metric_funcs})
        train_avg_meters = EasyDict(train_avg_meters)
        
        val_avg_meters = {"loss": AverageMeter()}
        val_avg_meters.update({metric: AverageMeter() for metric in self.metric_funcs})
        val_avg_meters = EasyDict(val_avg_meters)
        
        return train_avg_meters, val_avg_meters

    def __get_optimizer(self):
        r"""Get optimizer for training segmentation model"""
        max_iter = self.config.n_epochs * math.ceil(len(self.train_loader) / self.config.batch_size)
        optimizer, scheduler = get_quick_optimizer(model, max_iter, base_lr=self.config.base_lr)
        slow_start_scheduler = SlowStartDeeplabV3Scheduler(
            optimizer, self.config.base_lr,
            self.config.slow_start_lr, self.config.slow_start_step
        )
        return optimizer, scheduler, slow_start_scheduler

    def __get_dataloaders(self):
        r"""Get DataLoader for training and validation sets"""
        train_dataset = StandardDataset(self.config.data_config, split="train")
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        val_dataset = StandardDataset(self.config.data_config, split="val")
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        return train_dataset, train_loader, val_dataset, val_loader

    def __train_one_epoch(self, epoch):
        r"""Train single epoch"""
        self.model.train()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for step, batch in progress_bar:
            images = batch["image"].to(self.config.device)
            onehot_masks = batch["onehot_mask"].to(self.config.device)
            
            logits = self.model(images)
            loss = self.loss_func(logits, onehot_masks)
            
            onehot_pred_masks = logits_to_onehot(logits)
            metrics = {metric: self.metric_funcs[metric](onehot_pred_masks, onehot_masks) for metric in self.metric_funcs}
            metrics = self.__parse_metrics(loss, metrics)
            
            self.__update_model_params(loss)
            self.__update_average_meters(self.train_avg_meters, metrics)
            self.__display_progress_bar(progress_bar, "Train", epoch, step, self.train_avg_meters)
            self.global_training_step += 1
        # self.logger.list_of_scalars_summary([], epoch) # TODO
    
    def __update_model_params(self, loss):
        r"""Run gradient descent update and learning rate update"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.__update_learning_rate()
    
    def __update_learning_rate(self):
        r"""Adjust learning rate"""
        if self.global_training_step <= self.config.slow_start_step:
            self.slow_start_scheduler.step()
        else:
            self.scheduler.step()

    def __eval_one_epoch(self, epoch):
        r"""Evaluate single epoch"""
        self.model.eval()
        progress_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        for step, batch in progress_bar:
            images = batch["image"].to(self.config.device)
            onehot_masks = batch["onehot_mask"].to(self.config.device)
            
            with torch.no_grad():
                logits = self.model(images)
                loss = self.loss_func(logits, onehot_masks)
                
                onehot_pred_masks = logits_to_onehot(logits)
                metrics = {metric: self.metric_funcs[metric](onehot_pred_masks, onehot_masks) for metric in self.metric_funcs}
            
            metrics = self.__parse_metrics(loss, metrics)
            self.__update_average_meters(self.val_avg_meters, metrics)
            self.__display_progress_bar(progress_bar, "Val", epoch, step, self.val_avg_meters)
        # self.logger.list_of_scalars_summary([], epoch) # TODO
    
    def __record_results(self, epoch):
        r"""Record training results"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        state_dict = {
            "model": self.model.state_dict(),
            "epoch": epoch,
            "val_loss": self.val_avg_meters.loss.average(),
            "val_iou": self.val_avg_meters.iou.average(),
            "train_loss": self.train_avg_meters.loss.average(),
            "train_iou": self.train_avg_meters.iou.average()
        }
        if self.best_criteria < self.val_avg_meters.iou.average():
            path = self.__get_model_path(epoch, self.val_avg_meters.iou.average(), best=True)
            print(bcolors.log(f"epoch {epoch} improved from {self.best_criteria:.4f} to {self.val_avg_meters.iou.average():.4f} file {path}"))
            torch.save(state_dict, path)
            if self.best_epoch >= 0:
                path = self.__get_model_path(self.best_epoch, self.best_criteria, best=True)
                os.remove(path)
            self.best_criteria, self.best_epoch = self.val_avg_meters.iou.average(), epoch
        else:
            path = self.__get_model_path(epoch, self.val_avg_meters.iou.average())
            torch.save(state_dict, path)
            if self.last_epoch >= 0:
                path = self.__get_model_path(self.last_epoch, self.last_criteria)
                os.remove(path)
            self.last_criteria, self.last_epoch = self.val_avg_meters.iou.average(), epoch
    
    def __get_model_path(self, epoch, val_iou, best=False):
        r'''return model path with epoch number and validation iou'''
        return os.path.join(self.config.checkpoint_dir, f"{epoch:04d}_{val_iou:.4f}" + ("_best" if best else "") + ".pth")
    
    def __parse_metrics(self, loss, metrics):
        r"""Parse loss and metrics to a dict for display purpose"""
        metrics = {metric: metrics[metric].item() for metric in metrics}
        metrics["loss"] = loss.item()
        return metrics
    
    def __update_average_meters(self, meter, metrics):
        r"""Update AverageMeter"""
        for metric in metrics:
            meter[metric].update(metrics[metric])

    def __display_progress_bar(self, progress_bar, phase, epoch, step, meter):
        r"""Display progress bar"""
        desc = f"{phase} epoch {epoch}/{self.config.n_epochs} step {step}: "
        for metric in meter:
            avg_metric = meter[metric].average()
            desc += f"{metric}: {avg_metric:.4f} "
        progress_bar.set_description(desc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_module", type=str, required=True, help="Model module / package name")
    parser.add_argument("--model_config_file", type=str, required=True, help="JSON file containing model configuration")
    parser.add_argument("--data_root", type=str, required=True, help="The directory containing `VOCdevkit` directory")
    parser.add_argument("--label_map_file", type=str, required=True, help="The label map file for VOC-2012 dataset")
    parser.add_argument("--log_dir", type=str, default="tmp/log", help="Log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="tmp/checkpoints", help="Checkpoint directory to save models")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of train epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size used for training and evaluation")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of threads used for batch generation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device used for training and inference")
    args = parser.parse_args()
    print(args)
    
    with open(args.model_config_file) as f:
        model_config = json.load(f)
    module = importlib.import_module(args.model_module)
    model = getattr(module, "get_model")(model_config).to(args.device)
    
    augmentator = DeeplabV3Augmentator(
        target_size=(512, 512),
        crop_size=(512, 512)
    )
    data_config = EasyDict(dict(
        data_root=args.data_root,
        label_map_file=args.label_map_file,
        augment_data=augmentator,
        preprocess=None,
        target_size=(512, 512),
        ignored_class=21
    ))
    train_config = EasyDict(dict(
        model=model,
        loss_func=CELoss,
        metric_funcs={"pix_acc": pixel_acc, "iou": IoU, "dice": DICE},
        base_lr=1e-3,
        slow_start_lr=5e-5,
        slow_start_step=100,
        n_epochs=args.n_epochs,
        data_config=data_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    ))
    trainer = StandardTrainer(train_config)
    trainer.train()
    print("Done!")