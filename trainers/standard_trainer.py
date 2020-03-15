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

from datasets.standard_dataset import StandardDataset
from optimizers import get_quick_optimizer
from metrics import IoU, DICE
from trainers.utils import Logger, bcolors, logits_to_onehot


class StandardTrainer:
    def __init__(self, train_config):
        r"""Standard Trainer class for training a segmentation model with VOC-2012 dataset.
        
        Args:
            train_config (EasyDict): Contain training configuration of VOC-2012 trainer.
                * model (torch.nn.Module): Segmentation model to train.
                * loss_func (torch.nn.Module): Loss function used for training. The loss function must take ypred and ytrue, which are torch.FloatTensors and returns a torch.FloatTensor.
                * metric_funcs (dict): Metric functions used for evaluation. The format should be
                
                ```
                {
                    "metric_name": callable object,
                    ...
                }
                ```
                
                Each callable object must take ypred and ytrue, which are torch.FloatTensors and returns a torch.FloatTensor.

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
        self.optimizer, self.scheduler = self.__get_optimizer()
        self.logger = Logger(self.config.log_dir)

    def train(self):
        r"""Train a deep segmentation model"""
        self.best_criteria, self.best_epoch = float('-inf'), -1
        self.last_criteria, self.last_epoch = float('-inf'), -1
        for epoch in range(self.config.n_epochs):
            train_metrics = self.__train_one_epoch(epoch)
            val_metrics = self.__eval_one_epoch(epoch)
            self.__record_results(epoch, train_metrics, val_metrics)

    def __get_optimizer(self):
        r"""Get optimizer for training segmentation model"""
        max_iter = self.config.n_epochs * math.ceil(len(self.train_dataset) / self.config.batch_size)
        optimizer, scheduler = get_quick_optimizer(model, max_iter)
        return optimizer, scheduler

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
        metric_progress = []
        for step, batch in progress_bar:
            images = batch["image"].to(self.config.device)
            masks, onehot_masks = batch["mask"].to(self.config.device), batch["onehot_mask"].to(self.config.device)
            
            logits = self.model(images)
            loss = self.loss_func(logits, masks)
            
            onehot_pred_masks = logits_to_onehot(logits)
            metrics = {metric: self.metric_funcs[metric](onehot_pred_masks, masks, onehot_masks) for metric in self.metric_funcs}
            self.__update_model_params(loss)

            metric_progress.append(self.__parse_metrics(loss, metrics))
            metrics = self.__get_mean(metric_progress)
            self.__display_progress_bar(progress_bar, "Train", epoch, step, metrics)
        self.logger.list_of_scalars_summary([], epoch) # TODO
        return metrics
    
    def __update_model_params(self, loss):
        r"""Run gradient descent update and learning rate update"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def __eval_one_epoch(self, epoch):
        r"""Evaluate single epoch"""
        self.model.eval()
        progress_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
        metric_progress = []
        for step, batch in progress_bar:
            images = batch["image"].to(self.config.device)
            masks, onehot_masks = batch["mask"].to(self.config.device), batch["onehot_mask"].to(self.config.device)
            
            with torch.no_grad():
                logits = self.model(images)
                loss = self.loss_func(logits, masks)
                
                onehot_pred_masks = logits_to_onehot(logits)
                metrics = {metric: self.metric_funcs[metric](onehot_pred_masks, masks, onehot_masks) for metric in self.metric_funcs}
            
            metric_progress.append(self.__parse_metrics(loss, metrics))
            metrics = self.__get_mean(metric_progress)
            self.__display_progress_bar(progress_bar, "Val", epoch, step, metrics)
        self.logger.list_of_scalars_summary([], epoch) # TODO
        return metrics
    
    def __record_results(self, epoch, train_metrics, val_metrics):
        r"""Record training results"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        state_dict = {
            "model": self.model.state_dict(),
            "epoch": epoch,
            "val_loss": val_metrics.loss,
            "val_iou": val_metrics.iou,
            "train_loss": train_metrics.loss,
            "train_iou": train_metrics.iou
        }
        if self.best_criteria < val_metrics.iou:
            path = self.__get_model_path(epoch, val_metrics.iou, best=True)
            print(bcolors.log(f"epoch {epoch} improved from {self.best_criteria:.4f} to {val_metrics.iou:.4f} file {path}"))
            torch.save(state_dict, path)
            if self.best_epoch >= 0:
                path = self.__get_model_path(self.best_epoch, self.best_criteria, best=True)
                os.remove(path)
            self.best_criteria, self.best_epoch = val_metrics.iou, epoch
        else:
            path = self.__get_model_path(epoch, val_metrics.iou)
            torch.save(state_dict, path)
            if self.last_epoch >= 0:
                path = self.__get_model_path(self.last_epoch, self.last_criteria)
                os.remove(path)
            self.last_criteria, self.last_epoch = val_metrics.iou, epoch
    
    def __get_model_path(self, epoch, val_iou, best=False):
        r'''return model path with epoch number and validation iou'''
        return os.path.join(self.config.checkpoint_dir, f"{epoch:04d}_{val_iou:.4f}" + ("_best" if best else "") + ".pth")

    def __get_mean(self, metric_progress):
        r'''compute the average of metric_progress in a list of EasyDict(...) '''
        return EasyDict({key : np.mean([x[key] for x in metric_progress]) for key in metric_progress[0]})
    
    def __parse_metrics(self, loss, metrics):
        r"""Parse loss and metrics to a dict for display purpose"""
        metrics = {metric: metrics[metric].item() for metric in metrics}
        metrics["loss"] = loss.item()
        return metrics

    def __display_progress_bar(self, progress_bar, phase, epoch, step, metrics):
        r"""Display progress bar"""
        desc = f"{phase} epoch {epoch}/{self.config.n_epochs} step {step}: "
        for metric in metrics:
            metric_value = metrics[metric]
            desc += f"{metric}: {metric_value} "
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
    
    data_config = EasyDict(dict(
        data_root=args.data_root,
        label_map_file=args.label_map_file,
        augment_data=None,
        preprocess=None,
        target_size=(512, 512)
    ))
    train_config = EasyDict(dict(
        model=model,
        loss_func=nn.CrossEntropyLoss(ignore_index=21),
        metric_funcs={"iou": IoU, "dice": DICE},
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