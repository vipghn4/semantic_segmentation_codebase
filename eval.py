import os
import importlib
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm
from easydict import EasyDict
import argparse

from datasets.standard_dataset import StandardDataset
from metrics import CELoss, accuracy, pixel_acc, IoU, DICE, intersectionAndUnion
from trainers.utils import AverageMeter, Logger, bcolors, logits_to_onehot
from misc.voc2012_color_map import get_color_map

COLOR_MAP = get_color_map()


class StandardEvaluator:
    def __init__(self, eval_config):
        r"""Standard Evaluator class for evaluating a segmentation model with VOC-2012 dataset.
        
        Args:
            train_config (EasyDict): Contain evaluation configuration of VOC-2012 trainer.
                * model (torch.nn.Module): Segmentation model to evaluate.
                * data_config (EasyDict): Contain configuration of VOC-2012 dataset to pass into StandardDataset (see its docstring for more details).
                * num_workers (int): Number of workers to be invoked for loading data.
                * device (str): Device dedicated for training, e.g. "cpu", "cuda:0", etc.
        """
        self.config = eval_config
        self.model = self.config.model
        self.dataset, self.dataloader = self.__get_dataloaders()
    
    def __get_dataloaders(self):
        r"""Get DataLoader for validation sets"""
        dataset = StandardDataset(self.config.data_config, split="val")
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        return dataset, dataloader

    def evaluate(self):
        r"""Evaluate a model with a specific dataset"""
        avg_meters = self.__get_average_meter()
        self.model.eval()

        progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        for step, batch in progress_bar:
            images = batch["image"].to(self.config.device)
            masks = batch["mask"].to(self.config.device)
            onehot_masks = batch["onehot_mask"].to(self.config.device)
            
            with torch.no_grad():
                logits = self.model(images)
                onehot_pred_masks = logits_to_onehot(logits)

                acc, pix = accuracy(onehot_pred_masks, onehot_masks)
                intersection, union = intersectionAndUnion(onehot_pred_masks, onehot_masks)
            
            self.__update_average_meters(avg_meters, 
                                         acc.cpu().numpy(), pix, 
                                         intersection.cpu().numpy(), 
                                         union.cpu().numpy())
            self.__display_progress_bar(progress_bar, "Val", 1, step, avg_meters)
        
            if self.config.save_dir is not None and step < 30:
                visualize(step, images, logits, masks, acc.cpu().numpy(), self.config.save_dir)

        acc = avg_meters["acc"].average()
        iou = avg_meters["intersection"].sum / (avg_meters["union"].sum + 1e-10)
        for i, _iou in enumerate(iou):
            print(f"class {i}, IoU: {_iou}")
        
        print("Eval summary")
        print(f"\tMean IoU {iou.mean()}")
        print(f"\tAccuracy: {100*acc}%")

    def __get_average_meter(self):
        r"""Get AverageMeter(s) for evaluation"""
        avg_meters = {
            "acc": AverageMeter(),
            "intersection": AverageMeter(), 
            "union": AverageMeter()
        }
        return avg_meters
    
    def __update_average_meters(self, meter, acc, pix, intersection, union):
        r"""Update AverageMeter"""
        meter["acc"].update(acc, pix)
        meter["intersection"].update(intersection)
        meter["union"].update(union)
    
    def __display_progress_bar(self, progress_bar, phase, epoch, step, meter):
        r"""Display progress bar"""
        acc = meter["acc"].average()
        desc = f"{phase} step {step}: acc {acc:.4f}"
        progress_bar.set_description(desc)

def visualize(step, images, logits, masks, iou, save_dir):
    r"""Visualize prediction results"""
    os.makedirs(save_dir, exist_ok=True)
    image = images[0].cpu().numpy()
    image = (image * 255).transpose((1, 2, 0)).astype(np.uint8)
    
    mask = masks[0].cpu().numpy()
    colored_mask = np.zeros_like(image)
    for color in range(21):
        colored_mask[mask == color] = COLOR_MAP[color]

    onehot_pred = logits[0].cpu().numpy()
    onehot_pred = np.argmax(onehot_pred.transpose(1, 2, 0), axis=-1)
    colored_onehot_pred = np.zeros_like(image)
    for color in range(21):
        colored_onehot_pred[onehot_pred == color] = COLOR_MAP[color]
    
    visualized_image = np.concatenate([
        cv2.addWeighted(image, 1, colored_mask, 0.7, 0),
        cv2.addWeighted(image, 1, colored_onehot_pred, 0.7, 0),
    ], axis=1)
    
    cv2.imwrite(os.path.join(save_dir, f"{step}-{iou:.4f}.jpg"), visualized_image)

def colorize_class_from_mask(mask, _class, color):
    r"""Colorize segmentation mask"""
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask == _class] = color
    return colored_mask

def preprocess(image, mask, ignored_class=21):
    r"""Preprocess data for evaluation"""
    h, w, _ = image.shape
    h_new = round2nearest_multiple(h)
    w_new = round2nearest_multiple(w)
    new_image = np.ones((h_new, w_new, 3), dtype=np.uint8)
    new_image[:h, :w] = image
    new_mask = np.ones((h_new, w_new), dtype=np.uint8) * ignored_class
    new_mask[:h, :w] = mask
    return new_image, new_mask

def round2nearest_multiple(x, p=8):
    r"""Round x to the nearest multiple of p and x' >= x"""
    return ((x - 1) // p + 1) * p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_module", type=str, required=True, help="Model module / package name")
    parser.add_argument("--model_config_file", type=str, required=True, help="JSON file containing model configuration")
    parser.add_argument("--model_weights_file", type=str, required=True, help=".pth file containing model weights")
    parser.add_argument("--data_root", type=str, required=True, help="The directory containing `VOCdevkit` directory")
    parser.add_argument("--label_map_file", type=str, required=True, help="The label map file for VOC-2012 dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of threads used for batch generation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device used for training and inference")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save visualization result")
    args = parser.parse_args()
    print(args)
    
    with open(args.model_config_file) as f:
        model_config = json.load(f)
    module = importlib.import_module(args.model_module)
    model = getattr(module, "get_model")(model_config).to(args.device)
    ckpt_data = torch.load(args.model_weights_file)
    model.load_state_dict(ckpt_data["model"])

    data_config = EasyDict(dict(
        data_root=args.data_root,
        label_map_file=args.label_map_file,
        augment_data=None,
        preprocess=preprocess,
        ignored_class=21
    ))
    eval_config = EasyDict(dict(
        model=model,
        data_config=data_config,
        num_workers=args.num_workers,
        device=args.device,
        save_dir=args.save_dir
    ))
    evaluator = StandardEvaluator(eval_config)
    evaluator.evaluate()
    print("Done")