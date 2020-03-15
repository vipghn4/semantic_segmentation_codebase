import os
import json
import importlib
from easydict import EasyDict
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd
import argparse

import torch
from datasets.standard_dataset import StandardDataset
from metrics import IoU, DICE
from trainers.utils import logits_to_onehot

def visualize(step, images, logits, masks, iou, save_dir):
    r"""Visualize prediction results"""
    os.makedirs(save_dir, exist_ok=True)
    image = images[0].cpu().numpy().transpose((1, 2, 0))
    pred_mask = torch.argmax(logits[0], dim=0).cpu().numpy()
    mask = masks[0].cpu().numpy()

    _class = np.unique(mask)[1]
    image = np.uint8(image * 255)
    pred_mask = colorize_class_from_mask(pred_mask, _class, (0, 0, 255))
    mask = colorize_class_from_mask(mask, _class, (0, 255, 0))
    visualized_image = np.concatenate([
        cv2.addWeighted(image, 1, mask, 0.5, 0),
        cv2.addWeighted(image, 1, pred_mask, 0.5, 0),
    ])
    cv2.imwrite(os.path.join(save_dir, f"{step}-{iou:.4f}.jpg"), visualized_image)

def colorize_class_from_mask(mask, _class, color):
    r"""Colorize segmentation mask"""
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask == _class] = color
    return colored_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_module", type=str, required=True, help="Model module / package name")
    parser.add_argument("--model_config_file", type=str, required=True, help="JSON file containing model configuration")
    parser.add_argument("--model_weights_file", type=str, required=True, help=".pth file containing model weights")
    parser.add_argument("--data_root", type=str, required=True, help="The directory containing `VOCdevkit` directory")
    parser.add_argument("--label_map_file", type=str, required=True, help="The label map file for VOC-2012 dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of threads used for batch generation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device used for training and inference")
    parser.add_argument("--save_dir", type=str, default="tmp/visualization", help="Directory to save visualization results")
    args = parser.parse_args()
    print(args)

    # load model
    with open(args.model_config_file) as f:
        model_config = json.load(f)
    module = importlib.import_module(args.model_module)
    model = getattr(module, "get_model")(model_config).to(args.device)

    # load pretrained weights
    checkpoint_data = torch.load(args.model_weights_file)
    model.load_state_dict(checkpoint_data["model"])
    model.eval()

    # load dataset
    data_config = EasyDict(dict(
        data_root=args.data_root,
        label_map_file=args.label_map_file,
        augment_data=None,
        preprocess=None,
        target_size=(512, 512)
    ))
    val_dataset = StandardDataset(data_config, split="val")
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    # evaluation
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
    metric_progress = []
    for step, batch in progress_bar:
        images = batch["image"].to(args.device)
        masks, onehot_masks = batch["mask"].to(args.device), batch["onehot_mask"].to(args.device)

        with torch.no_grad():
            logits = model(images)
            onehot_pred_masks = logits_to_onehot(logits)
            iou = IoU(onehot_pred_masks, masks, onehot_masks)
        
        visualize(step, images, logits, masks, iou.item(), args.save_dir)
        metric_progress.append({"iou": iou.item()})
        break
    metric_progress = pd.DataFrame(metric_progress)
    metric_progress.to_csv("eval_results.csv")