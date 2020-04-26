import os
import glob
import uuid
import json
from easydict import EasyDict
import numpy as np
import cv2
from PIL import Image

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from misc.voc2012_color_map import get_color_map

COLOR_MAP = get_color_map(256)


class StandardDataset(Dataset):
    def __init__(self, data_config, split="train"):
        r"""Standard Dataset class for loading VOC-2012 dataset.

        Args:
            data_config (EasyDict): Contain configuration of the VOC-2012 dataset.
                * data_root (str): The directory containing `VOCdevkit` directory.
                * label_map_file (str): The JSON file containing VOC-2012 label map.
                * augment_data (callable): The augmentation function to perturbate image and mask. It should take two BGR cv2 images `image` and `mask`, perturbate them, then returns two BGR cv2 perturbated images

                ```
                def augment_data(image, mask):
                    ...
                    return image, mask
                ```

                leave this `None` if there is no augmentation step.
                * preprocess (callable): The preprocess function to preprocess image and mask. It should take two BGR cv2 images `image` and `mask`, preprocesses them for segmentation, then returns two BGR cv2 preprocessed images

                ```
                def preprocess(image, mask):
                    ...
                    return image, mask
                ```

                leave this `None` if there is no preprocessing step.
                * ignored_class (int): The integer used to assign to ignored class 
            split (str): Which split of the dataset to collect. This can be either "train", "trainval", or "val". Default "train".
        
        `data_config.augment_data` will be invoked before `data_config.preprocess`
        """
        self.to_tensor = transforms.ToTensor()
        self.config = data_config
        self.label_map = self.__load_label_map()
        self.split_file = os.path.join(self.config.data_root, f"VOCdevkit/VOC2012/ImageSets/Segmentation/{split}.txt")
        self.items = self.__collect_dataset()
    
    def __load_label_map(self):
        f"""Load label map for VOC-2012"""
        with open(self.config.label_map_file) as f:
            label_map = json.load(f)
        return label_map

    def __collect_dataset(self):
        r"""Load image paths and mask paths from data root"""
        items = []
        with open(self.split_file) as f:
            content = [line.strip() for line in f.readlines()]
        for image_id in content:
            image_path = os.path.join(self.config.data_root, "VOCdevkit/VOC2012/JPEGImages", f"{image_id}.jpg")
            mask_path = os.path.join(self.config.data_root, "VOCdevkit/VOC2012/SegmentationClass", f"{image_id}.png")
            items.append(EasyDict(dict(
                image_id=image_id,
                image_path=image_path,
                mask_path=mask_path
            )))
        return items
    
    def __getitem__(self, idx):
        r"""Get an item with index `idx` from the dataset"""
        original_image = cv2.imread(self.items[idx].image_path)
        original_mask = np.array(Image.open(self.items[idx].mask_path))
        original_mask[original_mask == 255] = self.config.ignored_class
        
        image, mask = original_image.copy(), original_mask.copy()
        if self.config.augment_data is not None:
            image, mask = self.config.augment_data(image, mask)
        
        if self.config.preprocess is not None:
            image, mask = self.config.preprocess(image, mask)
        
        print(type(image))
        image = self.to_tensor(image)
        mask = self.__get_mask_tensor(mask)
        onehot_mask = self.__get_onehot_mask(mask)
        return EasyDict(dict(image=image, mask=mask, onehot_mask=onehot_mask))
    
    def __get_mask_tensor(self, mask):
        r"""Convert ndarray mask to torch.tensor mask"""
        mask = torch.from_numpy(mask).squeeze().long()
        return mask
    
    def __get_onehot_mask(self, mask):
        r"""Get onehot mask from integer mask"""
        onehot_mask = F.one_hot(mask, num_classes=22)
        onehot_mask = onehot_mask.permute(2, 0, 1)
        onehot_mask = onehot_mask[:-1]
        return onehot_mask

    def __len__(self):
        r"""Get dataset size"""
        return len(self.items)


def preprocess(image, mask, **kwargs):
    r"""Preprocess image and mask for VOC-2012 dataset. Default mean-subtraction is performed. Please specify a model_variant.

    Args:
        image (np.array): BGR image of shape (h, w, c)
        mask (np.array): np.array of shape (h, w)
    Returns:
        np.array: Preprocessed BGR image
        np.array: Preprocessed mask
    """
    return image, mask


if __name__ == "__main__":
    data_config = EasyDict(dict(
        data_root="/home/cotai/giang/datasets/VOC-2012",
        label_map_file="/home/cotai/giang/datasets/VOC-2012/label_map.json",
        augment_data=None,
        preprocess=None,
        ignored_class=21
    ))
    dataset = StandardDataset(data_config, split="train")
    
    os.makedirs("tmp/visualization/StandardDataset", exist_ok=True)
    for i in range(len(dataset)):
        item = dataset[i]
        image, mask, onehot_mask = item.image, item.mask, item.onehot_mask
        
        image = image.cpu().numpy()
        image = (image * 255).transpose((1, 2, 0)).astype(np.uint8)
        
        mask = mask.cpu().numpy()
        colored_mask = np.zeros_like(image)
        for color in range(21):
            colored_mask[mask == color] = COLOR_MAP[color]
        
        onehot_mask = onehot_mask.cpu().numpy()
        onehot_mask = np.argmax(onehot_mask.transpose(1, 2, 0), axis=-1)
        colored_onehot_mask = np.zeros_like(image)
        for color in range(21):
            colored_onehot_mask[onehot_mask == color] = COLOR_MAP[color]
        
        vis = np.concatenate([
            cv2.addWeighted(image, 1, colored_mask, 0.7, 0),
            cv2.addWeighted(image, 1, colored_onehot_mask, 0.7, 0),
        ], axis=1)
        
        cv2.imwrite(f"tmp/visualization/StandardDataset/{i}.jpg", vis)
        if i == 20:
            break