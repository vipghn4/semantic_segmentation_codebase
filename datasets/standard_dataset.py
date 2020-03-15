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
                * target_size (tuple of 2 ints): The target size of images and masks of the dataset. The format is (w, h).
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
        image = cv2.imread(self.items[idx].image_path)
        mask = np.array(Image.open(self.items[idx].mask_path))
        mask[mask == 255] = 21
        image, mask = self.__resize_items(image, mask)
        
        if self.config.augment_data is not None:
            image, mask = self.config.augment_data(image, mask)
        
        if self.config.preprocess is not None:
            image, mask = self.config.preprocess(image, mask)
        
        image = self.to_tensor(image)
        mask = self.__get_mask_tensor(mask)
        onehot_mask = self.__get_onehot_mask(mask)        
        return EasyDict(dict(
            image=image, mask=mask, onehot_mask=onehot_mask
        ))
    
    def __get_mask_tensor(self, mask):
        r"""Convert ndarray mask to torch.tensor mask"""
        mask = torch.from_numpy(mask).squeeze().long()
        return mask
    
    def __get_onehot_mask(self, mask):
        r"""Get onehot mask from integer mask"""
        onehot_mask = F.one_hot(mask)
        onehot_mask = onehot_mask.permute(2, 0, 1)
        onehot_mask = onehot_mask[:-1]
        return onehot_mask

    def __resize_items(self, image, mask):
        r"""Pad and resize image and mask"""
        padded_shape = (int(max(image.shape[1], image.shape[0] * self.config.target_size[0] / self.config.target_size[1])), 
                        int(max(image.shape[0], image.shape[1] * self.config.target_size[1] / self.config.target_size[0])))
        pad_left = (padded_shape[0] - image.shape[1]) // 2
        pad_right = padded_shape[0] - image.shape[1] - pad_left
        pad_top = (padded_shape[1] - image.shape[0]) // 2
        pad_bottom = padded_shape[1] - image.shape[0] - pad_top
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")
        mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
        image = cv2.resize(image, tuple(self.config.target_size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, tuple(self.config.target_size), interpolation=cv2.INTER_NEAREST)
        return image, mask

    def __len__(self):
        r"""Get dataset size"""
        return len(self.items)
    
    def __visualize_class(self, image, mask, onehot_mask, _class):
        r"""Visualize generated data for data verification"""
        image = (255 * image.numpy().transpose((1, 2, 0))).astype(np.uint8)
        mask = mask.numpy()
        onehot_mask = onehot_mask.numpy().transpose((1, 2, 0))
        
        if _class in np.unique(mask):
            mask[mask != _class] = 0
            mask[mask == _class] = 255
            mask = np.repeat(mask[..., None], 3, axis=-1).astype(np.uint8)
            mask[..., [0, 2]] = 0
            onehot_mask = onehot_mask[..., _class] * 255
            onehot_mask = np.repeat(onehot_mask[..., None], 3, axis=-1).astype(np.uint8)
            onehot_mask[..., :-1] = 0
            vis = np.concatenate([
                cv2.addWeighted(image, 1, mask, 0.5, 0),
                cv2.addWeighted(image, 1, onehot_mask, 0.5, 0),
            ], axis=1)
            cv2.imwrite(f"tmp/{str(uuid.uuid4())}.jpg", vis)

    
if __name__ == "__main__":
    data_config = EasyDict(dict(
        data_root="/home/cotai/giang/datasets/VOC-2012",
        label_map_file="/home/cotai/giang/datasets/VOC-2012/label_map.json",
        augment_data=None,
        preprocess=None,
        target_size=(513, 513)
    ))
    dataset = StandardDataset(data_config, split="train")
    for i in range(len(dataset)):
        dataset[i]