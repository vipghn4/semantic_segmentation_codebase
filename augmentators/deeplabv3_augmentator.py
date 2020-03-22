import os
from easydict import EasyDict
import numpy as np
import cv2
from albumentations import Compose, RandomScale, RandomCrop, HorizontalFlip

from datasets import StandardDataset
from misc.voc2012_color_map import get_color_map

COLOR_MAP = get_color_map(256)


class DeeplabV3Augmentator:
    def __init__(self, target_size, crop_size, ignored_class=21):
        r"""Augmentator for DeeplabV3.
        Reference: https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/data_generator.py

        Args:
            target_size (tuple of 2 ints): Target size to resize image and mask before augmentation and cropping.
            crop_size (tuple of 2 ints): New size of the form (w, h)
        """
        self.w_target, self.h_target = target_size
        self.w_crop, self.h_crop = crop_size
        self.random_scale = RandomScale(scale_limit=0.5, interpolation=cv2.INTER_NEAREST)
        self.random_crop = RandomCrop(self.h_crop, self.w_crop, always_apply=True)
        self.random_flip = HorizontalFlip(p=0.5)
        self.ignored_class = ignored_class
    
    def __call__(self, image, mask):
        r"""Pad and resize image and mask. Image is zero-padded to have dimensions >= [crop_height, crop_width]. Mask is padded with ignored class pixel.
        
        Args:
            image (np.array): BGR image of shape (h, w, c)
            mask (np.array): np.array of shape (h, w)
        Returns:
            np.array: Preprocessed BGR image
            np.array: Preprocessed mask
        """
        image, mask = self.__resize(image, mask)
        augmented = self.random_scale(image=image, mask=mask)
        augmented = self.__crop(augmented["image"], augmented["mask"])
        augmented = self.random_flip(image=augmented["image"], mask=augmented["mask"])
        return augmented["image"], augmented["mask"]

    def __resize(self, image, mask):
        r"""Pad and resize image and mask"""
        h, w, _ = image.shape
        w_pad = int(max(w, h * self.w_target / self.h_target)) 
        h_pad = int(max(h, w * self.h_target / self.w_target))
        pad_left = (w_pad - w) // 2
        pad_right = w_pad - w - pad_left
        pad_top = (h_pad - h) // 2
        pad_bottom = h_pad - h - pad_top
        pad = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=self.ignored_class)
        image = cv2.resize(image, (self.w_target, self.h_target), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.w_target, self.h_target), interpolation=cv2.INTER_NEAREST)
        return image, mask

    def __crop(self, image, mask):
        r"""Pad image and mask. Pad image and mask to have dimensions >= [crop_height, crop_width] then crop them"""
        h, w, _ = image.shape
        w_crop, h_crop = max(self.w_crop, w), max(self.h_crop, h)
        image = np.pad(image, ((0, h_crop - h), (0, w_crop - w), (0, 0)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((0, h_crop - h), (0, w_crop - w)), mode="constant", constant_values=self.ignored_class)
        cropped = self.random_crop(image=image, mask=mask)
        return {"image": cropped["image"], "mask": cropped["mask"]}


if __name__ == "__main__":
    augmentator = DeeplabV3Augmentator(
        target_size=(700, 700),
        crop_size=(512, 512)
    )
    data_config = EasyDict(dict(
        data_root="/home/cotai/giang/datasets/VOC-2012",
        label_map_file="/home/cotai/giang/datasets/VOC-2012/label_map.json",
        augment_data=augmentator,
        preprocess=None,
        target_size=(512, 512),
        ignored_class=21
    ))
    dataset = StandardDataset(data_config, split="train")

    os.makedirs("tmp/visualization/DeeplabV3Augmentator", exist_ok=True)
    for i in range(len(dataset)):
        item = dataset[i]
        image, mask = item.image, item.mask
        original_image, original_mask = dataset[i].original_image, dataset[i].original_mask
        
        image = image.cpu().numpy()
        image = (image * 255).transpose((1, 2, 0)).astype(np.uint8)
        mask = mask.cpu().numpy()
        colored_mask = np.zeros_like(image)
        for color in range(21):
            colored_mask[mask == color] = COLOR_MAP[color]
        vis = cv2.addWeighted(image, 1, colored_mask, 0.7, 0)
        cv2.imwrite(f"tmp/visualization/DeeplabV3Augmentator/{i}.jpg", vis)
        
        colored_mask = np.zeros_like(original_image)
        for color in range(21):
            colored_mask[original_mask == color] = COLOR_MAP[color]
        original_vis = cv2.addWeighted(original_image, 1, colored_mask, 0.7, 0)
        cv2.imwrite(f"tmp/visualization/DeeplabV3Augmentator/{i}-original.jpg", original_vis)
        if i == 20:
            break