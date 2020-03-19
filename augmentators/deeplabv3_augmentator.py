import numpy as np
import cv2
from albumentaions import Compose, RandomScale, RandomCrop, VerticalFlip

class DeeplabV3Augmentator:
    def __init__(self, ignored_class=21):
        r"""Augmentator for DeeplabV3.
        
        Reference: https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/data_generator.py"""
        self.random_scale = RandomScale(scale_limit=0.5, interpolation=cv2.INTER_NEAREST)
        self.random_crop = RandomCrop(crop_size[1], crop_size[0], always_apply=True)
        self.random_flip = VerticalFlip(p=0.5)
        self.ignored_class = ignored_class
    
    def __call__(self, image, mask, target_size, crop_size, augment=False):
        r"""Pad and resize image and mask. Image is zero-padded to have dimensions >= [crop_height, crop_width]. Mask is padded with ignored class pixel.
        
        Args:
            image (np.array): BGR image of shape (h, w, c)
            mask (np.array): np.array of shape (h, w)
            target_size (tuple of 2 ints): Target size to resize image and mask before augmentation and cropping.
            crop_size (tuple of 2 ints): New size of the form (w, h)
        Returns:
            np.array: Preprocessed BGR image
            np.array: Preprocessed mask
        """
        image, mask = self.__resize(image, mask, target_size)
        augmented = self.random_scale(image=image, mask=mask)
        augmented = self.__crop(augmented["image"], augmented["mask"], crop_size)
        augmented = self.random_flip(image=augmented["image"], mask=augmented["mask"])
        return augmented["image"], augmented["mask"]

    def __resize(self, image, mask, target_size):
        r"""Pad and resize image and mask"""
        padded_shape = (int(max(image.shape[1], image.shape[0] * target_size[0] / target_size[1])), 
                        int(max(image.shape[0], image.shape[1] * target_size[1] / target_size[0])))
        pad_left = (padded_shape[0] - image.shape[1]) // 2
        pad_right = padded_shape[0] - image.shape[1] - pad_left
        pad_top = (padded_shape[1] - image.shape[0]) // 2
        pad_bottom = padded_shape[1] - image.shape[0] - pad_top
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")
        mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
        image = cv2.resize(image, tuple(target_size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, tuple(target_size), interpolation=cv2.INTER_NEAREST)
        return image, mask

    def __crop(self, image, mask, crop_size):
        r"""Pad image and mask. Pad image and mask to have dimensions >= [crop_height, crop_width] then crop them"""
        crop_size = max(crop_size[0], image.shape[1]), max(crop_size[1], image.shape[0])
        image = np.pad(image, pad_width=((0, crop_size[1] - image.shape[0]), (0, crop_size[0] - image.shape[1]), (0, 0)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((0, crop_size[1] - mask.shape[0]), (0, crop_size[0] - mask.shape[1]), (0, 0)), mode="constant", constant_values=self.ignored_class)
        cropped = self.random_crop(image=image, mask=mask)
        return cropped["image"], cropped["mask"]