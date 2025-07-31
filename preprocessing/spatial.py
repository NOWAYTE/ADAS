import cv2
import numpy as np

class resizer:
    def __init__(self, target_size, pad_mode='reflect'):
        self.target_size = target_size
        self.pad_mode = pad_mode

    def apply(self, image):
        """apply smart resizer"""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size

        image_ratio = h / w
        target_ratio = target_h / target_w

        if image_ratio > target_ratio:
            new_h = target_h
            new_w = int(target_h * (w / h))
        else:
            new_w = target_w
            new_h = int(target_w * (h / w))

        resized = cv2.resize(image, (new_w, new_h))
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        
        if len(image.shape) == 2:
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        else:
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT, value=[0, 0, 0])
        return padded
        