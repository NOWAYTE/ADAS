import cv2
import numpy as np
import io
from PIL import Image

class jpeg_compressor:
    def __init__(self, quality=75):
        self.quality = quality

    def apply(self, Image):
        """ Apply compression """
        if image.dtype != np.uint8:
            image = (image * 255) astype(np.uint8)
            img = image.fromArray(image)
            img.save(buffer, format='JPEG', quality=self.quality)
            buffer.seek(0)
            compressed = Image.open(buffer)
            return np.array(compressed)

    class bit_reducer:
        def __init__(self, target_bits=5):
            self.target_bits = min(max(target_bits, 2), 8)

        def apply(self, image):
            """ Reduce color depth"""
            if image.dtype != np.uint8:
                max_val = 255
            else:
                max_val = 1.0

            steps = 2**self.target_bits
            quantized = np.round(image * (steps-1) / (steps - 1))
            return np.clip(quantized,  0, max_val)

