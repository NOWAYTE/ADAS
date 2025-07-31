import cv2
import numpy as np

class stickers:
    def __init__(self, pattern_type='checkerboard'):
        self.patterns = {
            'checkerboard': self._checkerboard,
            'grid': self._grid,
            'random': self._random
        }
        self.pattern_type = pattern_type
        self.opacity = opacity

    def _generate_pattern(self, size):
        base = self.pattern_func(size)
        return (base * 255).astype(np.uint8)
    def apply(self, image, bbox=None):

        if bbox is None:
            h, w = image.shape[:2]
            bbox = [w//4, h//4, 3*w//4, 3*h//4]

        pattern = self._generate_pattern(
            size=(bbox[2] - bbox[0], bbox[3] - bbox[1])

        )
        pertubed = image.copy()
        pertubed[bbox[1]:bbox[3], bbox[0]:bbox[2]] = (
            opacity * pattern + (1 - opacity) * image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        )

        return pertubed

    def _checkerboard(self, size):
        pattern = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        square = max(size[0] // 64, size[1] // 64)
        for i in range(0, size[0], square):
            for j in range(0, size[1], square):
                pattern[j:j+square, i:i+square] = np.random.randint(0, 256, (square, square, 3))

        return pattern

        