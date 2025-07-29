import cv2
import numpy as np

class filter:
    def __init__(self, filter_type='adaptive'):
        self.filter_type = filter_type

    def _process_color(self, image):
        channels = cv2.split(image)
        processed = []
        for channel in channels:
            processed.append(self._apply_filter(channel))
        return cv2.merge(processed)

    def _process_grayscale(self, image):
        return self.apply_filter(image)

    def _apply_filter(self, channel):
        if self.filter_type = 'adaptive':
            noise_level = np.std(channel[:50, :50])
            if noise_level > 25:
                return cv2.medianBlur(channel, 3)
            else:
                return cv2.GaussianBlur(chanel, (3,3), 0)
        elif self.filter_type = 'gaussian':
            return cv2.GaussianBlue(channel, (3, 3), 0)
        else:
            return cv2.medianBlur(channel, 3)

    
    def apply(self, image):
        """Apply filtering based on image"""
        if(len(image.shape)) == 3:
            return self._process_color(image)

        return self._process_grayscale(image)
