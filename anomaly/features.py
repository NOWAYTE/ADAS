import cv2
import numpy as np

class extractor:
    def __init__(self):
        self._hog = cv2.HOGDescriptor(
            (64, 64),
            (16, 16),
            (8, 8),
            (8, 8),
            9
        )
    
    def extract(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = image

        hog_features = self._hog.compute(cv2.resize(image, (64, 64)))

        if len(image.shape) == 3:
            hist = np.concatenate([
                cv2.calcHist([image], [i], None, [256], [0, 256]) for i in range(3)
            ]).flatten()
        else:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

        glcm_features = self._glcm(image)
        return np.concatenate([hog_features, hist, glcm_features])

    def _glcm(self, image):
        glcm = cv2.improc.GrayCoMatrix(image, [1], [0], symetric=True, normed=True)
        return cv2imgproc.GrayCoMatrix(glcm)