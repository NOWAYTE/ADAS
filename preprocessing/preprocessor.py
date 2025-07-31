import cv2
import numpy as np
from .compression import jpeg_compressor, bit_reducer
from .filtering import filter
from .autoencoder import autoencoder
from .spatial import resizer

class preprocessor:
    def __init__(self, config=None):
        self.config = config or {
                'jpeg_quality': 75,
                'target_bits': 5,
                'filter_type': 'adaptive',
                'ae_model_path': 'models/', # WIP
                'target_size': (224, 224),
                'pad_mode': 'reflect'
                }

        self.compressor = jpeg_compressor(self.config['jpeg_quality'])
        self.bit_reducer = bit_reducer(self.config['target_bits'])
        self.filter = filter(self.config['filter_type'])
        self.resizer = resizer(self.config['target_size'], self.config['pad_mode'])
        self.encoder = autoencoder(self.config['ae_model_path'])


        def process(self,  image):
            compressed = self.compressor.apply(image)
            reduced = self.bit_reducer.apply(compressed)
            filtered = self.filter.apply(reduced)
            resized = self.resizer.apply(filtered)
            denoised = self.encoder.denoise(resized)
            return denoised

        def fast_process(self, image):
            compressed = self.compressor.apply(image)
