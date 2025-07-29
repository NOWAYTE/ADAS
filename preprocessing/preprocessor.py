import cv2
import numpy as np
from .compression import JPEGCompressor, BitDepthReducer
from .filtering import AdaptiveFilter
from .autoencoder import DenoisingAutoencoder
from .spatial import SmartResizer

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

        self.compressor = JPEGCompressor(self.config['jpeq_quality'])
        self.bit_reducer = bit_reduce(self.config['target_bits'])

        def process(self,  image):
            compressed = self.compressor.apply(image)

        def fast_process(self, image):
            compressed = self.compressor.apply(image)
