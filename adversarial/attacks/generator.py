from .attacks.digital import fgsm
from .attacks.physical import stickers

class generator:
    def __init__(self, config=None):
        self.config = config or {
            'digital': {
                'method': 'fgsm',
                'epsilon': 0.1
            },
            'physical': {
                'method': 'stickers',
                'pattern_type': 'checkerboard',
                'opacity': 0.5
            }
        }
        self.digital_attack = fgsm(self.config['digital']['epsilon'])
        self.physical_attack = stickers(self.config['physical']['pattern_type'], self.config['physical']['opacity'])


    def generate(self, image, bbox=None):
        if self.config['digital']['method'] == 'fgsm':
            return self.digital_attack.generate(image, bbox)
        elif self.config['physical']['method'] == 'stickers':
            return self.physical_attack.apply(image, bbox)
        else:
            raise ValueError(f"Unknown method: {self.config['digital']['method']}")