from .gradcam import gradcam
import numpy as np

class explainer:
    def __init__(self, model, config=None):
        self.config = config or {
            'layer_name': 'conv2d_1',
            'method': 'gradcam',
            'alpha': 0.5,
            'beta': 0.5
        }
        self._init_explainer(model)

    def _init_explainer(self, model):
        if self.config['method'] == 'gradcam':
            self.explainer = gradcam(model, self.config['layer_name'])
        else:
            raise ValueError(f"Unknown method: {self.config['method']}")

    def explain(self, image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=2)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return self.explainer.explain(image)
    
    def explain_batch(self, images):

        class_map = {0: 'STOP', 1: 'YIELD', 2: 'GREEN', 3: 'YELLOW', 4: 'RED', 5: 'SPEED_LIMIT', 6: 'OTHER'}
        explanations = []
        for image in images:
            explanations.append(self.explain(image))
        return explanations
    
       