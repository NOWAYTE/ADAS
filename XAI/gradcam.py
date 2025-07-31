import cv2
import numpy as np
import tensorflow as tf

class gradcam:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

    def _get_target_layer(self, layer_name):
        for layer in self.model.layers:
            if layer.name == layer_name:
                return layer
        raise ValueError(f"Layer {layer_name} not found in model")

    def explain(self, image, class_idx=None):
        img_tensor = tf.convert_to_tensor(image)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        

        cam = tf.redue_sum(weights * conv_outputs, axis=-1)
        cam = cv2.resize(cam.numpy()[0], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(255 * cam)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        explained = cv2.addWeighted(image, 0.4, cam, 0.6, 0)

        return {
            'cam': cam,
            'explained': explained,
            'class_idx': class_idx,
            'score': predictions[:, class_idx]
        }
        
            