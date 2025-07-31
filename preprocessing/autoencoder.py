import tensorflow as tf
import cv2
import numpy as np

class autoencoder:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = self.model.input[1:3]

    def denoise(self, image):
        """apply denoising"""
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if (image.shape != self.input_shape):
            resized = cv2.resize(image, self.input_shape)
        else:
            resized = image

        normalized = resized / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        output = self.model.predict(input_tensor)
        denoised = output[0] * 255.0
        if image.shape != denoised.shape:
            denoised = cv2.resize(denoised, image.shape)
        return denoised