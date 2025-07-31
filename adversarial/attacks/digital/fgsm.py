import tensorflow as tf
import numpy as np


class fgsm:
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon

    def generate(self, image, target_class=None):
        image_tensor = tf.convert_to_tensor(image[np.newaxis, ...])
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            predictions = self.model(image_tensor)
            loss = self.__compute_loss(predictions, target_class)
            gradient =  tape.gradient(loss, image_tensor)
            perturbation = self.epsilon * tf.sign(gradient)
            adversarial = image + perturbation
            adversarial = tf.clip_by_value(adversarial, 0, 1)
            return adversarial.numpy()[0]

    def __compute_loss(self, predictions, target_class):
        if target_class is None:
            class = tf.argmax(predictions, axis=-1)
            return -predictions[0, class]
        else:
            return -predictions[0, target_class]