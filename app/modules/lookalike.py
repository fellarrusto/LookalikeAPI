import logging
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

LOGGER = logging.getLogger(__name__)

class Lookalike:

    def __init__(self, model_path='app/model/encoder_model.h5'):
        # Custom Sampling layer
        class Sampling(Layer):
            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # Load the model with the custom layer
        with custom_object_scope({'Sampling': Sampling}):
            self.model = load_model(model_path)

    def embed(self, img):
        img = self.preprocess_image(img)
        embedding = self.model.predict(img)[0][0]
        LOGGER.info(embedding)
        return embedding
    
    # Utils
    def preprocess_image(self, image):

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image_np = np.array(image)

        # Resize image to 64x64
        image = tf.image.resize(image_np, [64, 64])
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        LOGGER.info(image.shape)
        return image
