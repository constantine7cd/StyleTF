import tensorflow as tf

from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow_addons.layers import InstanceNormalization
from tensorflow import nn


class InstanceNorm(tf.keras.layers.Layer):
    def __init__(
        self,
        epsilon: float = 1e-3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.moments_axes = [1, 2]

    def call(self, inputs):
        mean, variance = nn.moments(
            inputs, self.moments_axes, keepdims=True)

        outputs = nn.batch_normalization(
            inputs, mean, variance, None, None, self.epsilon, name='InstanceNorm')

        return outputs

class AdaptiveInstanceNorm(tf.keras.layers.Layer):
    def __init__(
        self,
        epsilon: float = 1e-3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.moments_axes = [1, 2]

    def call(self, inputs, gamma, beta):
        mean, variance = nn.moments(
            inputs, self.moments_axes, keepdims=True)

        outputs = nn.batch_normalization(
            inputs, mean, variance, gamma, beta, self.epsilon, name='AdaptiveInstanceNorm')

        return outputs
    
class LayerNorm(tf.keras.layers.Layer):
    def __init__(
        self,
        epsilon: float = 1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.moments_axes = -1

    def call(self, inputs):
        mean, variance = nn.moments(
            inputs, self.moments_axes, keepdims=True)

        outputs = nn.batch_normalization(
            inputs, mean, variance, None, None, self.epsilon, name='LayerInstanceNorm')

        return outputs