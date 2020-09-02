import torch
import tensorflow as tf
import torch.nn.functional as F

from torch import nn
from tensorflow.keras.layers import Conv2D

from .adain import AdaptiveInstanceNormalization


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class TinyTorch(nn.Module):
    def __init__(self, input_dim=3):
        super(TinyTorch, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, 16, 5)
        self.norm = AdaptiveInstanceNorm2d(16)

    def forward(self, input_):
        output = self.conv1(input_)
        output = self.norm(output)

        return output


def TinyTF(input_shape=(24, 24, 3)):
    inputs = tf.keras.Input(input_shape)
    gamma = tf.keras.Input((16,))
    beta = tf.keras.Input((16,))

    outputs = Conv2D(16, 5)(inputs)
    outputs = AdaptiveInstanceNormalization()(
        outputs, gamma, beta)

    model = tf.keras.Model(
        inputs=[inputs, gamma, beta], outputs=outputs)

    return model