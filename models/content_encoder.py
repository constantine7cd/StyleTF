import tensorflow as tf

from tensorflow.keras.layers import MaxPool2D

from .cnn_blocks import conv_block
from .cnn_blocks import conv_block_x2
from .cnn_blocks import conv_block_x3
from .cnn_blocks import conv_transpose_block
from .cnn_blocks import conv_transpose_block_x2
from .cnn_blocks import conv_transpose_block_x3


def content_encoder_v1(inputs, normalization, dropout, activation, \
    reg_coef, skip, **kwargs):

    conv_x2 = lambda inputs, filters: conv_block_x2(
        inputs, filters, normalization=normalization, dropout=dropout, \
        activation=activation, reg_coef=reg_coef, skip=skip, **kwargs)

    conv_x3 = lambda inputs, filters: conv_block_x3(
        inputs, filters, normalization=normalization, dropout=dropout, \
        activation=activation, reg_coef=reg_coef, skip=skip, **kwargs)

    outputs_b1 = conv_x2(inputs, filters=64)
    outputs = MaxPool2D()(outputs_b1)

    outputs_b2 = conv_x2(outputs, filters=128)
    outputs = MaxPool2D()(outputs_b2)

    outputs_b3 = conv_x3(outputs, filters=256)
    outputs = MaxPool2D()(outputs_b3)

    outputs = conv_x3(outputs, filters=512)
    outputs = MaxPool2D()(outputs)

    return outputs_b1, outputs_b2, outputs_b3, outputs

