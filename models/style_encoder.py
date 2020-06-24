import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.regularizers import l2

from .cnn_blocks import conv_block
from .cnn_blocks import conv_block_x2
from .cnn_blocks import conv_block_x3
from .cnn_blocks import conv_transpose_block
from .cnn_blocks import conv_transpose_block_x2
from .cnn_blocks import conv_transpose_block_x3


def style_encoder_v1(inputs, normalization, dropout, activation, \
    reg_coef, skip, **kwargs):

    conv_x2 = lambda inputs, filters: conv_block_x2(
        inputs, filters, normalization=normalization, dropout=dropout, \
        activation=activation, reg_coef=reg_coef, skip=skip, **kwargs)

    outputs = conv_x2(inputs, filters=64)
    outputs = MaxPool2D()(outputs)

    outputs = conv_x2(outputs, filters=128)
    outputs = MaxPool2D()(outputs)

    outputs = conv_x2(outputs, filters=256)
    outputs = MaxPool2D()(outputs)

    outputs = conv_x2(outputs, filters=512)
    outputs = MaxPool2D()(outputs)

    regularizer = l2(reg_coef) if reg_coef is not None else None

    outputs = Conv2D(
        filters=3, 
        kernel_size=1, 
        padding='same', 
        kernel_regularizer=regularizer
    )(outputs)

    outputs = tf.reduce_mean(outputs, axis=[1, 2])

    return outputs