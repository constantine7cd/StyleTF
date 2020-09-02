import tensorflow as tf

from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow_addons.layers import InstanceNormalization
from tensorflow import nn
from .conv_blocks import conv_block
from .conv_blocks import res_block
from .norm import InstanceNorm

def _preparation_layer(inputs, filters):
    norm = InstanceNorm
    #norm = BatchNormalization
    activation = ReLU

    # norm_kwargs = {
    #     'center': False,
    #     'scale': False}

    outputs = conv_block(
        inputs, filters, kernel_size=9, padding=4, \
            norm=norm, activation=activation, norm_kwargs={})

    return outputs


def _downsample_layer(inputs, filters):
    norm = InstanceNorm
    #norm = BatchNormalization
    activation = ReLU

    norm_kwargs = {
        'center': False,
        'scale': False}

    outputs = conv_block(inputs, filters, kernel_size=6, \
        padding=2, stride=2, norm=norm, activation=activation, \
        norm_kwargs={})

    return outputs


def _residual_layer(inputs, filters):
    norm = InstanceNorm
    #norm = BatchNormalization
    activation = ReLU

    # norm_kwargs = {
    #     'center': False,
    #     'scale': False}

    outputs = res_block(
        inputs, filters, norm=norm, activation=activation, \
        norm_kwargs={})

    return outputs


def content_encoder(
    inputs, 
    filters=48,  
    num_downsamples=2, 
    num_res_blocks=4, 
    skip_dim=5):

    outputs = _preparation_layer(inputs, filters)
    skips = []

    for _ in range(num_downsamples):
        filters *= 2

        skips.append(outputs[:,:,:, :skip_dim])
        outputs = _downsample_layer(outputs, filters)

    for _ in range(num_res_blocks):
        outputs = _residual_layer(outputs, filters)

    skips.reverse()

    return outputs, skips


def ContentEncoder(input_shape=(1024, 1024, 3)):
    inputs = tf.keras.Input(input_shape)
    #outputs, skips = content_encoder(inputs)
    outputs = content_encoder(inputs)

    #model = tf.keras.Model(inputs=inputs, outputs=[outputs, *skips])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


ceModel = ContentEncoder()
