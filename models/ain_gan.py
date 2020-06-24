import tensorflow as tf

from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import Input
from tensorflow.keras import Model

from .ain import AdaIN
from .style_encoder import style_encoder_v1
from .content_encoder import content_encoder_v1
from .cnn_blocks import conv_transpose_block
from .cnn_blocks import conv_transpose_block_x2
from .cnn_blocks import conv_transpose_block_x3


def ain_gan_v1(content_inputs, style_inputs, normalization, dropout, activation, \
    reg_coef, skip, **kwargs):

    conv_transpose_x2 = lambda inputs, filters: conv_transpose_block_x2(
        inputs, filters, normalization=normalization, dropout=dropout, \
        activation=activation, reg_coef=reg_coef, skip=skip, **kwargs)

    conv_transpose_x3 = lambda inputs, filters: conv_transpose_block_x3(
        inputs, filters, normalization=normalization, dropout=dropout, \
        activation=activation, reg_coef=reg_coef, skip=skip, **kwargs)

    conv_transpose = lambda inputs, filters, \
        normalization, dropout, activation: conv_transpose_block(
        inputs, filters, normalization=normalization, dropout=dropout, \
        activation=activation, reg_coef=reg_coef, skip=False, **kwargs)

    with tf.name_scope("ContentEncoder"):
        content_b1, content_b2, content_b3, \
            content_outputs = content_encoder_v1(
            content_inputs, normalization, dropout, activation, reg_coef, \
            skip, **kwargs)

    with tf.name_scope("StyleEncoder"):
        style_outputs = style_encoder_v1(
            style_inputs, normalization, dropout, activation, reg_coef, \
            skip, **kwargs)

    with tf.name_scope("AdaptiveDecoder"):
        outputs = AdaIN()(content_outputs, style_outputs)
        outputs = conv_transpose_x3(outputs, 512)
        outputs = UpSampling2D()(outputs)

        outputs = conv_transpose_x3(outputs, 256)
        outputs = UpSampling2D()(outputs)

        adain_outputs = AdaIN()(content_b3, style_outputs)
        outputs = Concatenate()([adain_outputs, outputs])
        outputs = conv_transpose_x3(outputs, 128)
        outputs = UpSampling2D()(outputs)

        adain_outputs = AdaIN()(content_b2, style_outputs)
        outputs = Concatenate()([adain_outputs, outputs])
        outputs = conv_transpose_x2(outputs, 64)
        outputs = UpSampling2D()(outputs)

        adain_outputs = AdaIN()(content_b1, style_outputs)
        outputs = Concatenate()([adain_outputs, outputs])
        outputs = conv_transpose(
            outputs, 64, normalization, dropout, activation)
        outputs = conv_transpose(outputs, 3, None, None, None)

        return outputs


def AdaINGAN_v1(content_shape, style_shape, skip=True, reg_coef=1e-3):
    content_input = Input(shape=content_shape, name='content')
    style_input = Input(shape=style_shape, name='style')


    outputs = ain_gan_v1(content_input, style_input, \
        LayerNormalization, None, LeakyReLU(0.01), reg_coef, skip)

    model = Model(inputs=[content_input, style_input], outputs=outputs)

    return model











