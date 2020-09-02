import tensorflow as tf

from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.layers import Add

from .utils import reflect_padding_2d


def conv_block(
    inputs, filters, kernel_size, \
        padding=0, stride=1, norm=None, activation=None, \
        norm_kwargs={}, activation_kwargs={}):

    outputs = reflect_padding_2d(inputs, padding)
    outputs = Conv2D(
        filters, kernel_size, strides=stride)(outputs)

    if norm is not None:
        outputs = norm(**norm_kwargs)(outputs)

    if activation is not None:
        outputs = activation(**activation_kwargs)(outputs)
     
    return outputs


def res_block(
    inputs, filters, kernel_size=3, \
        padding=1, stride=1, norm=None, activation=None, \
        norm_kwargs={}, activation_kwargs={}, skip=True):

    residual = inputs

    outputs = conv_block(
        inputs, filters, kernel_size, padding, stride, \
        norm, activation, norm_kwargs, activation_kwargs)

    outputs = conv_block(
        outputs, filters, kernel_size, padding, stride, \
        norm, None, norm_kwargs, activation_kwargs)

    if skip:
        outputs = Add()([outputs, residual])

    return outputs


def conv_block_adain(
    inputs, gamma, beta, filters, kernel_size = 3, padding = 1, stride = 1, \
        norm = None, activation = None, norm_kwargs = {}, activation_kwargs = {}):

    outputs = reflect_padding_2d(inputs, padding)
    outputs = Conv2D(
        filters, kernel_size, strides=stride)(outputs)

    if norm is not None:
        outputs = norm(**norm_kwargs)(outputs, gamma, beta)

    if activation is not None:
        outputs = activation(**activation_kwargs)(outputs)
     
    return outputs   
    
    
def res_block_adain(
    inputs, filters, \
    kernel_size=3, padding=1, stride=1, norm=None, \
    activation=None, norm_kwargs={}, activation_kwargs={}, skip=True):

    inputs, gamma1, beta1, gamma2, beta2 = inputs

    residual = inputs

    outputs = conv_block_adain(
        inputs, gamma1, beta1, filters, kernel_size, padding, stride, \
        norm, ReLU, norm_kwargs, activation_kwargs)

    outputs = conv_block_adain(
        outputs, gamma2, beta2, filters, kernel_size, padding, stride, \
        norm, None, norm_kwargs, activation_kwargs)

    if skip:
        outputs = Add()([outputs, residual])

    return outputs
    

    





