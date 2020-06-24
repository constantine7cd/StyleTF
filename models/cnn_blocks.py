import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Convolution2DTranspose
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ReLU
from tensorflow.keras.regularizers import l2


def __nn_ops(outputs, normalization, dropout, activation, **kwargs):
    if normalization is not None:
        normalization_params = kwargs.get('normalization_params', {})
        outputs = normalization(**normalization_params)(outputs)
    if dropout is not None:
        dropout_params = kwargs.get('dropout_params', {})
        outputs = dropout(**dropout_params)(outputs)
    if activation is not None:
        outputs = activation(outputs)

    return outputs


def _conv_block(
    inputs, 
    filters, 
    normalization, 
    dropout, 
    conv_cls, 
    activation=ReLU(), 
    reg_coef=None, 
    skip=False,
    **kwargs):

    regularizer = l2(reg_coef) if reg_coef is not None else None

    conv_outputs = conv_cls(
        filters=filters, 
        kernel_size=3, 
        padding='same', 
        kernel_regularizer=regularizer
    )(inputs)

    outputs = conv_outputs

    outputs = __nn_ops(
        outputs, normalization, dropout, activation, **kwargs)

    if skip:
        return outputs, conv_outputs

    return outputs


def _conv_block_x2(
    inputs, 
    filters, 
    conv_block_callable, 
    activation=ReLU(),
    normalization=None, 
    dropout=None, 
    reg_coef=None, 
    skip=False,
    **kwargs):

    llayer_activation = activation if skip else None
    llayer_normalization = normalization if skip else None
    llayer_dropout = dropout if skip else None

    outputs = conv_block_callable(
        inputs, filters, normalization, dropout, \
        activation=activation, reg_coef=reg_coef, skip=skip, **kwargs)

    if skip:
        outputs, conv_outputs = outputs

    outputs = conv_block_callable(
        outputs, filters, llayer_normalization, llayer_dropout, \
        activation=llayer_activation, reg_coef=reg_coef, **kwargs)

    if skip:
        outputs = Add()([outputs, conv_outputs])

        outputs = __nn_ops(
            outputs, normalization, dropout, activation, **kwargs)

    return outputs


def _conv_block_x3(
    inputs, 
    filters, 
    conv_block_callable, 
    activation=ReLU(),
    normalization=None, 
    dropout=None, 
    reg_coef=None, 
    skip=False,
    **kwargs):

    llayer_activation = activation if skip else None
    llayer_normalization = normalization if skip else None
    llayer_dropout = dropout if skip else None

    outputs = conv_block_callable(
        inputs, filters, normalization, dropout, \
        activation=activation, reg_coef=reg_coef, skip=skip, **kwargs)

    if skip:
        outputs, conv_outputs = outputs

    outputs = conv_block_callable(
        outputs, filters, normalization, dropout, \
        activation=activation, reg_coef=reg_coef, **kwargs)

    outputs = conv_block_callable(
        outputs, filters, llayer_normalization, llayer_dropout, \
        activation=llayer_activation, reg_coef=reg_coef, **kwargs)

    if skip:
        outputs = Add()([outputs, conv_outputs])

        outputs = __nn_ops(
            outputs, normalization, dropout, activation, **kwargs)
        
    return outputs


def conv_block(inputs, filters, normalization=None, dropout=None, \
    activation=ReLU(), reg_coef=None, **kwargs):

    return _conv_block(
        inputs, filters, normalization, dropout, Conv2D, \
        activation=activation, reg_coef=reg_coef, **kwargs)


def conv_transpose_block(inputs, filters, normalization=None, dropout=None, \
    activation=ReLU(), reg_coef=None, **kwargs):

    return _conv_block(
        inputs, filters, normalization, dropout, Convolution2DTranspose, \
        activation=activation, reg_coef=reg_coef, **kwargs)


def conv_block_x2(inputs, filters, normalization=None, dropout=None, \
    activation=ReLU(), reg_coef=None, skip=False, **kwargs):

    return _conv_block_x2(
        inputs, filters, conv_block, activation=activation, \
        normalization=normalization, dropout=dropout, \
        reg_coef=reg_coef, skip=skip, **kwargs)


def conv_transpose_block_x2(inputs, filters, normalization=None, dropout=None, \
    activation=ReLU(), reg_coef=None, skip=False, **kwargs):

    return _conv_block_x2(
        inputs, filters, conv_transpose_block, activation=activation, \
        normalization=normalization, dropout=dropout, \
        reg_coef=reg_coef, skip=skip, **kwargs)


def conv_block_x3(inputs, filters, normalization=None, dropout=None, \
    activation=ReLU(), reg_coef=None, skip=False, **kwargs):

    return _conv_block_x3(
        inputs, filters, conv_block, activation=activation, \
        normalization=normalization, dropout=dropout, \
        reg_coef=reg_coef, skip=skip, **kwargs)


def conv_transpose_block_x3(inputs, filters, normalization=None, dropout=None, \
    activation=ReLU(), reg_coef=None, skip=False, **kwargs):

    return _conv_block_x3(
        inputs, filters, conv_transpose_block, activation=activation, \
        normalization=normalization, dropout=dropout, \
        reg_coef=reg_coef, skip=skip, **kwargs)




    

    

    
