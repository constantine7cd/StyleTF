import tensorflow as tf

from tensorflow.keras.layers import Dense


def linear_block(
    inputs, units, norm=None, activation=None, \
    norm_kwargs={}, activation_kwargs={}):

    outputs = Dense(units)(inputs)

    if norm is not None:
        outputs = norm(**norm_kwargs)(outputs)

    if activation is not None:
        outputs = activation(**activation_kwargs)(outputs)
     
    return outputs