import tensorflow as tf


def reflect_padding_2d(inputs, padding):
    paddings = tf.constant(
        [[0, 0], [padding, padding], [padding, padding], [0, 0]])
    outputs = tf.pad(inputs, paddings, 'REFLECT')

    return outputs