import tensorflow as tf


class AdaIN(tf.keras.layers.Layer):
    def __init__(self, input_axes=[1, 2], style_axes=-1,  eps=1e-6):
        super(AdaIN, self).__init__()

        self.input_axes = input_axes
        self.style_axes = style_axes
        self.eps = eps

    def call(self, inputs, style):
        with tf.name_scope("AdaIn") as _:
            inputs_mean, inputs_var = tf.nn.moments(inputs, axes=self.input_axes)
            style_mean, style_var = tf.nn.moments(style, axes=self.style_axes)

            inputs_normalized = (inputs - inputs_mean) / (inputs_var + self.eps)   

            return style_var * inputs_normalized + style_mean
    