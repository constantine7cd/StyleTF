from collections import OrderedDict

import tensorflow as tf

from tensorflow.keras.layers import ReLU, LayerNormalization
from tensorflow.keras.layers import UpSampling2D

from .adain import AdaptiveInstanceNormalization
from .linear_blocks import linear_block
from .conv_blocks import res_block
from .conv_blocks import conv_block
from .conv_blocks import res_block_adain
from .conv_blocks import conv_block_adain
from .norm import InstanceNorm, AdaptiveInstanceNorm, LayerNorm


def _adain_params_iter(adain_config, adain_params):
    slices = []
    curr_slice = 0

    for _, dim in adain_config.items():
        slices.append(adain_params[:, curr_slice: curr_slice + dim])
        curr_slice += dim
    return slices.__iter__()


def _get_adain_layer_params(adain_params, slices_iter):
    beta_left, beta_right = next(slices_iter)
    gamma_left, gamma_right = next(slices_iter)

    beta = adain_params[:, beta_left:beta_right]
    gamma = adain_params[:, gamma_left:gamma_right]

    return gamma, beta


def _adain_net(inputs, dim=64, output_dim=3860):
    outputs = linear_block(inputs, dim, activation=ReLU)
    outputs = linear_block(outputs, dim, activation=ReLU)
    outputs = linear_block(outputs, output_dim, activation=None)

    return outputs


def _body(inputs, adain_params_iter, num_res_blocks, dim):
    norm = AdaptiveInstanceNorm
    output = inputs
    for _ in range(num_res_blocks):
        gamma1, beta1 = next(adain_params_iter), next(adain_params_iter)
        gamma2, beta2 = next(adain_params_iter), next(adain_params_iter)

        res_block_inputs = (output, gamma1, beta1, gamma2, beta2)

        output = res_block_adain(res_block_inputs, dim, norm = norm)

    return output, adain_params_iter


def _upsample_postprocess(inputs, skip_tensors, adain_params_iter, skip_dim=5, dim = 192):
    outputs = inputs
    norm = AdaptiveInstanceNorm

    for skip_tensor in skip_tensors:
        outputs = UpSampling2D(interpolation = 'bilinear')(outputs)
        gamma, beta = next(adain_params_iter), next(adain_params_iter)
        
        skip_outputs = conv_block_adain(
            skip_tensor, gamma, beta, filters = skip_dim, kernel_size = 7, padding = 3, stride = 1, \
            norm = norm, activation = ReLU)


        print(outputs.shape, skip_outputs.shape)  
        outputs = tf.concat([outputs, skip_outputs], -1)
        print(outputs.shape)
        outputs = conv_block( outputs, filters = dim // 2, kernel_size = 7, \
                            padding=3, stride=1, norm=LayerNorm, activation=ReLU, \
                            norm_kwargs={}, activation_kwargs={})
        dim //= 2
    
    outputs = conv_block( outputs, filters = 6, kernel_size = 9, \
                            padding=4, stride=1, norm=None, activation=None, \
                            norm_kwargs={}, activation_kwargs={})
    return outputs

def decoder(
    content_input, skip_tensors, style_input, adain_config, \
    num_upsamples=2, num_res_blocks=5, dim=192):

    adain_params = _adain_net(style_input)
    adain_params_iter = _adain_params_iter(adain_config, adain_params)

    outputs, adain_params_iter = _body(
        content_input, adain_params_iter, num_res_blocks, dim)

    outputs = _upsample_postprocess(outputs, skip_tensors, adain_params_iter, dim = dim)

    return outputs


def Decoder(
    input_shape=(64, 64, 192),
    skip2_shape=(128, 128, 5),
    skip1_shape=(256, 256, 5), 
    style_shape=(3,),
    num_res_blocks=5, dim=192, 
    num_upsamples=2, 
    skip_dim=5):

    adain_config = OrderedDict()

    for i in range(num_res_blocks * 2):
        adain_config['res_block_{}_beta'.format(i)] = dim
        adain_config['res_block_{}_gamma'.format(i)] = dim

    for i in range(num_upsamples):
        adain_config['upsample_block_{}_beta'.format(i)] = skip_dim
        adain_config['upsample_block_{}_gamma'.format(i)] = skip_dim



    content_inputs = tf.keras.Input(input_shape)
    skip2_inputs = tf.keras.Input(skip2_shape)
    skip1_inputs = tf.keras.Input(skip1_shape)
    style_inputs = tf.keras.Input(style_shape)

    outputs = decoder(
        content_inputs, [skip2_inputs, skip1_inputs], style_inputs, adain_config, \
        num_upsamples=num_upsamples, num_res_blocks=num_res_blocks, dim=dim)

    print(outputs.shape)

    model = tf.keras.models.Model(inputs=[content_inputs, skip2_inputs, skip1_inputs, style_inputs], outputs=outputs)

    return model

    


