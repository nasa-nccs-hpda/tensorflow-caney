"""
@Reference: https://github.com/luyanger1799/amazing-semantic-segmentation
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Input, Dropout
import tensorflow.keras.backend as backend
from tensorflow.keras import layers

from tensorflow_caney.networks.Xception import Xception


class GlobalAveragePooling2D(layers.GlobalAveragePooling2D):
    def __init__(self, keep_dims=False, **kwargs):
        super(GlobalAveragePooling2D, self).__init__(**kwargs)
        self.keep_dims = keep_dims

    def call(self, inputs):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).call(inputs)
        else:
            return backend.mean(inputs, axis=[1, 2], keepdims=True)


class Concatenate(layers.Concatenate):
    def __init__(self, out_size=None, axis=-1, name=None):
        super(Concatenate, self).__init__(axis=axis, name=name)
        self.out_size = out_size

    def call(self, inputs):
        return backend.concatenate(inputs, self.axis)


def convolution_block(inputs,
                      filter_count=256,
                      kernel_size=3,
                      strides=1,
                      dilation_rate=1):
    block = Conv2D(
        filter_count,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding='same',
    )(inputs)
    block = BatchNormalization()(block)
    block = layers.ReLU()(block)
    return block


def dilated_spatial_pyramid_pooling(inputs, out_filters, dspp_size):
    out_1 = convolution_block(inputs, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(inputs, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(inputs, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(inputs, kernel_size=3, dilation_rate=18)

    conv_list = [out_1, out_6, out_12, out_18]

    img_pool = GlobalAveragePooling2D(keep_dims=True)(inputs)
    img_pool = \
        Conv2D(out_filters, 1, 1, kernel_initializer='he_normal')(img_pool)
    img_pool = UpSampling2D(size=dspp_size, interpolation='bilinear')(img_pool)

    conv_list.append(img_pool)

    x = Concatenate(out_size=dspp_size)(conv_list)
    x = Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    return x


def deeplab_v3_plus(nclass: int = 19,
                    input_size: tuple = (256, 256, 8),
                    weight_file: str = None):
    """
    TF implementation of DeepLabV3+ model.
    Currently only implements with Xception backbone.
    """
    inputs = Input(input_size)
    dilation = [1, 2]
    dspp_size = (input_size[0] // 16, input_size[1] // 16)
    enc = Xception(version='Xception-DeepLab', dilation=dilation)
    input_a, input_b = enc(inputs, output_stages=['c1', 'c5'])

    x = dilated_spatial_pyramid_pooling(input_b,
                                        input_size[0],
                                        dspp_size=dspp_size)
    x = Dropout(rate=0.5)(x)

    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = convolution_block(x, filter_count=48, kernel_size=1)

    x = Concatenate(out_size=dspp_size)([x, input_a])
    x = convolution_block(x, filter_count=256, kernel_size=3)
    x = Dropout(rate=0.5)(x)

    x = convolution_block(x, filter_count=256, kernel_size=3)
    x = Dropout(rate=0.1)(x)

    actv = 'softmax'
    if nclass == 1:
        actv = 'sigmoid'

    x = Conv2D(nclass, (1, 1), activation=actv)(x)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    model_output = x

    model = Model(inputs=inputs, outputs=model_output, name='DeepLabV3_Plus')

    if weight_file:
        model.load_weights(weight_file)
    return model
