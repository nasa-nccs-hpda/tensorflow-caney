from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate, UpSampling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers, mixed_precision
from tensorflow.keras import backend as K


def unet_batchnorm_regression(
            nclass=1,
            input_size=(256, 256, 8),
            weight_file=None,
            kr=l2(0.0001),
            maps=[64, 128, 256, 512, 1024]
        ):
    """
    UNet network using batch normalization features.
    """
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c1)
    n1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(n1)

    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c2)
    n2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(n2)

    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c3)
    n3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(n3)

    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c4)
    n4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(n4)

    # Squeeze
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    n6 = BatchNormalization()(u6)
    u6 = concatenate([n6, n4])
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    n7 = BatchNormalization()(u7)
    u7 = concatenate([n7, n3])
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    n8 = BatchNormalization()(u8)
    u8 = concatenate([n8, n2])
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    n9 = BatchNormalization()(u9)
    u9 = concatenate([n9, n1], axis=3)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c9)

    actv = 'relu'  # 'relu' #  'softmax'
    # if nclass == 1:
    #    actv = 'sigmoid'

    c10 = Conv2D(nclass, (1, 1), activation=actv, kernel_regularizer=kr)(c9)
    # c10 = Conv2D(nclass, (1, 1))(c9)
    # model = Model(inputs=inputs, outputs=c10, name="UNetBatchNorm")
    model = Model(inputs=inputs, outputs=c10, name="UNetBatchNormRegression")

    if weight_file:
        model.load_weights(weight_file)
    return model


def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(
        num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder


def get_model_new():
    inputs = layers.Input(shape=[64, 64, 2])  # 256
    encoder0_pool, encoder0 = encoder_block(inputs, 32)  # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)  # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)  # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)  # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)  # 8
    center = conv_block(encoder4_pool, 1024)  # center
    decoder4 = decoder_block(center, encoder4, 512)  # 16
    decoder3 = decoder_block(decoder4, encoder3, 256)  # 32
    decoder2 = decoder_block(decoder3, encoder2, 128)  # 64
    decoder1 = decoder_block(decoder2, encoder1, 64)  # 128
    decoder0 = decoder_block(decoder1, encoder0, 32)  # 256
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


def attention_unet_regression(
            input_shape,
            input_label_channel_count: int,
            layer_count=64,
            regularizers=regularizers.l2(0.0001),
            weight_file=None,
            summary=False,
            countbranch=False
        ):
    """ Method to declare the UNet model.
    Args:
        input_shape: tuple(int, int, int, int)
            Shape of the input in the format (batch, height, width, channels).
        input_label_channel_count: int
            index count of label channels, used for calculating the
            number of channels in model output.
        layer_count: (int, optional)
            Count of kernels in first layer. Number of kernels in other
            layers grows with a fixed factor.
        regularizers: keras.regularizers
            regularizers to use in each layer.
        weight_file: str
            path to the weight file.
        summary: bool
            Whether to print the model summary
    """

    # NOTE(Jesse): Use "mixed_bfloat16" string for TPUs
    mixed_precision.set_global_policy("mixed_bfloat16")

    input_img = layers.Input(input_shape[1:], dtype="float32", name='Input')
    pp_in_layer = input_img

    c1 = layers.Conv2D(
        1 * layer_count, (3, 3), activation='relu',
        padding='same')(pp_in_layer)
    c1 = layers.Conv2D(
        1 * layer_count, (3, 3), activation='relu', padding='same')(c1)
    n1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(n1)

    c2 = layers.Conv2D(
        2 * layer_count, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(
        2 * layer_count, (3, 3), activation='relu', padding='same')(c2)
    n2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(n2)

    c3 = layers.Conv2D(
        4 * layer_count, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(
        4 * layer_count, (3, 3), activation='relu', padding='same')(c3)
    n3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(n3)

    c4 = layers.Conv2D(
        8 * layer_count, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(
        8 * layer_count, (3, 3), activation='relu', padding='same')(c4)
    n4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(n4)

    c5 = layers.Conv2D(
        16 * layer_count, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(
        16 * layer_count, (3, 3), activation='relu', padding='same')(c5)
    n5 = layers.BatchNormalization()(c5)

    u6 = attention_up_and_concat(n5, n4)
    c6 = layers.Conv2D(
        8 * layer_count, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(
        8 * layer_count, (3, 3), activation='relu', padding='same')(c6)
    n6 = layers.BatchNormalization()(c6)

    u7 = attention_up_and_concat(n6, n3)
    c7 = layers.Conv2D(
        4 * layer_count, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(
        4 * layer_count, (3, 3), activation='relu', padding='same')(c7)
    n7 = layers.BatchNormalization()(c7)

    u8 = attention_up_and_concat(n7, n2)
    c8 = layers.Conv2D(
        2 * layer_count, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(
        2 * layer_count, (3, 3), activation='relu', padding='same')(c8)
    n8 = layers.BatchNormalization()(c8)

    u9 = attention_up_and_concat(n8, n1)
    c9 = layers.Conv2D(
        1 * layer_count, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(
        1 * layer_count, (3, 3), activation='relu', padding='same')(c9)
    n9 = layers.BatchNormalization()(c9)

# =============================================================================
    # density map
    if countbranch:
        d = layers.Conv2D(
            input_label_channel_count, (1, 1), activation='sigmoid',
            kernel_regularizer=regularizers, name='output_seg')(n9)

        d2 = layers.Conv2D(
            input_label_channel_count, (1, 1), activation='linear',
            dtype='float32', kernel_regularizer=regularizers,
            name='output_dens')(n9)

        seg_model = models.Model(inputs=[input_img], outputs=[d, d2])
# =============================================================================
    else:
        d = layers.Conv2D(
            input_label_channel_count, (1, 1), activation='sigmoid',
            dtype='float32', kernel_regularizer=regularizers)(n9)
        seg_model = models.Model(inputs=[input_img], outputs=[d])
    if weight_file:
        seg_model.load_weights(weight_file)
    if summary:
        seg_model.summary()

    return seg_model


def attention_up_and_concat(down_layer, layer):
    in_channel = down_layer.get_shape().as_list()[3]
    up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(down_layer)
    layer = attention_block_2d(
        x=layer, g=up, inter_channel=in_channel // 4)
    my_concat = layers.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    concat = my_concat([up, layer])

    return concat


def attention_block_2d(x, g, inter_channel):
    theta_x = layers.Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
    phi_g = layers.Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
    psi_f = layers.Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = layers.Activation('sigmoid')(psi_f)
    att_x = layers.multiply([x, rate])

    return att_x


if __name__ == "__main__":

    # Can add different images sizes - for now: (256,256,6)
    simple_unet = unet_batchnorm_regression(
            nclass=1,
            input_size=(32, 32, 4),
            maps=[64, 128, 256, 512, 1024]
    )
    simple_unet.summary()
