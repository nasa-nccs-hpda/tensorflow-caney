import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Conv2DTranspose
from tensorflow.keras.layers import concatenate, Input, UpSampling2D, add
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Multiply
from tensorflow.keras.regularizers import l2

# ---------------------------------------------------------------------------
# module unet
#
# Build UNet NN architecture using Keras. Any of these functions can be
# called from an external script. UNets can be modified as needed.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module Methods
# ---------------------------------------------------------------------------


# --------------------------- Convolution Functions ----------------------- #

def unet_dropout(nclass=19, input_size=(256, 256, 8), weight_file=None,
                 maps=[64, 128, 256, 512, 1024]
                 ):
    """
    UNet network using dropout features.
    """
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c4)
    d4 = Dropout(0.5)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(d4)

    # Squeeze
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(c5)
    d5 = Dropout(0.5)(c5)

    # Decoder
    u6 = UpSampling2D(size=(2, 2))(d5)
    m6 = concatenate([d4, u6], axis=3)
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(m6)
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D(size=(2, 2))(c6)
    m7 = concatenate([c3, u7], axis=3)
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(m7)
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D(size=(2, 2))(c7)
    m8 = concatenate([c2, u8], axis=3)
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(m8)
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D(size=(2, 2))(c8)
    m9 = concatenate([c1, u9], axis=3)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(m9)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c9)

    actv = 'softmax'
    if nclass == 1:
        actv = 'sigmoid'

    conv10 = Conv2D(nclass, (1, 1), activation=actv)(c9)
    model = Model(inputs=inputs, outputs=conv10, name="UNetDropout")

    if weight_file:
        model.load_weights(weight_file)
    return model


def unet_batchnorm(nclass=19, input_size=(256, 256, 8), weight_file=None,
                   kr=l2(0.0001), maps=[64, 128, 256, 512, 1024]
                   ):
    """
    UNet network using batch normalization features.
    """
    inputs = Input(input_size, name='Input')

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

    actv = 'softmax'
    if nclass == 1:
        actv = 'sigmoid'

    c10 = Conv2D(nclass, (1, 1), kernel_regularizer=kr)(c9)
    outputs = Activation(actv, dtype='float32', name='predictions')(c10)

    model = Model(inputs=inputs, outputs=outputs, name="UNetBatchNorm")

    if weight_file:
        model.load_weights(weight_file).expect_partial()
    return model


def se_block(input, channels, r=8):
    x = GlobalAveragePooling2D()(input)
    x = Dense(channels//r, activation="relu", kernel_initializer='Ones')(x)
    x = Dense(channels, activation="sigmoid", kernel_initializer='Ones')(x)
    return Multiply()([input, x])


def add_encoding_layer(filter_count, sequence, ds):

    # Residual part
    res_sequence = sequence

    res_sequence = BatchNormalization()(res_sequence)
    res_sequence = Activation(activation='relu')(res_sequence)
    res_sequence = Conv2D(
        filter_count, 3, strides=1, padding="same",
        kernel_initializer='he_uniform')(res_sequence)

    res_sequence = BatchNormalization()(res_sequence)
    res_sequence = Activation(activation='relu')(res_sequence)
    res_sequence = Conv2D(
        filter_count, 3, strides=1, padding="same",
        kernel_initializer='he_uniform')(res_sequence)

    # SE block
    res_sequence = se_block(res_sequence, filter_count)

    # shortcut part
    shortcut_sequence = sequence
    # 1x1 projection
    shortcut_sequence = Conv2D(
        filter_count, 1, strides=1, padding="same")(shortcut_sequence)

    # add & export
    add_sequence = add([res_sequence, shortcut_sequence])
    add_sequence = Activation(activation='relu')(add_sequence)

    if ds:
        # Downsampling with stride
        new_sequence = Conv2D(
            filter_count, 2, strides=2, padding="same")(add_sequence)
    else:
        new_sequence = Conv2D(
            filter_count, 1, strides=1, padding="same")(add_sequence)
    return new_sequence, add_sequence


def add_decoding_layer(filter_count, add_drop_layer, sequence, res_enc, us):

    # Residual part
    res_sequence = sequence

    # import & concatenate
    res_sequence = concatenate([res_sequence, res_enc], axis=-1)

    res_sequence = BatchNormalization()(res_sequence)
    res_sequence = Activation(activation='relu')(res_sequence)
    res_sequence = Conv2D(
        int(filter_count*2), 3, strides=1, padding="same",
        kernel_initializer='he_uniform')(res_sequence)

    # In original papre, kernel size set to be  2, but in the author's
    # github, the kernel size = 3.
    res_sequence = BatchNormalization()(res_sequence)
    res_sequence = Activation(activation='relu')(res_sequence)
    res_sequence = Conv2D(
        filter_count, 3, strides=1, padding="same",
        kernel_initializer='he_uniform')(res_sequence)

    # se
    res_sequence = se_block(res_sequence, filter_count)

    # shortcut part
    shortcut_sequence = sequence

    # 1x1 projection
    shortcut_sequence = Conv2D(
        filter_count, 1, strides=1, padding="same")(shortcut_sequence)

    # add
    add_sequence = add([res_sequence, shortcut_sequence])
    add_sequence = Activation(activation='relu')(add_sequence)

    # Dropout?
    if add_drop_layer:
        add_sequence = Dropout(0.2)(add_sequence)

    if us:
        # Replacing Upsampling with deconvolution, better ??
        new_sequence = Conv2DTranspose(
            filter_count, 2, strides=2, padding="same",
            kernel_initializer='he_uniform')(add_sequence)

        # Simple upsampling
        # new_sequence = UpSampling2D(size=(2,2))(add_sequence)
    else:
        new_sequence = Conv2D(
            filter_count, 1, strides=1, padding="same")(add_sequence)
    return new_sequence


def deep_unet(
            nclass=1,
            input_size=(256, 256, 8),
            first_layer_filter_count=32
        ):
    # Referece: DeepUNet: A Deep Fully Convolutional Network for
    # Pixel-level Sea-Land Segmentation, Li et al., 2017

    inputs = Input(input_size, name='Input')

    filter_count = first_layer_filter_count  # 32

    # Encoder part:
    enc1, res_enc1 = add_encoding_layer(
        filter_count, inputs, True)  # 256 => 128
    enc2, res_enc2 = add_encoding_layer(
        filter_count, enc1, True)  # 128 =>  64
    enc3, res_enc3 = add_encoding_layer(
        filter_count*1, enc2, True)  # 64 =>  32
    enc4, res_enc4 = add_encoding_layer(
        filter_count*2, enc3, True)  # 32 =>  16
    enc5, res_enc5 = add_encoding_layer(
        filter_count*4, enc4, True)  # 16 =>   8
    enc6, res_enc6 = add_encoding_layer(
        filter_count*8, enc5, True)  # 8 =>   4
    enc7, res_enc7 = add_encoding_layer(
        filter_count*16, enc6, False)  # 4 =>   4
    enc8, res_enc8 = add_encoding_layer(
        filter_count*32, enc7, False)  # 4 =>   4

    # Decoder part:
    dec1 = add_decoding_layer(
        filter_count*32, True, enc8, res_enc8, False)  # 4 => 4
    dec2 = add_decoding_layer(
        filter_count*16, True, dec1, res_enc7, True)  # 4 => 8
    dec3 = add_decoding_layer(
        filter_count*8, True, dec2, res_enc6, True)  # 8 => 16
    dec4 = add_decoding_layer(
        filter_count*4, True, dec3, res_enc5, True)  # 16 => 32
    dec5 = add_decoding_layer(
        filter_count*2, True, dec4, res_enc4, True)  # 32 => 64
    dec6 = add_decoding_layer(
        filter_count*1, True, dec5, res_enc3, True)  # 64 => 128
    dec7 = add_decoding_layer(
        filter_count, True, dec6, res_enc2, True)  # 128 => 256

    # Output layer with softmax or sigmoid activation :
    # This layer is simpler than original in the reference
    dec8 = concatenate([dec7, res_enc1], axis=-1)
    dec8 = Conv2D(
        filter_count, 3, strides=1, padding="same",
        kernel_initializer='he_uniform')(dec8)
    dec8 = BatchNormalization()(dec8)
    dec8 = Activation(activation='relu')(dec8)

    dec8 = Conv2D(nclass, 1, strides=1, padding="same")(dec8)
    dec8 = Activation(activation='sigmoid')(dec8)

    model = Model(inputs=inputs, outputs=dec8)
    return model


# -------------------------------------------------------------------------------
# module unet Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Can add different images sizes - for now: (256,256,6)
    simple_unet = unet_dropout()
    simple_unet.summary()

    # Batch Normalization UNet
    simple_unet = unet_batchnorm()
    simple_unet.summary()

    # DeepUNet
    simple_unet = deep_unet()
    simple_unet.summary()
