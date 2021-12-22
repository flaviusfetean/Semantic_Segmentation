from tensorflow.keras import layers
from tensorflow.keras import Model


def define_conv(layer, num_filters, kernel_size, dropout=0.0):
    layer = layers.BatchNormalization()(layer)
    layer = layers.Conv2D(num_filters, kernel_size, padding='same',
                          kernel_initializer='he_uniform')(layer)
    if dropout > 0.0:
        layer = layers.Dropout(dropout)(layer)
    return layer


def define_dense(layer, growth_rate, num_layers, kernel_size=(3, 3),
                 add_bottleneck=False, dropout=0.0):
    block_layers = []
    for i in range(num_layers):
        if add_bottleneck:
            new_layer = define_conv(layer, 4*growth_rate, (1, 1), dropout)
        else:
            new_layer = layer

        new_layer = define_conv(new_layer, growth_rate, kernel_size, dropout)
        block_layers.append(new_layer)
        layer = layers.Concatenate()([layer, new_layer])

    return layer, layers.Concatenate()(block_layers)


def define_downscale(layer, compression_factor=1.0, dropout=0.0):
    num_filters_compressed = int(layer.shape[-1] * compression_factor)
    layer = define_conv(layer, num_filters_compressed, (1, 1), dropout)
    return layers.AveragePooling2D((2, 2), (2, 2), padding='same')(layer)


def define_upscale(layer, skip_connection):
    num_filters = int(layer.shape[-1])
    layer = layers.Conv2DTranspose(num_filters, kernel_size=(3, 3), strides=(2, 2),
                                   padding='same', kernel_initializer='he_uniform')(layer)
    return layers.Concatenate()([layer, skip_connection])


def define_model(input_shape, num_classes, growth_rate=12, num_dense_layers=4, add_bottleneck=False,
                 dropout=0.0, compression_factor=1.0, num_groups=5):
    input = layers.Input(input_shape)
    layer = layers.Conv2D(32, (7, 7), padding='same')(input)

    skip_connections = []

    for idx in range(num_groups):
        layer, _ = define_dense(layer, growth_rate, num_dense_layers,
                                add_bottleneck=add_bottleneck, dropout=dropout)
        skip_connections.append(layer)
        layer = define_downscale(layer, compression_factor, dropout)

    skip_connections.reverse()

    (layer, block_layers) = define_dense(layer, growth_rate, num_dense_layers,
                                         add_bottleneck=add_bottleneck, dropout=dropout)

    for idx in range(num_groups):
        layer = define_upscale(block_layers, skip_connections[idx])
        (layer, block_layers) = define_dense(layer, growth_rate, num_dense_layers,
                                             add_bottleneck=add_bottleneck, dropout=dropout)

    layer = layers.Conv2D(num_classes, (1, 1), padding='same',
                          kernel_initializer='he_uniform')(layer)
    output = layers.Softmax()(layer)

    return Model(input, output)