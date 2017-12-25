from keras import Sequential, optimizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


def build_model():
    # Input size: 224x256x3
    model = Sequential()
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(5, 5),
            input_shape=(224, 256, 3),
            padding="same",
            data_format='channels_last',
            activation='relu'
        )
    )
    model.add(
        MaxPooling2D(
            pool_size=(4, 4)
        )
    )

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(5, 5),
            input_shape=(56, 64, 32),
            padding="same",
            data_format='channels_last',
            activation='relu'
        )
    )
    model.add(
        MaxPooling2D(
            pool_size=(4, 4)
        )
    )

    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            input_shape=(14, 16, 64),
            padding="same",
            data_format='channels_last',
            activation='relu'
        )
    )
    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    model.add(
        Conv2D(
            filters=256,
            kernel_size=(5, 5),
            input_shape=(7, 8, 128),
            padding="same",
            data_format='channels_last',
            activation='relu'
        )
    )
    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    model.add(
        Flatten()
    )

    model.add(
        Dense(
            units=200,
            activation='relu'
        )
    )

    model.add(
        Dropout(
            0.6
        )
    )

    model.add(
        Dense(
            units=64,
            activation='sigmoid'
        )
    )

    model.compile(
        optimizer=optimizers.RMSprop(lr=0.06),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
