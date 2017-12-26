from keras import Sequential, optimizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import model_from_json


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
        optimizer=optimizers.RMSprop(lr=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def save_model(model_to_save,
               model_file='model.json',
               weights_file='model.h5'):
    # serialize model to JSON
    model_json = model_to_save.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model_to_save.save_weights(weights_file)


def load_model(model_file="./model.json",
               weights_file="./model.h5"):

    # load json and create model
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_file)

    loaded_model.compile(
        optimizer=optimizers.RMSprop(lr=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return loaded_model
