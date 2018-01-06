import keras
import tensorflow
from keras import Sequential, optimizers
from keras.layers import Conv2D, Dense, Flatten, Dropout, SpatialDropout2D, Merge, Concatenate
from keras.models import model_from_json


def build_model(history_size=4):
    # Input size: 1x224x256
    action_history = Sequential()
    action_history.add(
        Dense(
            input_shape=(history_size*6,),
            units=history_size*2
        )
    )

    frame = Sequential()
    frame.add(
        Conv2D(
            filters=16,
            kernel_size=(8, 8),
            input_shape=(1, 84, 84),
            strides=(4, 4),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )

    frame.add(
        SpatialDropout2D(
            0.1
        )
    )

    frame.add(
        Conv2D(
            filters=32,
            kernel_size=(4, 4),
            input_shape=(16, 21, 21),
            strides=(2, 2),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )

    frame.add(
        SpatialDropout2D(
            0.1
        )
    )

    frame.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            input_shape=(32, 16, 16),
            data_format='channels_first',
            activation='relu'
        )
    )

    frame.add(
        Flatten()
    )

    model = Sequential()

    model.add(
        Merge([action_history, frame], mode='concat')
    )

    model.add(
        Dropout(
            0.2
        )
    )

    model.add(
        Dense(
            units=256,
            kernel_initializer='random_uniform',
            activation='relu'
        )
    )

    model.add(
        Dropout(
            0.2
        )
    )

    model.add(
        Dense(
            units=14,
            kernel_initializer='random_uniform'
        )
    )

    model.compile(
        optimizer=optimizers.RMSprop(lr=0.0001),
        loss='mse',
        metrics=['accuracy']
    )

    model._make_predict_function()

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
        optimizer=optimizers.RMSprop(lr=0.0001),
        loss='mse',
        metrics=['accuracy']
    )

    return loaded_model
