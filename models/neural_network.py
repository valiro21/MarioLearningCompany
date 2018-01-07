from keras import Sequential, optimizers, Input
from keras.layers import Conv2D, Dense, Flatten, Merge, BatchNormalization, Reshape, MaxPooling2D, Dropout
from keras.models import model_from_json


def build_actions_model(history_size=4):
    model = Sequential()
    model.add(Reshape((history_size*6,), input_shape=(history_size*6,)))
    return model


def build_history_model(history_size=4):
    model = Sequential()

    model.add(
        Conv2D(
            filters=16,
            kernel_size=(3, 3),
            input_shape=(history_size, 32, 32),
            strides=(1, 1),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )
    model.add(BatchNormalization())

    model.add(
        MaxPooling2D(
            pool_size=(4, 4),
            data_format='channels_first'
        )
    )

    model.add(Flatten())

    return model


def build_frame_model():
    model = Sequential()
    model.add(
        Conv2D(
            filters=16,
            kernel_size=(3, 3),
            input_shape=(3, 128, 128),
            strides=(1, 1),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )
    model.add(BatchNormalization())

    model.add(
        MaxPooling2D(
            pool_size=(4, 4),
            data_format='channels_first'
        )
    )

    model.add(
        Conv2D(
            filters=32,
            kernel_size=(5, 5),
            input_shape=(16, 32, 32),
            strides=(1, 1),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            data_format='channels_first'
        )
    )

    model.add(Flatten())

    return model


def build_model(history_size=4, learning_rate=0.001):
    action_model = build_actions_model(history_size=history_size)
    history_model = build_history_model(history_size=history_size)
    frame_model = build_frame_model()

    model = Sequential()

    model.add(
        Merge([action_model, history_model, frame_model],
              mode='concat')
    )

    model.add(Dropout(0.4))
    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(units=14, use_bias=False))

    model.compile(
        optimizer=optimizers.RMSprop(lr=learning_rate),
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
               weights_file="./model.h5",
               learning_rate=0.001):

    # load json and create model
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_file)

    loaded_model.compile(
        optimizer=optimizers.RMSprop(lr=learning_rate),
        loss='mse',
        metrics=['accuracy']
    )

    return loaded_model
