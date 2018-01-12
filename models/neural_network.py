from keras import Sequential, optimizers, Input, Model
from keras.layers import Conv2D, Dense, Flatten, Merge, BatchNormalization, Reshape, MaxPooling2D, Dropout
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16


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
    
    model.add(
        Conv2D(
            filters=32,
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
            pool_size=(2, 2),
            data_format='channels_first'
        )
    )

    model.add(Flatten())

    return model


def build_frame_model():
    model = Sequential()
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            input_shape=(3, 224, 224),
            strides=(1, 1),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )
    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            data_format='channels_first'
        )
    )

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
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
            strides=(2, 2),
            data_format='channels_first'
        )
    )
    
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            input_shape=(32, 16, 16),
            strides=(2, 2),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )
    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            data_format='channels_first'
        )
    )

    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            input_shape=(32, 16, 16),
            strides=(2, 2),
            padding="same",
            data_format='channels_first',
            activation='relu'
        )
    )
    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            data_format='channels_first'
        )
    )
    model.add(Flatten())

    return model


def build_frame_model_VGG16():
    # frame_model = VGG16(weights='imagenet',
    #                     include_top=False,
    #                     input_shape=(224, 224, 3))
    model = Sequential()
    model.add(
        Conv2D(
            filters=32, 
            kernel_size=(3, 3),
            input_shape=(224, 224, 3),
            padding="same",
            strides=(1, 1),
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=32, 
            kernel_size=(3, 3),
            input_shape=(224, 224, 3),
            padding="same",
            strides=(1, 1),
            activation='relu'
        )
    )
    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )

    model.add(
        Conv2D(
            filters=64, 
            kernel_size=(3, 3),
            padding="same",
            strides=(1, 1),
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=64, 
            kernel_size=(3, 3),
            padding="same",
            strides=(2, 2),
            activation='relu'
        )
    )
    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )
     
    model.add(Flatten())

    # flattened_model = x = Flatten()(frame_model.output)
    # return Model(input=frame_model.input, output=flattened_model)
    return model

def build_model(frame_history_size=2, actions_history_size=4, learning_rate=0.001):
    action_model = build_actions_model(history_size=actions_history_size)
    history_model = build_history_model(history_size=frame_history_size)
    frame_model = build_frame_model()

    model = Sequential()

    model.add(
        Merge([action_model, history_model, frame_model],
              mode='concat')
    )

    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=14, use_bias=False))

    model.compile(
        optimizer=optimizers.SGD(lr=learning_rate, momentum=0.7, nesterov=True),
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
        optimizer=optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.7, nesterov=True),
        loss='mse',
        metrics=['accuracy']
    )

    return loaded_model
