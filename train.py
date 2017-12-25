import random
from functools import reduce

import numpy as np
import gym
import ppaquette_gym_super_mario.wrappers
from operator import add
from consts import *

from models.neural_network import build_model


def get_action_from_idx(idx):
    return list(
        map(
            lambda x: ord(x) - 48,
            '{0:06b}'.format(idx)
        )
    )


def get_idx_from_action(act):
    return reduce(
        add,
        map(
            lambda x: x[1] * (2 ** (len(act) - x[0] - 1)),
            enumerate(act))
    )


def save_model(model_to_save,
               model_file='model.json',
               weights_file='model.h5'):
    # serialize model to JSON
    model_json = model_to_save.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model_to_save.save_weights(weights_file)


_ITERATIONS = 10

if __name__ == '__main__':
    model = build_model()

    for _ in range(_ITERATIONS):
        level = random.choice(LEVELS)
        env = gym.make(level)
        env.reset()
        done = False
        observation = np.zeros(shape=(1, 224, 256, 3))

        env.render()
        while not done:
            scores = model.predict(observation)
            action_idx = np.argmax(scores)
            action = get_action_from_idx(action_idx)

            random_number = random.uniform(0.0, 1.0)
            if random_number > RANDOM_ACTION_THRESHOLD:
                action = env.action_space.sample()
                action_idx = get_idx_from_action(action)

            observation, reward, done, info = env.step(action)
            observation = observation.reshape(((1,) + observation.shape))
            # Update network with reward
            scores = model.predict(observation)
            max_next_score = np.max(scores)
            updated_score = reward * REWARD_NORMALIZATION_FACTOR + Y * max_next_score

            scores[0, action_idx] = updated_score
            model.fit(
                x=observation,
                y=scores
            )

        env.close()

        save_model(model)
