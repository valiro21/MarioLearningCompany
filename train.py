import random
from functools import reduce

import numpy as np
import gym
import ppaquette_gym_super_mario.wrappers
from operator import add

from consts import LEVELS

from models.neural_network import build_model, save_model, load_model


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


def learn_model(
        model,
        iterations=10,
        y=0.9,
        reward_factor=0.06,
        random_factor_chance=0.1,
        random_factor_minimum=0.00001,
        random_factor_decay=0.001,
        verbose=False,
        save_model_iteration=True):
    level = random.choice(LEVELS)
    for _ in range(iterations):
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
            if random_number < random_factor_chance:
                action = env.action_space.sample()
                action_idx = get_idx_from_action(action)

            if random_factor_chance > random_factor_minimum:
                random_factor_chance -= random_factor_decay * random_factor_chance

            observation, reward, done, info = env.step(action)
            observation = observation.reshape(((1,) + observation.shape))

            if verbose:
                print("Reward:", reward)
                print("Done:", done)
                print("Info:", info)
                print("Random factor:", random_factor_chance)

            # Update network with reward
            scores = model.predict(observation)
            max_next_score = np.max(scores)
            updated_score = reward * reward_factor + y * max_next_score

            scores[0, action_idx] = updated_score
            model.fit(
                x=observation,
                y=scores
            )

        env.close()

        if save_model_iteration:
            save_model(model)


if __name__ == '__main__':
    # mario_model = build_model()
    mario_model = load_model()
    learn_model(mario_model,
                iterations=50000,
                verbose=True,
                random_factor_chance=0.2,
                random_factor_decay=0.000001,
                random_factor_minimum=0.0001,
                save_model_iteration=True)
    save_model(mario_model)



