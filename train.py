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


class Memory(object):
    def __init__(self, max_memory_steps, alpha=0.2, gamma=0.7):
        self.max_memory_steps = max_memory_steps
        self._memory_idx = 0
        self.actions = [-1] * max_memory_steps
        self.rewards = [-1] * max_memory_steps
        self.alpha = alpha
        self.gamma = gamma
        self.observations = None

    def _initialize(self, game_image_shape):
        self.observations = np.zeros(shape=((self.max_memory_steps,) + game_image_shape[1:]))

    def add(self, observation, reward, action_idx):
        if self.observations is None:
            self._initialize(observation.shape)

        self.observations[self._memory_idx] = observation
        self.actions[self._memory_idx] = action_idx
        self.rewards[self._memory_idx] = reward

        self._memory_idx = (self._memory_idx + 1) % self.max_memory_steps

    def train(self, model, train_epochs=5, train_batch_size=1):
        y = model.predict(self.observations)
        for idx in range(self.max_memory_steps):
            if self.actions[idx] != -1:
                next_score = 0
                if idx != self._memory_idx:
                    next_score = np.max(y[(idx + 1) % self.max_memory_steps])
                old_score = y[idx, self.actions[idx]]
                updated_score = (1 - self.alpha) * old_score + self.alpha * (self.rewards[idx] + self.gamma * next_score)
                y[idx, self.actions[idx]] = updated_score

        model.fit(
            x=self.observations,
            y=y,
            epochs=train_epochs,
            batch_size=train_batch_size
        )


def learn_model(
        model,
        iterations=10,
        gamma=0.7,
        alpha=0.2,
        train_epochs=1,
        train_batch_size=1,
        random_factor_chance=0.1,
        random_factor_minimum=0.00001,
        random_factor_decay=0.001,
        max_memory_steps=50,
        observe_steps=0,
        verbose=False,
        save_model_iteration=True):

    level = random.choice(LEVELS)
    for _ in range(iterations):
        env = gym.make(level)
        env.reset()
        done = False

        env.render()
        scores = [0] * 64
        memory = Memory(
            max_memory_steps,
            alpha=alpha,
            gamma=gamma
        )

        step = 0
        while not done:
            random_number = random.uniform(0.0, 1.0)
            if random_number < random_factor_chance:
                action = env.action_space.sample()
                action_idx = get_idx_from_action(action)
            else:
                action_idx = np.argmax(scores)
                action = get_action_from_idx(action_idx)

            if random_factor_chance > random_factor_minimum:
                random_factor_chance -= random_factor_decay * random_factor_chance

            observation, reward, done, info = env.step(action)
            observation = observation.reshape(((1,) + observation.shape))
            observation = observation / 255.

            if info['life'] == 0:
                reward = -1000
            reward -= 1

            if verbose:
                print("Reward:", reward)
                print("Done:", done)
                print("Info:", info)
                print("Random factor:", random_factor_chance)

            # Update network with reward
            scores = model.predict(observation)

            memory.add(observation, reward, action_idx)

            if done or step >= observe_steps:
                memory.train(model,
                             train_epochs=train_epochs,
                             train_batch_size=train_batch_size)

            step += 1

        env.close()

        if save_model_iteration:
            save_model(model)


if __name__ == '__main__':
    # mario_model = build_model()
    mario_model = load_model()
    learn_model(mario_model,
                iterations=50000,
                verbose=True,
                alpha=0.4,
                train_batch_size=50,
                train_epochs=1,
                random_factor_chance=0.1,
                random_factor_decay=0.0001,
                random_factor_minimum=0.0001,
                save_model_iteration=True)
    save_model(mario_model)



