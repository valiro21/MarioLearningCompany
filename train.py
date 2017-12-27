import random
from collections import Counter
from functools import reduce

import numpy as np
import gym
from termcolor import cprint
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


class ExperienceReplay(object):
    def __init__(self, size=100, alpha=0.2,
                 gamma=0.7, sample_size=5,
                 train_epochs=1, batch_size=5,
                 queue_behaviour=True):
        self.size = size
        self._memory_idx = 0
        self.alpha = alpha
        self.gamma = gamma
        self.observations = None
        self._queue_behaviour = queue_behaviour
        self._full = False
        self.sample_size = sample_size
        self.model_train_epochs = train_epochs
        self.model_batch_size = batch_size

        self.states = None
        self.next_states = None
        self.actions = [-1] * size
        self.rewards = [0] * size
        self.is_next_final_state = [False] * size

    def is_full(self):
        return self._full

    def _initialize(self, game_image_shape):
        self.states = np.zeros(shape=((self.size,) + game_image_shape[1:]))
        self.next_states = np.zeros(shape=((self.size,) + game_image_shape[1:]))

    def add(self, state, reward, action_idx, next_state, is_final_state):
        if self.states is None:
            self._initialize(state.shape)

        self.states[self._memory_idx] = state
        self.next_states[self._memory_idx] = next_state
        self.actions[self._memory_idx] = action_idx
        self.rewards[self._memory_idx] = reward
        self.is_next_final_state[self._memory_idx] = is_final_state

        self._memory_idx += 1

        if self._memory_idx == self.size:
            self._full = True
            if self._queue_behaviour:
                self._memory_idx = self._memory_idx % self.size

        if self._full and not self._queue_behaviour:
            self._memory_idx = random.randint(0, self.size)

    def train(self, model):
        has_uninitialized = any(filter(lambda x: x < 0, self.actions))

        num_choices = self._memory_idx if has_uninitialized else self.size
        if num_choices == 0:
            return

        sample_size = self.sample_size
        if num_choices < self.sample_size:
            sample_size = num_choices

        sample = random.sample(range(num_choices), sample_size)

        next_scores = np.max(
            model.predict(
                self.next_states[sample]
            ),
            axis=1
        )
        y = model.predict(self.states[sample])

        for yidx, val in enumerate(zip(sample, next_scores)):
            idx, next_score = val

            action = self.actions[idx]
            reward = self.rewards[idx]
            is_final_state = self.is_next_final_state[idx]

            old_score = y[yidx, action]
            if is_final_state:
                updated_score = reward
            else:
                new_score = reward + self.gamma * next_score - old_score
                updated_score = old_score + self.alpha * new_score
            y[yidx, action] = updated_score

            color = 'red'
            if updated_score > old_score:
                color = 'green'
            cprint("%s -> %s, reward: %s" % (old_score, updated_score, reward), color)

        model.fit(
            x=self.states[sample],
            y=y,
            epochs=self.model_train_epochs,
            batch_size=self.model_batch_size
        )


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114]).reshape((rgb.shape[:-1] + (1,)))


def learn_model(
        model,
        memory,
        iterations=10,
        random_factor_chance=0.1,
        random_factor_minimum=0.00001,
        random_factor_decay=0.001,
        initial_level=None,
        observe_steps=0,
        switch_level_on_pass=True,
        verbose=False,
        save_model_iteration=True):

    level = random.choice(LEVELS) if initial_level is None else LEVELS[initial_level]
    for _ in range(iterations):
        env = gym.make(level)
        env.reset()
        env.render()

        step = 0
        level_passed = True
        last_state = np.zeros(shape=(1, 224, 256, 1))
        scores = [0] * 64

        done = False
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
            observation = rgb2gray(observation)
            observation = observation.reshape(((1,) + observation.shape))
            observation = observation / 255.
            reward -= 0.4
            reward *= 5

            # Update network with reward
            scores = model.predict(observation)

            final_state = False
            if done:
                final_state = True
                reward = 50
                if info['life'] == 0:
                    level_passed = False
                    reward = -50
            reward /= 50

            if verbose:
                print("Reward:", reward)
                print("Random factor:", random_factor_chance)
                print("Best 4 Scores:", sorted(scores.flatten().tolist()[:4], reverse=True))

            memory.add(observation, reward, action_idx, last_state, final_state)
            last_state = observation

            if done or memory.is_full() or step >= observe_steps:
                memory.train(model)

            step += 1
        env.close()

        if level_passed and switch_level_on_pass:
            level = random.choice(LEVELS)

        if save_model_iteration:
            save_model(model)


if __name__ == '__main__':
    mario_model = build_model()
    # mario_model = load_model()

    replay_memory = ExperienceReplay(
        size=3000,
        alpha=0.1,
        gamma=0.9,
        train_epochs=1,
        sample_size=20,
        queue_behaviour=True
    )

    learn_model(mario_model,
                replay_memory,
                iterations=5000000,
                observe_steps=10,
                initial_level=27,
                verbose=True,
                random_factor_chance=1,
                random_factor_decay=0.0002,
                random_factor_minimum=0.05,
                save_model_iteration=True)
    save_model(mario_model)



