import random
from collections import Counter
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


class ExperienceReplay(object):
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

    def train(self, model, experience_sample_size,
              train_epochs=5, train_batch_size=1,
              next_state_score=0, ):
        last_idx = self._memory_idx - 1 if self._memory_idx > 0 else self.max_memory_steps - 1

        y = model.predict(self.observations[[last_idx]])
        old_score = y[0, self.actions[last_idx]]
        updated_score = (1 - self.alpha) * old_score + self.alpha * (self.rewards[last_idx] + self.gamma * next_state_score)
        y[0, self.actions[last_idx]] = updated_score

        model.fit(
            x=self.observations[[last_idx]],
            y=y,
            epochs=train_epochs
        )

        has_uninitialized = any(filter(lambda x: x < 0, self.actions))

        num_choices = last_idx if has_uninitialized else self.max_memory_steps
        if num_choices == 0:
            return

        if num_choices < experience_sample_size:
            experience_sample_size = num_choices

        sample = random.sample(range(num_choices), experience_sample_size)

        samples_next_states = list(
            map(
                lambda x: (x + 1) % self.max_memory_steps,
                sample
            )
        )

        next_scores = np.max(
            model.predict(
                self.observations[samples_next_states]
            ),
            axis=1
        )
        y = model.predict(self.observations[sample])
        for yidx, val in enumerate(zip(sample, next_scores)):
            idx, next_score = val
            old_score = y[yidx, self.actions[idx]]
            updated_score = (1 - self.alpha) * old_score + self.alpha * (self.rewards[idx] + self.gamma * next_score)
            y[yidx, self.actions[idx]] = updated_score

        model.fit(
            x=self.observations[sample],
            y=y,
            epochs=train_epochs,
            batch_size=train_batch_size
        )


def learn_model(
        model,
        iterations=10,
        gamma=0.7,
        alpha=0.2,
        experience_sample_size=20,
        train_epochs=1,
        train_batch_size=1,
        random_factor_chance=0.1,
        random_factor_minimum=0.00001,
        random_factor_decay=0.001,
        max_memory_steps=50,
        observe_steps=0,
        switch_level_on_pass=True,
        verbose=False,
        save_model_iteration=True):

    level = random.choice(LEVELS)
    level_switched = True
    for _ in range(iterations):
        env = gym.make(level)
        env.reset()
        done = False

        env.render()
        scores = [0] * 64

        if level_switched:
            memory = ExperienceReplay(
                max_memory_steps,
                alpha=alpha,
                gamma=gamma
            )

        level_switched = False
        step = 0
        level_passed = True
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

            # Update network with reward
            scores = model.predict(observation)
            next_state_score = np.max(scores)

            if info['life'] == 0:
                level_passed = False
                reward = -1000
                next_state_score = 0

            if reward == 0:
                reward = -2
            reward /= 20

            if verbose:
                print("Reward:", reward)
                print("Done:", done)
                print("Info:", info)
                print("Random factor:", random_factor_chance)
                print("Scores:", scores)

            memory.add(observation, reward, action_idx)

            if done or step >= observe_steps:
                memory.train(model,
                             experience_sample_size=experience_sample_size,
                             train_epochs=train_epochs,
                             train_batch_size=train_batch_size,
                             next_state_score=next_state_score)

            step += 1
        env.close()

        if level_passed and switch_level_on_pass:
            level_switched = True
            level = random.choice(LEVELS)

        if save_model_iteration:
            save_model(model)


if __name__ == '__main__':
    mario_model = build_model()
    # mario_model = load_model()
    learn_model(mario_model,
                iterations=5000000,
                verbose=True,
                alpha=0.2,
                gamma=0.9,
                experience_sample_size=15,
                max_memory_steps=30,
                train_batch_size=50,
                train_epochs=1,
                random_factor_chance=0.1,
                random_factor_decay=0.0001,
                random_factor_minimum=0.0001,
                save_model_iteration=True)
    save_model(mario_model)



