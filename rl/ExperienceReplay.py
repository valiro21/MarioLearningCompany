import numpy as np
import random
from termcolor import cprint

from rl.CustomEnv import get_action


class ExperienceReplay(object):
    def __init__(self, max_size=100,
                 gamma=0.7, sample_size=5,
                 train_epochs=1, batch_size=None,
                 queue_behaviour=True):
        self.max_size = max_size
        self._size = 0
        self._memory_idx = 0
        self.time = 0
        self.gamma = gamma
        self.observations = None
        self._queue_behaviour = queue_behaviour
        self._full = False
        self.sample_size = sample_size
        self.model_train_epochs = train_epochs
        if batch_size is None:
            batch_size = sample_size
        self.model_batch_size = batch_size

        self.states = None
        self.next_states = None
        self.actions = [-1] * max_size
        self.rewards = [0] * max_size
        self.is_next_final_state = [False] * max_size

    def size(self):
        return self._size

    def is_full(self):
        return self._full

    def _initialize(self, idx, game_image_shape):
        self.states[idx] = np.zeros(shape=((self.max_size,) + game_image_shape[1:]))
        self.next_states[idx] = np.zeros(shape=((self.max_size,) + game_image_shape[1:]))

    def add(self, state, reward, scores, chosen_action, next_state, is_final_state):
        if self.states is None:
            self.states = [None] * len(state)
            self.next_states = [None] * len(state)
            for idx, item in enumerate(state):
                self._initialize(idx, item.shape)

        for idx, item in enumerate(zip(state, next_state)):
            self.states[idx][self._memory_idx] = item[0]
            self.next_states[idx][self._memory_idx] = item[1]
        self.actions[self._memory_idx] = chosen_action
        self.rewards[self._memory_idx] = reward
        self.is_next_final_state[self._memory_idx] = is_final_state

        self._memory_idx += 1

        if not self._full:
            self._size += 1

        if self._memory_idx == self.max_size:
            self._full = True
            if self._queue_behaviour:
                self._memory_idx = self._memory_idx % self.max_size

        if self._full and not self._queue_behaviour:
            self._memory_idx = random.randint(0, self.max_size - 1)

    def _compute_new_score(self, scores, action, reward, next_score, is_final_state):
        if is_final_state:
            updated_score = reward
        else:
            new_score = reward + self.gamma * next_score
            updated_score = new_score
        return updated_score

    def train(self, model):
        has_uninitialized = any(filter(lambda x: x < 0, self.actions))

        num_choices = self._memory_idx if has_uninitialized else self.max_size
        if num_choices == 0:
            return

        sample_size = self.sample_size
        if num_choices < self.sample_size:
            sample_size = num_choices

        sample = random.sample(range(num_choices), sample_size)

        next_states_info = []
        for item in self.states:
            next_states_info.append(item[sample])
        next_scores = np.max(
            model.predict(
                next_states_info
            ),
            axis=1
        )

        states_info = []
        for item in self.states:
            states_info.append(item[sample])
        y = model.predict(states_info)

        for yidx, val in enumerate(zip(sample, next_scores)):
            idx, next_score = val

            action = self.actions[idx]
            updated_score = self._compute_new_score(
                y[yidx],
                action,
                self.rewards[idx],
                next_score,
                self.is_next_final_state[idx]
            )

            y[yidx, action] = updated_score

        model.fit(
            x=states_info,
            y=y,
            epochs=self.model_train_epochs,
            batch_size=self.model_batch_size,
            verbose=0
        )

        self.time += 1
