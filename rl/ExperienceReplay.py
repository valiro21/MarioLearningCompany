import numpy as np
import random
from termcolor import cprint

from rl.CustomEnv import get_action


class ExperienceReplay(object):
    def __init__(self, max_size=100, alpha=0.2,
                 alpha_decay_function=None,
                 gamma=0.7, sample_size=5,
                 train_epochs=1, batch_size=None,
                 queue_behaviour=True):
        self.max_size = max_size
        self._size = 0
        self._memory_idx = 0
        self.time = 0
        self.alpha = alpha
        self._alpha_decay_function = alpha_decay_function
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

    def _initialize(self, game_image_shape):
        self.states = np.zeros(shape=((self.max_size,) + game_image_shape[1:]))
        self.next_states = np.zeros(shape=((self.max_size,) + game_image_shape[1:]))

    def add(self, state, reward, scores, next_state, is_final_state):
        if self.states is None:
            self._initialize(state.shape)

        self.states[self._memory_idx] = state
        self.next_states[self._memory_idx] = next_state
        self.actions[self._memory_idx] = np.argmax(scores)
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
        old_score = scores[action]
        if is_final_state:
            updated_score = reward
        else:
            new_score = np.clip(reward + self.gamma * next_score - old_score, -1., 1.)
            updated_score = old_score + self.alpha * new_score
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

        next_scores = np.max(
            model.predict(
                self.next_states[sample]
            ),
            axis=1
        )
        y = model.predict(self.states[sample])

        self.alpha = self._alpha_decay_function(self.alpha, self.time)
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
            x=self.states[sample],
            y=y,
            epochs=self.model_train_epochs,
            batch_size=self.model_batch_size,
            verbose=0
        )

        self.time += 1
