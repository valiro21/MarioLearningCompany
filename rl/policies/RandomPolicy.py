import random
import numpy as np

from rl.policies import Policy


class RandomPolicy(Policy):
    def __init__(self,
                 action_mapper,
                 epsilon=0.1,
                 epsilon_decay_step=0.0001,
                 epsilon_decay_epoch=0.,
                 epsilon_min=0.001,
                 reset_on_game_changed=True,
                 dropout=0.4,
                 reset_on_epoch_start=False):
        super().__init__(action_mapper)

        self.current_epsilon = epsilon

        self._initial_epsilon = epsilon
        self._epsilon_decay_step = epsilon_decay_step
        self._epsilon_decay_epoch = epsilon_decay_epoch
        self._epsilon_min = epsilon_min

        self._reset_on_game_changed = reset_on_game_changed
        self._reset_on_epoch_start = reset_on_epoch_start
        if reset_on_epoch_start:
            assert epsilon_decay_epoch == 0, "Decay per level should be 0 if epsilon is reset each epoch."

        self._dropout = dropout
        self._is_dropout = False

    def allows_async_training(self):
        return False

    def epoch_start(self):
        dropout_chance = random.uniform(0., 1.)
        self._is_dropout = False
        if dropout_chance < self._dropout:
            self._is_dropout = True
        elif self._reset_on_epoch_start:
            self.current_epsilon = self._initial_epsilon
        else:
            self.current_epsilon = max(
                self._epsilon_min,
                self.current_epsilon - self._epsilon_decay_epoch
            )

    def game_changed(self):
        if self._reset_on_game_changed:
            self.current_epsilon = self._initial_epsilon

    def get_action(self, env, qvalues):
        turn_epsilon = random.uniform(0., 1.)
        if not self._is_dropout and turn_epsilon < self.current_epsilon:
            action = env.action_space.sample()
        else:
            action = super().get_action(env, qvalues)

        if not self._is_dropout:
            self.current_epsilon = max(self._epsilon_min,
                                       self.current_epsilon - self._epsilon_decay_step)
        return action
