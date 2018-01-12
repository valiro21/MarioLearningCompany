import random
import numpy as np

from rl.Policy import Policy


class RandomPolicy(Policy):
    def __init__(self, epsilon=0.1, epsilon_decay=0.0001,
                 epsilon_min=0.001, reset_on_game_changed=True,
                 dropout=0.4, reset_on_game_load=False):
        super().__init__()

        self.initial_epsilon = epsilon
        self.current_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.reset_on_game_changed = reset_on_game_changed
        self.reset_on_game_load = reset_on_game_load
        self.dropout = dropout
        self._is_dropout = False

    def allows_async_training(self):
        return False

    def game_loaded(self):
        if self.reset_on_game_load:
            self.current_epsilon = self.initial_epsilon
        
        dropout_chance = random.uniform(0., 1.)
        self._is_dropout = False
        if dropout_chance < self.dropout:
            self._is_dropout = True

    def game_changed(self):
        if self.reset_on_game_changed:
            self.current_epsilon = self.initial_epsilon

    def get_action(self, scores):
        turn_epsilon = random.uniform(0., 1.)
        if not self._is_dropout and turn_epsilon < self.current_epsilon:
            action = random.randint(0, 13)
        else:
            action = np.argmax(scores)

        if not self._is_dropout:
            self.current_epsilon = max(self.epsilon_min,
                                       self.current_epsilon - self.epsilon_decay)
        print("Epsilon: %s, Dropout: %s" % (self.current_epsilon, self._is_dropout))
        return action
