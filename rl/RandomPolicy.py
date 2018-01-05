import random
import numpy as np


class RandomPolicy(object):
    def __init__(self, epsilon=0.1, epsilon_decay=0.0001,
                 epsilon_min=0.001, reset_on_level_change=False):
        self.initial_epsilon = epsilon
        self.current_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.reset_on_level_change = reset_on_level_change

    def game_changed(self):
        if self.reset_on_level_change:
            self.current_epsilon = self.initial_epsilon

    def get_action(self, scores):
        turn_epsilon = random.uniform(0., 1.)
        if turn_epsilon < self.current_epsilon:
            action = random.randint(0, 13)
        else:
            action = np.argmax(scores)

        self.current_epsilon = max(self.epsilon_min,
                                   self.current_epsilon * (1 - self.epsilon_decay))

        return action
