import random
import numpy as np

from rl.Policy import Policy


class LevelRandomPolicy(Policy):
    def __init__(self, epsilon=0.9,
                 decay_interval=2, decay_val=0.5,
                 epsilon_min=0.1,
                 reset_on_game_changed=True):
        super().__init__()

        self.initial_epsilon = epsilon
        self.current_epsilon = epsilon
        self.decay_interval = decay_interval
        self.decay_val = decay_val
        self.epsilon_min = epsilon_min
        self.reset_on_game_changed = reset_on_game_changed
        self._epoch_since_last_decay = 0
        
    def allows_async_training(self):
        return False

    def game_loaded(self):
        self._epoch_since_last_decay += 1
        if self._epoch_since_last_decay % self.decay_interval == 0:
            self._epoch_since_last_decay = 0
            self.current_epsilon *= self.decay_val
            self.current_epsilon = max(self.current_epsilon, self.epsilon_min)

    def game_changed(self):
        if self.reset_on_game_changed:
            self.current_epsilon = self.initial_epsilon
            self._epoch_since_last_decay = 0

    def get_action(self, scores):
        turn_epsilon = random.uniform(0., 1.)
        if turn_epsilon < self.current_epsilon:
            action = random.randint(0, 13)
        else:
            action = np.argmax(scores)

        print("Epsilon:", self.current_epsilon)
        return action
