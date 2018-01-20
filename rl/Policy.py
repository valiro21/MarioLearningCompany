from abc import abstractmethod
import numpy as np


class Policy(object):
    def __init__(self):
        pass

    @abstractmethod
    def epoch_start(self):
        pass

    @abstractmethod
    def allows_async_training(self):
        pass

    @abstractmethod
    def game_changed(self):
        pass

    def get_action(self, scores):
        return np.argmax(scores)
