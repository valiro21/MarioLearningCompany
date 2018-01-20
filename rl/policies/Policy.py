from abc import abstractmethod
import numpy as np


class Policy(object):
    def __init__(self, action_mapper):
        assert action_mapper is not None, "The model to env action mapper must be present."
        self.action_mapper = action_mapper
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

    def get_action(self, env, qvalues):
        return self.action_mapper(np.argmax(qvalues))
