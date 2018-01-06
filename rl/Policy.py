import pygame
from abc import ABC, abstractmethod


class Policy(object):
    def __init__(self):
        pass

    @abstractmethod
    def game_loaded(self):
        pass

    @abstractmethod
    def allows_async_training(self):
        pass

    @abstractmethod
    def game_changed(self):
        pass

    @abstractmethod
    def get_action(self, scores):
        pass
