import pygame

from rl.Policy import Policy


class HumanPlayerPolicy(Policy):
    def __init__(self):
        super().__init__()

        pygame.init()
        w = 640
        h = 480
        size = (w, h)
        screen = pygame.display.set_mode(size)

    def game_loaded(self):
        pass

    def game_changed(self):
        pass

    def allows_async_training(self):
        return True

    def get_action(self, scores):
        mapping = {
            0: ([0, 0, 0, 0, 0, 0], "NOOP"),
            1: ([1, 0, 0, 0, 0, 0], "Up"),
            2: ([0, 0, 1, 0, 0, 0], "Down"),
            3: ([0, 1, 0, 0, 0, 0], "Left"),
            4: ([0, 1, 0, 0, 1, 0], "Left + A"),
            5: ([0, 1, 0, 0, 0, 1], "Left + B"),
            6: ([0, 1, 0, 0, 1, 1], "Left + A + B"),
            7: ([0, 0, 0, 1, 0, 0], "Right"),
            8: ([0, 0, 0, 1, 1, 0], "Right + A"),
            9: ([0, 0, 0, 1, 0, 1], "Right + B"),
            10: ([0, 0, 0, 1, 1, 1], "Right + A + B"),
            11: ([0, 0, 0, 0, 1, 0], "A"),
            12: ([0, 0, 0, 0, 0, 1], "B"),
            13: ([0, 0, 0, 0, 1, 1], "A + B")
        }

        key_indexes = {
            pygame.K_UP: 4, # JUMP
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 1,
            pygame.K_RIGHT: 3,
            pygame.K_d: 0,
            pygame.K_f: 5,
        }

        events = pygame.event.get() # no idea why this line makes it work
        pressed_keys = pygame.key.get_pressed()

        action_arr = [0 for _ in range(6)]

        for k in key_indexes:
            if pressed_keys[k]:
                action_arr[key_indexes[k]] = 1

        action = 0
        for key, value in mapping.items():
            if value[0] == action_arr:
                action = key

        # action might be 0 in the end

        return action
