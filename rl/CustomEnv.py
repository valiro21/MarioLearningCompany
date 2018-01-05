import numpy as np
from skimage.transform import resize


class FrameBuffer(object):
    def __init__(self, frame_buffer_size=1, width=224, height=256):
        self.frames = np.zeros((1, frame_buffer_size, width, height))
        self.rewards = np.zeros(frame_buffer_size)
        self._frames_buffer_size = frame_buffer_size

    def add(self, frame, reward):
        self.frames = np.roll(self.frames, 1, axis=1)
        self.frames[0, self._frames_buffer_size - 1] = frame
        self.rewards = np.roll(self.rewards, 1)
        self.rewards[self._frames_buffer_size - 1] = reward

    def get_reward(self):
        return self.rewards[-1]


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).reshape(((1,) + rgb.shape[:-1]))


def normalize(array):
    a_min = np.min(array)
    a_max = np.max(array)

    if a_min == a_max:
        return np.zeros(array.shape)
    return (array - a_min) / (a_max - a_min)


def get_action(action):
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

    return mapping[action]


class CustomEnv(object):
    def __init__(self, env, frame_buffer_size=4, width=224, height=256):
        self.__class__ = type(env.__class__.__name__,
                              (self.__class__, env.__class__),
                              {})
        self.__dict__ = env.__dict__
        self._env = env
        self._frame_buffer = FrameBuffer(frame_buffer_size=frame_buffer_size, width=width, height=height)
        self._width = width
        self._height = height

    def convert_for_network(self, observation):
        # Convert to grayscale and normalize
        observation = resize(normalize(observation), (self._width, self._height))

        observation = rgb2gray(observation)
        return observation

    def reset(self):
        observation = self._env.reset()

        observation = self.convert_for_network(observation)

        return observation.reshape(((1,) + observation.shape))

    def step(self, action):
        action = get_action(action)[0]
        observation, reward, done, info = self._env.step(action)

        observation = self.convert_for_network(observation)

        self._frame_buffer.add(observation, reward)

        return self._frame_buffer.frames, self._frame_buffer.get_reward(), done, info
