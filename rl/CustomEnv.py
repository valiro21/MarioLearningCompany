import numpy as np
from skimage.transform import resize
from copy import copy


class FrameBuffer(object):
    def __init__(self, actions_history_size=4,
                 frame_history_size=2,
                 frame_width=224, frame_height=256,
                 history_width=84, history_height=84):
        self.frame = np.zeros((1, frame_width, frame_height))
        self.history = np.zeros((frame_history_size, history_width, history_height))
        self.rewards = np.zeros(frame_history_size + 1)
        self.controller_states = np.zeros((actions_history_size, 6))

    def reset(self):
        self.frame.fill(0)
        self.history.fill(0)
        self.rewards.fill(0)
        self.controller_states.fill(0)

    def add(self, frame, reward, controller_state):
        self.history = np.roll(self.history, 1, axis=0)
        self.history[-1] = resize(self.frame, output_shape=((1,) + self.history.shape[1:]))
        self.frame = frame
        self.rewards = np.roll(self.rewards, -1)
        self.rewards[-1] = reward
        self.controller_states = np.roll(self.controller_states, -1, axis=0)
        self.controller_states[-1] = controller_state

    def get_reward(self):
        return self.rewards[-1]

    def get_last_controller_states(self):
        return self.controller_states.flatten()

    def get_last_frame(self):
        return self.frame

    def get_previous_frames(self):
        return self.history


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
    def __init__(self, env,
                 frame_width=224, frame_height=256,
                 history_width=84, history_height=84,
                 frame_history_size=1, actions_history_size=4):
        self.__class__ = type(env.__class__.__name__,
                              (self.__class__, env.__class__),
                              {})
        self.__dict__ = env.__dict__
        self._env = env
        self._frame_buffer = FrameBuffer(
            frame_width=frame_width,
            frame_height=frame_height,
            history_width=history_width,
            history_height=history_height,
            frame_history_size=frame_history_size,
            actions_history_size=actions_history_size
        )
        self._width = frame_width
        self._height = frame_height
        self.last_info = None

    def convert_for_network(self, observation):
        observation = resize(observation, (self._width, self._height))
        observation = np.swapaxes(observation, 1, 2)
        observation= np.swapaxes(observation, 0, 1)
        observation /= 255.
        return observation

    def reset(self):
        observation = self._env.reset()
        observation = self.convert_for_network(observation)

        self._frame_buffer.reset()
        self._frame_buffer.add(observation, 0, [0, 0, 0, 0, 0, 0])

        return [self._frame_buffer.get_last_controller_states(),
                self._frame_buffer.get_previous_frames(),
                self._frame_buffer.get_last_frame()]

    def step(self, action):
        controller_state = get_action(action)[0]
        observation, reward, done, info = self._env.step(controller_state)

        self.last_info = copy(info)
        print(info)
        
        if info['life'] == 0:
            reward = -1

        terminate_iteration = False
#         if info['distance'] < 400 - info['time'] and info['time'] < 370:
#             terminate_iteration = True

#        if terminate_iteration:
#            done = True
#            info['distance'] = -1
#            reward = -1

        if reward > 0:
            reward = reward * info['time'] / 800.
        reward = np.clip(reward, -1, 1)

        observation = self.convert_for_network(observation)

        self._frame_buffer.add(observation, reward, controller_state)

        observation = [self._frame_buffer.get_last_controller_states(),
                       self._frame_buffer.get_previous_frames(),
                       self._frame_buffer.get_last_frame()]
        
        return observation, self._frame_buffer.get_reward(), done, info
