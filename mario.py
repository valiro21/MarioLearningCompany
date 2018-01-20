import os
import random

import gym
import ppaquette_gym_super_mario

from consts import LEVELS
from rl.policies import RandomPolicy
from rl import ExperienceReplay, Agent

from CustomEnv import CustomEnv
from CustomEnv import action_mapper
from rl.train import train

from models.neural_network import build_model




def create_working_dir(*args, base_dir="./training_data"):
    current_dir = base_dir
    if not os.path.exists(current_dir):
        os.mkdir(current_dir)

    for training_parameter in args:
        current_dir = os.path.join(current_dir, str(training_parameter))
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
    return current_dir


def train_mario_model(epochs=500,
                      gamma=0.9,
                      learning_rate=0.004,
                      memory_size=100000,
                      sample_size=32,
                      level=LEVELS[0],
                      actions_history_size=16,
                      frame_history_size=4):
    working_dir = create_working_dir(
        epochs,
        gamma,
        learning_rate,
        memory_size,
        sample_size,
        actions_history_size,
        frame_history_size
    )

    env = CustomEnv(
        gym.make(level),
        frame_width=84,
        frame_height=84,
        history_width=32,
        history_height=32,
        actions_history_size=actions_history_size,
        frame_history_size=frame_history_size
    )

    model = build_model(
        actions_history_size=actions_history_size,
        frame_history_size=frame_history_size,
        learning_rate=learning_rate
    )

    agent = Agent(model, gamma=gamma)
    memory = ExperienceReplay(
        max_size=memory_size,
        sample_size=sample_size,
        database_file='memory.db',
        should_pop_oldest=True,
        reuse_db=False,
        verbose=True
    )
    policy = RandomPolicy(
        action_mapper,
        epsilon=1.,
        epsilon_decay_step=0.00001,
        epsilon_min=0.05,
        dropout=0.01
    )

    train(
        agent,
        env,
        policy,
        memory,
        epochs=epochs,
        test_interval=True,
        working_dir=working_dir
    )


if __name__ == '__main__':
    seed = 123123223
    random.seed(seed)

    train_mario_model(
        epochs=500,
        gamma=0.9,
        learning_rate=0.0004,
        memory_size=135000,
        sample_size=32,
        actions_history_size=16,
        frame_history_size=2
    )
