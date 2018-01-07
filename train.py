import random
import gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario.wrappers import SetPlayingMode

from consts import LEVELS

from models.neural_network import build_model, save_model, load_model
from rl.Agent import Agent
from rl.CustomEnv import CustomEnv
from rl.DebugLoggerThread import DebugLoggerThread
from rl.AgentConvolutionDebug import AgentConvolutionDebug
from rl.RandomPolicy import RandomPolicy
from rl.ExperienceReplay import ExperienceReplay
from rl.MemoryLogger import MemoryLogger


def train(agent, memory, policy, iterations=50,
          history_size=1,
          initial_level=None, change_level=True,
          save_on_iteration=True):
    level = random.choice(LEVELS) if initial_level is None else initial_level
    policy.game_changed()
    for _ in range(iterations):
        env = gym.make(level)

        env = CustomEnv(env,
                        frame_width=64,
                        frame_height=64,
                        history_width=32,
                        history_height=32,
                        history_size=history_size)
        last_info = agent.train(env, memory, policy)

        if change_level and last_info['life'] > 0:
            level = random.choice(LEVELS)
            policy.game_changed()

        if save_on_iteration:
            save_model(agent.model)


if __name__ == '__main__':
    seed = 12312413232
    random.seed(seed)
    history_size = 6
    learning_rate = 0.00001
    mario_model = build_model(history_size=history_size, learning_rate=learning_rate)
    # mario_model = load_model(learning_rate=learning_rate)

    replay_memory = ExperienceReplay(
        max_size=10000,
        gamma=0.75,
        train_epochs=1,
        sample_size=50,
        queue_behaviour=True
    )

    debug_logger_thread = DebugLoggerThread()
    debug_logger_thread.start()

    agent = Agent(mario_model)

    policy = RandomPolicy(epsilon=1., epsilon_decay=0.00001, epsilon_min=0.1)

    # agent = AgentConvolutionDebug(agent, debug_logger_thread, layers=[0])

    train(agent,
          MemoryLogger(replay_memory, debug_logger_thread, log_training=False),
          policy,
          history_size=history_size,
          initial_level=LEVELS[16],
          iterations=2000)

    save_model(mario_model)
