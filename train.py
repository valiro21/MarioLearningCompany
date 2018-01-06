import random
import gym
import ppaquette_gym_super_mario

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
          initial_level=None, change_level=True,
          save_on_iteration=True):
    level = random.choice(LEVELS) if initial_level is None else initial_level
    policy.game_changed()
    for _ in range(iterations):
        env = CustomEnv(gym.make(level),
                        frame_buffer_size=1,
                        width=84,
                        height=84,
                        action_history_size=1)
        last_info = agent.train(env, memory, policy)

        if change_level and last_info['life'] > 0:
            level = random.choice(LEVELS)
            policy.game_changed()

        if save_on_iteration:
            save_model(agent.model)


if __name__ == '__main__':
    action_history_size = 1
    mario_model = build_model(history_size=action_history_size)
    # mario_model = load_model()

    replay_memory = ExperienceReplay(
        max_size=10000,
        gamma=0.9,
        train_epochs=1,
        sample_size=50,
        queue_behaviour=True
    )

    debug_logger_thread = DebugLoggerThread()
    debug_logger_thread.start()

    agent = Agent(mario_model)

    policy = RandomPolicy(epsilon=1., epsilon_decay=0.00001, epsilon_min=0.01)

    # agent = AgentConvolutionDebug(agent, debug_logger_thread, layers=[0])

    train(agent,
          MemoryLogger(replay_memory, debug_logger_thread, log_training=False),
          policy,
          initial_level=LEVELS[17],
          iterations=2000)

    save_model(mario_model)
