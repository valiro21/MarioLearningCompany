import random
import gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario.wrappers import SetPlayingMode

from consts import LEVELS

from models.neural_network import build_model, save_model, load_model
from rl.Agent import Agent
from rl.CustomEnv import CustomEnv
from rl.AsyncMethodExecutor import AsyncMethodExecutor
from rl.AgentConvolutionDebug import AgentConvolutionDebug
from rl.HumanPlayerPolicy import HumanPlayerPolicy
from rl.RandomPolicy import RandomPolicy
from rl.ExperienceReplay import ExperienceReplay
from rl.MemoryLogger import MemoryLogger
from rl.FullMemoryTrain import FullMemoryTrain
from rl.LevelRandomPolicy import LevelRandomPolicy


def train(agent, memory, policy, iterations=50,
          frame_history_size=2, actions_history_size=4,
          initial_level=None, change_level=True,
          save_model_iteration_interval=1):
    level = random.choice(LEVELS) if initial_level is None else initial_level
    policy.game_changed()
    for iteration in range(iterations):
        env = gym.make(level)

        env = CustomEnv(env,
                        frame_width=224,
                        frame_height=224,
                        history_width=64,
                        history_height=64,
                        frame_history_size=frame_history_size,
                        actions_history_size=actions_history_size)
        last_info = agent.train(env, memory, policy)

        if change_level and last_info['life'] > 0 and last_info['distance'] != -1:
            level = random.choice(LEVELS)
            policy.game_changed()

        if save_model_iteration_interval is not None:
            if iteration % save_model_iteration_interval == 0:
                save_model(agent.model)

    save_model(agent.model)


if __name__ == '__main__':
    seed = 123123218
    random.seed(seed)
    actions_history_size = 4
    frame_history_size = 2
    learning_rate = 0.0001
    mario_model = build_model(actions_history_size=actions_history_size,
                              frame_history_size=frame_history_size,
                              learning_rate=learning_rate)
    # mario_model = load_model(learning_rate=learning_rate)

    replay_memory = ExperienceReplay(
        max_size=1200,
        gamma=0.9,
        train_epochs=1,
        train_interval=280,
        sample_size=None,
        batch_size=200,
        queue_behaviour=True
    )
    
    # replay_memory = FullMemoryTrain(
    #     max_size=1000,
    #     gamma=0.9,
    #     train_epochs=1,
    #     sample_size=50,
    #     queue_behaviour=True
    # )

    debug_logger_thread = AsyncMethodExecutor()
    debug_logger_thread.start()

    agent = Agent(mario_model)

    # policy = HumanPlayerPolicy()
    policy = RandomPolicy(epsilon=0.75, epsilon_decay=0.00055, epsilon_min=0.15, dropout=0.1)
    # policy = LevelRandomPolicy(epsilon=1., epsilon_min=0.1)
    # agent = AgentConvolutionDebug(agent, debug_logger_thread, layers=[2, 3], show_network_input=False)

    train(agent,
          MemoryLogger(replay_memory, debug_logger_thread, log_training=50, log_action=1),
          policy,
          actions_history_size=actions_history_size,
          frame_history_size=frame_history_size,
          initial_level=LEVELS[1],
          save_model_iteration_interval=2,
          iterations=2000)

    save_model(mario_model)
