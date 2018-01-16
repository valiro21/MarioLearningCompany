import random
import gym
import ppaquette_gym_super_mario

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
import matplotlib.pyplot as plt


def train(agent, memory, policy, iterations=50,
          frame_history_size=2, actions_history_size=4,
          initial_level=None, change_level=True,
          save_model_iteration_interval=1,
          total_reward_figure_file='rewards.png'):
    level = random.choice(LEVELS) if initial_level is None else initial_level
    policy.game_changed()
    total_rewards = []
    for iteration in range(iterations):
        env = gym.make(level)

        env = CustomEnv(env,
                        frame_width=128,
                        frame_height=128,
                        history_width=32,
                        history_height=32,
                        frame_history_size=frame_history_size,
                        actions_history_size=actions_history_size)
        last_info, total_reward = agent.train(env, memory, policy)
        print("Last info:", last_info)
        print("Total reward:", total_reward)

        total_rewards.append(total_reward)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(range(1, iteration+2), total_rewards)
        fig.savefig(total_reward_figure_file)
        plt.close(fig)

        if change_level and last_info['life'] > 0 and last_info['distance'] != -1:
            level = random.choice(LEVELS)
            policy.game_changed()

        if save_model_iteration_interval is not None:
            if iteration % save_model_iteration_interval == 0:
                save_model(agent.model)

    save_model(agent.model)


def play(agent, policy, level=None):
    level = random.choice(LEVELS) if level is None else level
    env = gym.make(level)

    env = CustomEnv(env,
                    frame_width=128,
                    frame_height=128,
                    history_width=32,
                    history_height=32,
                    frame_history_size=frame_history_size,
                    actions_history_size=actions_history_size)
    last_info, total_reward = agent.play(env, policy)
    print("Last info:", last_info)
    print("Total reward:", total_reward)


if __name__ == '__main__':
    seed = 123123223
    random.seed(seed)
    actions_history_size = 6
    frame_history_size = 4
    learning_rate = 0.0004
    mario_model = build_model(actions_history_size=actions_history_size,
                              frame_history_size=frame_history_size,
                              learning_rate=learning_rate)
    # mario_model = load_model(learning_rate=learning_rate)
    
    debug_logger_thread = AsyncMethodExecutor()
    debug_logger_thread.start()

    agent = Agent(mario_model)
    # play(agent, RandomPolicy(epsilon=0., epsilon_min=0., dropout=1.), level=LEVELS[1])
    # exit()

    replay_memory = ExperienceReplay(
        max_size=100000,
        gamma=0.9,
        sample_size=32,
        database_file='cache/memory.db',
        should_pop_oldest=True,
        reuse_db=True
    )
    
    # replay_memory = FullMemoryTrain(
    #     max_size=1000,
    #     gamma=0.9,
    #     train_epochs=1,
    #     sample_size=50,
    #     queue_behaviour=True
    # )


    # policy = HumanPlayerPolicy()
    policy = RandomPolicy(epsilon=1., epsilon_decay=0.00001, epsilon_min=0.05, dropout=0.01)
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
