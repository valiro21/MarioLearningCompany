import os
from rl.policies import Policy
import matplotlib
import numpy as np
import skvideo.io

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_plot(x, y, file):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x, y)
    fig.savefig(os.path.join(file))
    plt.close(fig)


def play(agent, env, policy, record=True, working_dir="./train"):
    info_history, total_reward, _ = agent.train(
        env,
        policy,
    )

    if record:
        frames = np.array([info['frames'] for info in info_history])
        skvideo.io.vwrite(
            os.path.join(working_dir, 'play.mp4'),
            frames
        )


def train(agent, env, policy, memory, epochs=50, test_interval=None, working_dir="./train"):
    train_epochs = []
    test_epochs = []

    train_sum_rewards = []
    train_sum_loss = []
    train_avg_acc = []
    test_sum_rewards = []

    for epoch, epoch_data in enumerate(agent.train(env, policy,
                                                   memory=memory,
                                                   epochs=epochs)):
        info_history, total_reward, model_train_histories = epoch_data
        total_loss_sum = sum(
            map(
                lambda x: sum(x.history['loss']),
                model_train_histories
            )
        )
        avg_acc_sum = np.mean(
            list(map(
                lambda x: np.mean(x.history['acc']),
                model_train_histories
            ))
        )

        train_epochs.append(epoch)
        train_sum_rewards.append(total_reward)
        train_sum_loss.append(total_loss_sum)
        train_avg_acc.append(avg_acc_sum)

        if test_interval is not None and (epoch + 1) % test_interval == 0:
            info_history, total_reward, _ = agent.train(
                env,
                Policy(policy.action_mapper),
            )
            test_epochs.append(epoch)
            test_sum_rewards.append(total_reward)

        save_plot(train_epochs, train_sum_rewards, os.path.join(working_dir, "train_total_rewards.png"))
        save_plot(train_sum_loss, train_sum_rewards, os.path.join(working_dir, "train_total_loss.png"))
        save_plot(train_avg_acc, train_sum_rewards, os.path.join(working_dir, "train_avg_acc.png"))
        save_plot(test_sum_rewards, train_sum_rewards, os.path.join(working_dir, "test_total_rewards.png"))

        agent.save_model(
            os.path.join(working_dir, "model.json"),
            os.path.join(working_dir, "model.h5"),
        )

    play(agent, env, Policy(policy.action_mapper), record=True, working_dir=working_dir)
