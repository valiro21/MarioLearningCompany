import numpy as np

from rl import AsyncMethodExecutor


def _convert_data_for_model(data):
    return [np.array([row[idx] for row in data]) for idx in range(len(data[0]))]


def _log_move_details(policy, reward, scores, chosen_action):
    max_values_argmax = scores.argsort()[-4:][::-1]

    print("Reward:", reward)
    print("Chosen action", chosen_action)
    print("Best 4 values:")
    for idx in max_values_argmax:
        action = policy.action_mapper(idx)
        print("%s, %s -> %s" % (idx, action, scores[idx]))


class Agent(object):
    def __init__(self, model, gamma=0.9):
        self.model = model
        self.gamma = gamma

    def compute_qvalues(self, state):
        """
        Predict the qvalues for the given state.

        :param state: The state of the game.
        :return: The predicted qvalues.
        """
        return self.model.predict([np.expand_dims(item, 0) for item in state])[0]

    def compute_training_data(self, samples):
        """
        Computes the learned qvalues for each state of the given sample.

        :param samples: The samples for memory.
        :return: A tuple of game states and qvalues to feed the model.
        """
        states = _convert_data_for_model([row[0] for row in samples])
        next_states = _convert_data_for_model([row[3] for row in samples])

        next_scores = np.max(
            self.model.predict(
                next_states
            ),
            axis=1
        )
        qvalues = self.model.predict(states)

        for idx, val in enumerate(zip(samples, next_scores)):
            data_val, next_score = val
            state, reward, action, next_state, is_final_state = data_val

            if is_final_state:
                updated_score = reward
            else:
                new_score = reward + self.gamma * next_score
                updated_score = new_score

            qvalues[idx, action] = updated_score

        return states, qvalues

    def play(self, env, policy):
        is_final_state = False

        observation = env.reset()
        last_observation = []
        for item in observation:
            last_observation.append(np.zeros(shape=item.shape))

        policy.epoch_start()
        qvalues = self.compute_qvalues(observation)

        while not is_final_state:
            chosen_action = policy.get_action(env, qvalues)

            for dst, src in zip(last_observation, observation):
                np.copyto(dst, src)

            observation, reward, is_final_state, info = env.step(chosen_action)

            yield last_observation, reward, chosen_action, observation, is_final_state, info, qvalues

            qvalues = self.compute_qvalues(observation)
        env.close()

    def train_epoch(self, env, policy,
                    memory=None,
                    batch_size=None,
                    train_epoches=1,
                    verbose=1):
        debug_logger_thread = AsyncMethodExecutor()
        if verbose:
            debug_logger_thread.start()

        total_reward = 0
        model_train_histories = []
        info_history = []

        for time, step_data in enumerate(self.play(env, policy)):
            state, reward, chosen_action, next_state, is_final_state, info, qvalues = step_data

            total_reward += reward
            info_history.append(info)

            if memory is None:
                continue

            memory.add(
                state,
                reward,
                chosen_action,
                next_state,
                is_final_state,
                qvalues
            )

            if verbose:
                debug_logger_thread.run_on_thread(
                    _log_move_details,
                    policy,
                    reward,
                    qvalues,
                    chosen_action
                )

            if memory.is_ready():
                samples = memory.get_sample_data()

                x, y = self.compute_training_data(samples)

                train_history = self.model.fit(
                    x=x,
                    y=y,
                    epochs=train_epoches,
                    batch_size=batch_size,
                    verbose=verbose
                )

                model_train_histories.append(train_history)

        if verbose:
            debug_logger_thread.finalize()

        return info_history, total_reward, model_train_histories

    def train(self, env, policy, memory=None, epochs=50):
        for epoch in range(epochs):
            info_history, total_reward, model_train_histories = self.train_epoch(env, policy, memory)
            yield info_history, total_reward, model_train_histories

    def save_model(self, model, weights):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(weights)