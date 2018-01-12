import threading
import numpy as np


class Agent(object):
    def __init__(self, model):
        self.model = model

    def _compute_scores(self, observation):
        return self.model.predict(observation)[0]

    def train(self, env, memory, policy, **kwargs):
        if policy.allows_async_training():
            return self._train_async(env, memory, policy, **kwargs)
        else:
            return self._train_sync(env, memory, policy, **kwargs)

    def _train_sync(self, env, memory, policy):
        done = False
        info = {}
        observation = env.reset()
        last_observation = []
        for item in observation:
            last_observation.append(np.zeros(shape=item.shape))
        scores = self.model.predict(observation)

        env.render()
        policy.game_loaded()
        while not done:
            action = policy.get_action(scores)

            for dst, src in zip(last_observation, observation):
                np.copyto(dst, src)
            observation, reward, done, info = env.step(action)

            scores = self._compute_scores(observation)

            memory.add(
                last_observation,
                reward,
                scores,
                action,
                observation,
                done
            )

            if memory.allow_training():
                memory.train(self.model)
        env.close()

        return info

    def _train_async(self, env, memory, policy):
        done = False
        info = {}
        observation = env.reset()
        last_observation = []
        for item in observation:
            last_observation.append(np.zeros(shape=item.shape))
        scores = self.model.predict(observation)

        env.render()
        policy.game_loaded()

        def _train_memory():
            while True:
                memory.train(self.model)

        train_started = False

        while not done:
            action = policy.get_action(scores)

            for dst, src in zip(last_observation, observation):
                np.copyto(dst, src)
            observation, reward, done, info = env.step(action)

            scores = self._compute_scores(observation)

            memory.add(
                last_observation,
                reward,
                scores,
                action,
                observation,
                done
            )

            if not train_started:
                threading.Thread(target=_train_memory).start()
                train_started = True
        env.close()

        return info
