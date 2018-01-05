import random
import numpy as np


class Agent(object):
    def __init__(self, model):
        self.model = model

    def _compute_scores(self, observation):
        return self.model.predict(observation)[0]

    def train(self, env, memory, policy, observe_steps=10):
        done = False
        info = {}
        observation = env.reset()
        last_observation = np.zeros(shape=observation.shape)
        scores = self.model.predict(observation)

        env.render()
        policy.game_changed()
        while not done:
            action = policy.get_action(scores)

            np.copyto(last_observation, observation)
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

            if memory.is_full() or memory.size() >= observe_steps:
                memory.train(self.model)
        env.close()

        return info
