from rl.CustomEnv import get_action
import numpy as np

from rl.ExperienceReplay import ExperienceReplay


class FullMemory(ExperienceReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._allow_training = False

    def allow_training(self):
        return self._allow_training

    def add(self, state, reward, scores, chosen_action, next_state, is_final_state):
        super().add(state, reward, scores, chosen_action, next_state, is_final_state)

        if is_final_state:
            self._allow_training = True
        else:
            self._allow_training = False

    def train(self, model):
        self._allow_training = False

        next_states = self.next_states
        if not self.is_full():
            next_states = self.next_states[:self._memory_idx]

        states = self.states
        if not self.is_full():
            states = self.states[:self._memory_idx]

        next_scores = np.max(
            model.predict(
                next_states
            ),
            axis=1
        )
        y = model.predict(states)

        for idx, next_score in enumerate(next_scores):
            action = self.actions[idx]
            updated_score = self._compute_new_score(
                y[idx],
                action,
                self.rewards[idx],
                next_score,
                self.is_next_final_state[idx]
            )

            y[idx, action] = updated_score

        model.fit(
            x=states,
            y=y,
            epochs=self.model_train_epochs,
            batch_size=self.sample_size,
            verbose=1
        )
