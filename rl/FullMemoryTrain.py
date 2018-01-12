from rl.ExperienceReplay import ExperienceReplay

class FullMemoryTrain(ExperienceReplay):
    def __init__(self, max_size=100,
                 gamma=0.7, sample_size=5,
                 observe_steps=10,
                 train_epochs=1, batch_size=None,
                 queue_behaviour=True):
        super().__init__(max_size=max_size,
                         gamma=gamma, sample_size=None,
                         observe_steps=observe_steps,
                         train_epochs=train_epochs, batch_size=sample_size,
                         queue_behaviour=queue_behaviour)
        self._allow_training = False

    def allow_training(self):
        return self._allow_training

    def add(self, state, reward, scores, chosen_action, next_state, is_final_state):
        
        super().add(state, reward, scores, chosen_action, next_state, is_final_state)

        if is_final_state:
            self._allow_training = True

    def train(self, model, verbose=1):
        super().train(model, verbose=verbose)
        
        if self._allow_training:
            self._allow_training = False

