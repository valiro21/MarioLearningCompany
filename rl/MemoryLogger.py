from termcolor import cprint

from rl.CustomEnv import get_action
from rl.AsyncMethodExecutor import AsyncMethodExecutor


def _log_train_details(old_score, action, reward, next_score, updated_score):
    color = 'red'
    if float(updated_score) > float(old_score):
        color = 'green'

    action_name = get_action(action)[1]
    cprint("%s: %s -> %s, next_score: %s, reward: %s, diff: %s" % (action_name, float(old_score), float(updated_score), float(next_score), reward, abs(updated_score - old_score)), color)


class MemoryLogger(object):
    def __init__(self, memory, debug_logger_thread, log_action=1, log_training=1, log_action_statistics=False):
        self.__class__ = type(memory.__class__.__name__,
                              (self.__class__, memory.__class__),
                              {})
        self.__dict__ = memory.__dict__
        self._memory = memory
        self._debug_logger_thread = debug_logger_thread
        self._log_training = log_training
        self._log_action = log_action
        self._log_action_statistics = log_action_statistics
        self._add_time = 0

    def _log_move_details(self, state, reward, scores, chosen_action, next_state, is_final_state):
        print("Reward:", reward)
        action_name = get_action(chosen_action)[1]
        print("Memory used: %s / %s" % (self._memory.size() , self._memory.max_size))
        print("Chosen action", action_name)
        max_values_argmax = scores.argsort()[-4:][::-1]
        print("Best 4 values:")
        for idx in max_values_argmax:
            print("%s, %s -> %s" % (idx, get_action(idx)[1], scores[idx]))
        
        if self._log_action_statistics:
            print("Memory stats:")
            stats_map = self._memory.actions_stats
            stats = sorted(list(map(lambda x: (stats_map[x], x), stats_map)), reverse=True)
            for num_entries, action in stats:
                action_name = get_action(action)[1]
                print("%s: %s" % (action_name, num_entries))

    def add(self, state, reward, chosen_action, next_state, is_final_state, scores):
        self._memory.add(state, reward, chosen_action, next_state, is_final_state, scores)

        if self._log_action is not None and self._add_time % self._log_action == 0:
            self._debug_logger_thread.run_on_thread(
                self._log_move_details,
                state,
                reward,
                scores,
                chosen_action,
                next_state,
                is_final_state
            )
        self._add_time += 1

    def _compute_new_score(self, time, scores, action, reward, next_score, is_final_state):
        old_score = scores[action]
        updated_score = self._memory._compute_new_score(time, scores, action, reward, next_score, is_final_state)

        if self._log_training is not None and time % self._log_training == 0:
            self._debug_logger_thread.run_on_thread(
                _log_train_details,
                old_score,
                action,
                reward,
                next_score,
                updated_score
            )

        return updated_score
