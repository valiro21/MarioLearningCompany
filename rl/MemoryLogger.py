from termcolor import cprint

from rl.CustomEnv import get_action
from rl.DebugLoggerThread import DebugLoggerThread


def _log_move_details(state, reward, scores, chosen_action, next_state, is_final_state):
    print("Reward:", reward)
    action_name = get_action(chosen_action)[1]
    print("Chosen action", action_name)
    max_values_argmax = scores.argsort()[-4:][::-1]
    print("Best 4 values:")
    for idx in max_values_argmax:
        print("%s, %s -> %s" % (idx, get_action(idx)[1], scores[idx]))


def _log_train_details(scores, action, reward, updated_score):
    color = 'red'
    if updated_score > scores[action]:
        color = 'green'

    action_name = get_action(action)[1]
    cprint("%s: %s -> %s, reward: %s" % (action_name, scores[action], updated_score, reward), color)


class MemoryLogger(object):
    def __init__(self, memory, debug_logger_thread):
        self.__class__ = type(memory.__class__.__name__,
                              (self.__class__, memory.__class__),
                              {})
        self.__dict__ = memory.__dict__
        self._memory = memory
        self._debug_logger_thread = debug_logger_thread

    def add(self, state, reward, scores, chosen_action, next_state, is_final_state):
        self._memory.add(state, reward, scores, chosen_action, next_state, is_final_state)

        self._debug_logger_thread.run_on_thread(
            _log_move_details,
            state,
            reward,
            scores,
            chosen_action,
            next_state,
            is_final_state
        )

    def _compute_new_score(self, scores, action, reward, next_score, is_final_state):
        updated_score = self._memory._compute_new_score(scores, action, reward, next_score, is_final_state)

        self._debug_logger_thread.run_on_thread(
            _log_train_details,
            scores,
            action,
            reward,
            updated_score
        )

        return updated_score