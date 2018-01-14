import random

import numpy as np
import sqlite3
import pickle
from contextlib import closing
from blist import sortedlist


class ExperienceReplay(object):
    def __init__(
        self,
        max_size=100,
        gamma=0.7,
        sample_size=32,
        should_pop_oldest=True,
        database_file='memory.db',
        table_name='memory',
        reuse_db=True,
    ):
        self.max_size = max_size
        self.gamma = gamma
        self.sample_size = sample_size
        self.time = 0

        self._should_pop_oldest = should_pop_oldest
        self._size = 0
        self._table_name = table_name
        self._database_connection = sqlite3.connect(database_file, isolation_level=None)
        self._ids = sortedlist()
        self._ids_idx = []
        self._init_db(reuse_db)

    def _init_db(self, reuse_db):
        with closing(self._database_connection.cursor()) as cursor:
            if not reuse_db:
                cursor.execute('DROP TABLE IF EXISTS %s' % self._table_name)

            cursor.execute(
                'CREATE TABLE IF NOT EXISTS %s ('
                'id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'state blob,'
                'reward blob,'
                'chosen_action blob,'
                'next_state blob,'
                'is_final_state blob)'
                % self._table_name
            )

            data = cursor.execute('SELECT id FROM %s' % self._table_name).fetchall()
            ids = [row[0] for row in data]
            self._ids = sortedlist(ids)
            self._ids_idx = list(range(len(self._ids)))

    def allow_training(self):
        return len(self._ids) >= self.sample_size

    def size(self):
        return len(self._ids)

    def is_full(self):
        return len(self._ids) == self.max_size

    @staticmethod
    def _to_sqlite_blob(obj):
        return sqlite3.Binary(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))

    @staticmethod
    def _to_python_obj(blob):
        return pickle.loads(blob)

    @staticmethod
    def _convert_data_for_model(data):
        return [np.array([row[idx] for row in data]) for idx in range(len(data[0]))]

    def add(self, state, reward, action, next_state, is_final_state, scores):
        with closing(self._database_connection.cursor()) as cursor:
            if self.is_full():
                id_to_delete = self._ids[0]
                if not self._should_pop_oldest:
                    id_to_delete = self._ids[random.randint(0, len(self._ids) - 1)]
                cursor.execute('DELETE FROM %s WHERE id = ?;' % self._table_name, (id_to_delete, ))
                self._ids.remove(id_to_delete)
                self._ids_idx.pop()
            else:
                self._size += 1
            pickled_data = tuple(map(ExperienceReplay._to_sqlite_blob, (state, reward, action, next_state, is_final_state)))

            cursor.execute(
                'INSERT INTO %s' 
                '(state, reward, chosen_action, next_state, is_final_state)'
                'VALUES (?, ?, ?, ?, ?)'
                % self._table_name,
                pickled_data
            )

            self._ids.add(cursor.lastrowid)
            self._ids_idx.append(len(self._ids_idx))

    def _compute_new_score(self, time, scores, chosen_action, reward, next_score, is_final_state):
        if is_final_state:
            updated_score = reward
        else:
            new_score = reward + self.gamma * next_score
            updated_score = new_score
        return updated_score

    def get_train_data(self, model):
        self.time += 1

        sample_idx = random.sample(self._ids_idx, self.sample_size)
        ids = [self._ids[idx] for idx in sample_idx]
        id_list = ",".join(map(str, ids))

        with closing(self._database_connection.cursor()) as cursor:
            sqlite_data = cursor.execute(
                'SELECT * FROM %s WHERE id IN (%s)' % (self._table_name, id_list)
            ).fetchall()

            if not self.allow_training():
                return

            samples = []
            for row in sqlite_data:
                memory_row = list(map(
                    ExperienceReplay._to_python_obj,
                    row[1:]
                ))

                samples.append(memory_row)

        states = ExperienceReplay._convert_data_for_model([row[0] for row in samples])
        next_states = ExperienceReplay._convert_data_for_model([row[3] for row in samples])

        next_scores = np.max(
            model.predict(
                next_states
            ),
            axis=1
        )
        y = model.predict(states)

        for idx, val in enumerate(zip(samples, next_scores)):
            data_val, next_score = val
            state, reward, action, next_state, is_final_state = data_val

            updated_score = self._compute_new_score(
                self.time,
                y[idx],
                action,
                reward,
                next_score,
                is_final_state
            )

            y[idx, action] = updated_score
        return states, y

