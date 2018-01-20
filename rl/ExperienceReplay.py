import random
import threading

import numpy as np
import sqlite3
import pickle
from contextlib import closing
from blist import sortedlist
import time
from rl import AsyncMethodExecutor


class DataPacket(object):
    def __init__(self):
        self.data = None


class ExperienceReplay(object):
    def __init__(
        self,
        max_size=100,
        sample_size=32,
        should_pop_oldest=True,
        database_file='memory.db',
        table_name='memory',
        reuse_db=True,
        verbose=False,
    ):
        self.max_size = max_size
        self.sample_size = sample_size

        self._time = 0
        self._should_pop_oldest = should_pop_oldest
        self._size = 0
        self._verbose = verbose
        self._table_name = table_name
        self._ids = sortedlist()
        self._ids_idx = []
        self._last_query_data = None
        self._last_query_time = 0.
        self._db_thread = AsyncMethodExecutor()
        self._db_thread.start()
        self._db_lock = threading.Event()
        self._db_lock.clear()

        self.log("Database initialization started.")
        self._db_thread.run_on_thread(
            self._init_db,
            database_file,
            reuse_db
        )

        self._db_lock.wait()
        self.log("Database initialization complete.")

    def log(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    def _init_db(self, database_file, reuse_db):
        self._database_connection = sqlite3.connect(database_file, isolation_level=None)
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
            
            self.log("Initial memory size:", len(self._ids))

        self._db_lock.set()

    def is_ready(self):
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

    def _add(self, state, reward, action, next_state, is_final_state, scores):
        start_time = time.process_time()
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

        end_time = time.process_time()
        self.log("Insert query time:", end_time - start_time)

    def add(self, state, reward, action, next_state, is_final_state, scores):
        self._db_thread.run_on_thread(
            self._add,
            state,
            reward,
            action,
            next_state,
            is_final_state,
            scores
        )

    def _fetch_sample_from_db(self):
        start_time = time.process_time()
        sample_idx = random.sample(self._ids_idx, self.sample_size)
        ids = [self._ids[idx] for idx in sample_idx]
        id_list = ",".join(map(str, ids))

        with closing(self._database_connection.cursor()) as cursor:
            sqlite_data = cursor.execute(
                'SELECT * FROM %s WHERE id IN (%s)' % (self._table_name, id_list)
            ).fetchall()

            samples = []
            for row in sqlite_data:
                memory_row = list(map(
                    ExperienceReplay._to_python_obj,
                    row[1:]
                ))

                samples.append(memory_row)
        self._last_query_data = samples
        end_time = time.process_time()
        self._last_query_time = end_time - start_time
        self._db_lock.set()

    def _fetch_data_async(self):
        self.log("Async fetching sample.")
        self._db_lock.clear()
        self._db_thread.run_on_thread(
            self._fetch_sample_from_db
        )

    def get_sample_data(self):
        if not self.is_ready():
            return

        start_time = time.process_time()
        self._db_lock.wait()
        end_time = time.process_time()
        self.log("Waiting for async fetch:", end_time - start_time)

        if self._last_query_data is None:
            self.log("Query not fetched. Started async fetch.")
            self._fetch_data_async()
            self._db_lock.wait()

        self.log("Query time:", self._last_query_time)
        self._fetch_data_async()
        self._time += 1

        return self._last_query_data
