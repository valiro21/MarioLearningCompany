import queue
import threading


class AsyncMethodExecutor(threading.Thread):
    def __init__(self, q=None, loop_time = 1.):
        self.q = queue.Queue() if q is None else q
        self.timeout = loop_time
        self.should_finalize = False
        super(AsyncMethodExecutor, self).__init__()

    def finalize(self):
        self.should_finalize = True

    def run_on_thread(self, function, *args, **kwargs):
        if not self.should_finalize:
            self.q.put((function, args, kwargs))

    def run(self):
        while True:
            try:
                function, args, kwargs = self.q.get(timeout=self.timeout)
                function(*args, **kwargs)
            except queue.Empty:
                if self.should_finalize:
                    break
                pass
