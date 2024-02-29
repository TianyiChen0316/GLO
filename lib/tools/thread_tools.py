import threading as _threading


class LoopThread(_threading.Thread):
    def __init__(self, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None):
        if kwargs is None:
            kwargs = {}
        super().__init__(None, target, name, args, kwargs, daemon=daemon)
        self._stop_event = _threading.Event()
        self._stopped_event = _threading.Event()
        self._suspend_event = _threading.Event()
        self._resume_event = _threading.Event()
        self._lock = _threading.Lock()
        self._target = target
        self._args = args
        self._kwargs = kwargs

    def stop(self, block=False):
        self._stop_event.set()
        if block:
            self._stopped_event.wait()

    def suspend(self):
        self._lock.acquire()
        self._suspend_event.set()
        self._lock.release()

    def resume(self):
        self._lock.acquire()
        if self._suspend_event.is_set():
            self._resume_event.set()
            self._suspend_event.clear()
        self._lock.release()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self) -> None:
        try:
            if self._target is not None:
                while True:
                    self._target(*self._args, **self._kwargs)
                    if self._suspend_event.is_set():
                        self._resume_event.wait()
                        self._lock.acquire()
                        self._resume_event.clear()
                        self._lock.release()
                    if self.stopped():
                        self._stopped_event.set()
                        break
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs
