import time as _time
import typing as _typing

class Timer:
    def __init__(self, callback: _typing.Callable[[_typing.Any], _typing.Any] = None):
        self.__t = []
        self.__time = None
        self.__callback = callback

    def __enter__(self):
        self.__t.append(_time.time())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__time = _time.time() - self.__t.pop()
        if self.__callback is not None:
            self.__callback(self)

    def reset(self):
        self.__time = None

    @property
    def time(self):
        return self.__time

timer = Timer()
