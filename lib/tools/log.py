import os as _os
import sys as _sys
import logging as _logging
from logging import NOTSET, DEBUG, INFO, WARN, WARNING, ERROR, FATAL, CRITICAL


class _Logger(_logging.Logger):
    def __init__(self, name, level=NOTSET):
        super().__init__(name, level)
        self._file_handler = None
        self._stdout_handler = None
        self._stderr_handler = None
        self.format('[%(levelname)s %(asctime)s.%(msecs)d] %(message)s', '%Y-%m-%d %H:%M:%S')

    def format(self, format : str, date_format : str = None, style='%'):
        self._formatter = _logging.Formatter(format, date_format, style)
        for h in self.handlers:
            h.setFormatter(self._formatter)

    def __call__(self, *value, sep=' ', end='', flush=False, level=INFO):
        if sep is None:
            sep = ' '
        if end is None:
            end = ''
        msg = sep.join(map(str, value)) + end
        self.log(level, msg)
        if flush:
            for handler in self.handlers:
                handler.flush()

def Logger(name, level=None, file=None, to_stderr=False, to_stdout=False):
    ori_logger_class = _logging.getLoggerClass()
    _logging.setLoggerClass(_Logger)
    logger = _logging.getLogger(name)
    if not isinstance(logger, _Logger):
        raise RuntimeError(f"logger '{name}' has already been created")
    if level is not None:
        logger.setLevel(level)
    if file:
        if logger._file_handler is not None:
            logger.removeHandler(logger._file_handler)
        _os.makedirs(_os.path.dirname(file), exist_ok=True)
        logger._file_handler = _logging.FileHandler(file, 'a', 'utf8')
        logger.addHandler(logger._file_handler)
        if logger._formatter:
            logger._file_handler.setFormatter(logger._formatter)
    if to_stdout is not None:
        if to_stdout:
            if logger._stdout_handler is None:
                logger._stdout_handler = _logging.StreamHandler(_sys.stdout)
                logger.addHandler(logger._stdout_handler)
                if logger._formatter:
                    logger._stdout_handler.setFormatter(logger._formatter)
        else:
            if logger._stdout_handler is not None:
                logger.removeHandler(logger._stdout_handler)
                logger._stdout_handler = None
    if to_stderr is not None:
        if to_stderr:
            if logger._stderr_handler is None:
                logger._stderr_handler = _logging.StreamHandler(_sys.stderr)
                logger.addHandler(logger._stderr_handler)
                if logger._formatter:
                    logger._stderr_handler.setFormatter(logger._formatter)
        else:
            if logger._stderr_handler is not None:
                logger.removeHandler(logger._stderr_handler)
                logger._stderr_handler = None
    _logging.setLoggerClass(ori_logger_class)
    return logger


__all__ = ['Logger', 'NOTSET', 'DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'FATAL', 'CRITICAL']
