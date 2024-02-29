import os
import pickle
from pathlib import Path as _Path
import typing as _typing

import torch

_DEFAULT_TEMP_FILE = '.%s.tmp'

_PathLike = _typing.Union[str, os.PathLike]

def _safe_save(obj, path : _PathLike, save_func : _typing.Callable[[_typing.Any, _PathLike], _typing.Any], temp_file : str = None):
    if temp_file is None:
        temp_file = _DEFAULT_TEMP_FILE

    path = _Path(path)
    os.makedirs(path.parent, exist_ok=True)
    temp_file_path = path.parent / (temp_file % path.name)

    res = save_func(obj, temp_file_path)

    if not os.path.isfile(temp_file_path):
        # failed to save the object to temp file
        raise FileNotFoundError(str(temp_file_path))

    if os.path.exists(path):
        os.remove(path)

    os.rename(temp_file_path, path)
    return res

def _safe_load(path : _PathLike, load_func : _typing.Callable[[_PathLike], _typing.Any], temp_file : str = None):
    if temp_file is None:
        temp_file = _DEFAULT_TEMP_FILE

    _path = _Path(path)
    temp_file_path = _path.parent / (temp_file % _path.name)

    if os.path.isfile(_path):
        return load_func(_path)

    if os.path.isfile(temp_file_path):
        res = load_func(temp_file_path)
        try:
            os.rename(temp_file_path, _path)
        except Exception as e:
            pass
        return res

    raise FileNotFoundError(path)

def file_exists(path : _PathLike, temp_file : str = None):
    if temp_file is None:
        temp_file = _DEFAULT_TEMP_FILE

    _path = _Path(path)
    temp_file_path = _path.parent / (temp_file % _path.name)

    return os.path.isfile(_path) or os.path.isfile(temp_file_path)

def save_pickle(obj, path : _PathLike, temp_file : str = None):
    def save(o, p):
        with open(p, 'wb') as f:
            res = pickle.dump(o, f)
        return res
    return _safe_save(obj, path, save, temp_file)

def load_pickle(path : _PathLike, temp_file : str = None):
    def load(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    return _safe_load(path, load, temp_file)

def save_torch(obj, path : _PathLike, temp_file : str = None):
    def save(o, p):
        return torch.save(o, p)
    return _safe_save(obj, path, save, temp_file)

def load_torch(path : _PathLike, map_location = None, temp_file : str = None):
    def load(p):
        return torch.load(p, map_location=map_location)
    return _safe_load(path, load, temp_file)
