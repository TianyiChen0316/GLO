import sys
import os
from pathlib import Path
import typing

from tqdm import tqdm
import torch

from lib.torch.safe_save import save_pickle, load_pickle, save_torch, load_torch
from lib.tools import timer

from ..core.sql_featurizer import database, Sql, PlanParser
from ..core.plan_featurizer import Plan, Operators

class CacheManager:
    def __init__(self, cache_file=None, auto_save_interval=0):
        self._cache = {}
        self._cache_file = Path(cache_file) if cache_file else None
        self._auto_save_interval = auto_save_interval
        self._auto_save_count = 0
        if self._cache_file is not None:
            self.restore()

    def restore(self):
        if self._cache_file is None or not os.path.isfile(self._cache_file):
            return
        state_dict = load_pickle(self._cache_file)
        self.load_state_dict(state_dict)

    def flush(self):
        if self._cache_file is None:
            return
        save_pickle(self.state_dict(), self._cache_file)

    def _update_count(self):
        self._auto_save_count += 1
        if self._auto_save_interval > 0 and self._auto_save_count >= self._auto_save_interval:
            self._auto_save_count = 0
            self.flush()

    def put(self, key, value, hash_key=None):
        if hash_key is None:
            hash_key = hash(key)
        self._cache[hash_key] = (key, value)
        self._update_count()

    def update(self, hash_key, value):
        origin = self._cache.get(hash_key, None)
        if origin is not None:
            self._cache[hash_key] = (origin[0], value)
        self._update_count()

    def get(self, hash_key, default=None, get_key=False):
        res = self._cache.get(hash_key, None)
        if res is None:
            return default
        if get_key:
            return res
        return res[1]

    def __setitem__(self, key, value):
        self.put(key, value)

    def __getitem__(self, item):
        return self.get(item)

    def __len__(self):
        return len(self._cache)

    def __bool__(self):
        return len(self._cache) == 0

    def state_dict(self):
        return {
            'cache': self._cache,
        }

    def load_state_dict(self, state_dict):
        if 'cache' in state_dict:
            self._cache = state_dict['cache']


def plan_latency(sql : typing.Union[Plan, Sql], cache_manager : CacheManager = None, detail = False):
    if isinstance(sql, Plan):
        hash_key = f'{sql.sql.filename} {sql.hints(True)}'
    elif isinstance(sql, Sql):
        hash_key = f'{sql.filename}'
    else:
        hash_key = str(sql)
    if cache_manager is not None:
        cache_value = cache_manager.get(hash_key, None)
        if cache_value is not None:
            if detail:
                return cache_value
            return cache_value['latency']
    key = str(sql)
    with timer:
        origin = str(sql.sql) if isinstance(sql, Plan) else None
        db_latency, plan = database.latency(key, origin, return_plan=True, cache=cache_manager is None)
    timer_latency = timer.time * 1000
    res = {
        'latency': db_latency,
        'timer_latency': timer_latency,
        'plan': plan,
    }
    if cache_manager is not None:
        cache_manager.put(key, res, hash_key)
    if detail:
        return res
    return res['latency']


class PlanManager:
    def __init__(self, sqls=None):
        self.data = {}
        self.timeout = None
        self.max_time = None
        if sqls:
            self.init(sqls)

    def state_dict(self):
        return {
            'data': self.data,
            'timeout': self.timeout,
            'max_time': self.max_time
        }

    def load_state_dict(self, state_dict):
        res = state_dict.get('data', None)
        if res is not None:
            self.data = res
        res = state_dict.get('timeout', NotImplemented)
        if res is not NotImplemented:
            self.timeout = res
        res = state_dict.get('max_time', NotImplemented)
        if res is not NotImplemented:
            self.max_time = res

    def init(self, sqls, cache_manager : CacheManager = None, set_timeout=True, verbose=False):
        if verbose:
            sqls = tqdm(sqls)
        costs = []
        for sql in sqls:
            if verbose:
                sqls.set_postfix({'sql': sql.filename})
            res = plan_latency(sql, cache_manager, detail=True)
            _cost, base_plan_dict = res['latency'], res['plan']
            if base_plan_dict is None:
                raise RuntimeError(f'failed to execute baseline of {sql.filename}')

            costs.append(_cost)

            base_plan_parsed = PlanParser(base_plan_dict)
            baseline = base_plan_parsed.join_order
            is_left_deep = self._action_sequence_is_left_deep(baseline)

            valid = baseline and len(baseline) + 1 == len(sql.aliases)
            if not valid:
                if baseline:
                    print(f'Warning: baseline aliases {base_plan_parsed._aliases} does not match sql aliases {sql.aliases}, might be because of subqueries')
                print(f'Warning: Baseline of SQL {sql.filename} is not valid', file=sys.stderr)
                continue

            plan = Plan(sql)
            new_baseline = []
            for left, right, join in baseline:
                plan.join(left, right)
                new_baseline.append((left, right, Operators.default))

            plan_res = plan_latency(plan, cache_manager, detail=False)
            self.data[str(sql)] = {
                'value': plan_res,
                'actions': tuple(new_baseline),
                'parsed_plan': base_plan_parsed,
                'is_left_deep': is_left_deep,
                'base_value': _cost,
            }
        cache_manager.flush()
        self.max_time = max(costs)
        self.set_timeout(set_database=set_timeout)

    def set_timeout(self, set_database=True):
        if self.max_time:
            self.timeout = max(60000, int(database.config.sql_timeout_limit * self.max_time))
        if self.timeout and set_database:
            database.timeout = self.timeout
        print(f'Set timeout limit to {database.timeout}', file=sys.stderr)

    def _action_sequence_is_left_deep(self, actions):
        for index, (left, right, action) in enumerate(actions):
            if index > 0:
                if not isinstance(left, int) ^ isinstance(right, int):
                    return False
        return True

    def subset(self, sqls):
        res = {}
        costs = []
        for sql in sqls:
            sql = str(sql)
            _res = self.data.get(sql, None)
            if _res is not None:
                res[sql] = _res
                costs.append(_res['base_value'])
        new_object = self.__class__()
        new_object.data = res
        new_object.max_time = max(costs)
        new_object.set_timeout(False)
        return new_object

    def update(self, sql, actions, value, parsed_plan : PlanParser = None):
        s = str(sql)
        prev = self.data.get(s, None)
        if prev is None or value < prev['value']:
            actions = tuple(actions)
            is_left_deep = self._action_sequence_is_left_deep(actions)
            if prev is None:
                base_value = None
            else:
                base_value = prev['base_value']
            if isinstance(parsed_plan, dict):
                parsed_plan = PlanParser(parsed_plan)
            self.data[s] = {
                'value': value,
                'actions': actions,
                'parsed_plan': parsed_plan,
                'is_left_deep': is_left_deep,
                'base_value': base_value,
            }

    def get(self, sql, detail=False):
        res = self.data.get(str(sql), None)
        if res is not None:
            if detail:
                return res
            return res['actions']
        return None

def load_dataset(path, device=None, use_cache=True, verbose=False):
    if device is None:
        device = torch.device('cpu')

    path = Path(path)
    cache_file = path.parent / f'.{path.name}.dataset.pkl'

    if use_cache:
        if os.path.isfile(cache_file):
            if verbose:
                print(f'Loading dataset from cached file {cache_file}', file=sys.stderr)
            res = load_torch(cache_file, map_location=device)
            return res

    sql_files = []
    for parent, dirs, files in os.walk(path):
        parent = Path(parent)
        for file in files:
            file = parent / file
            if file.suffix == '.sql':
                sql_files.append(file)
    res = []
    if verbose:
        sql_files = tqdm(sql_files, desc=f'Loading dataset from {path}')
    for sql_file in sql_files:
        with open(sql_file, 'r') as f:
            sql_str = f.read()
        sql = Sql(sql_str, filename=sql_file.name, device=device)
        res.append(sql)

    if use_cache:
        save_torch(res, cache_file)

    return res
