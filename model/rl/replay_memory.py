import time
from queue import PriorityQueue
from collections.abc import Sequence
from collections import Counter

from lib.syntax import Interface, abstractmethod
from lib.tools.randomizer import Randomizer

class RoundRobinQueue(Sequence):
    def __init__(self, capacity):
        self._capacity = capacity
        self.clear()

    def state_dict(self):
        return {
            'data': self._data,
            'capacity': self._capacity,
            'pointer': self._p,
        }

    def load_state_dict(self, state_dict, load_config=False):
        if 'data' in state_dict:
            self._data = state_dict['data']
        if load_config:
            if 'pointer' in state_dict:
                self._p = state_dict['pointer']
            if 'capacity' in state_dict:
                self._capacity = state_dict['capacity']
        else:
            if 'capacity' in state_dict:
                diff = self._capacity - state_dict['capacity']
                if diff > 0:
                    self._p = 0
                else:
                    if diff < 0:
                        self._data = self._data[:diff]
                    if 'pointer' in state_dict:
                        self._p = state_dict['pointer']
            elif 'pointer' in state_dict:
                self._p = state_dict['pointer']
        return self

    def remove_duplicates(self, key=None):
        data = {}
        old_data = self._data[self._p:] + self._data[:self._p]
        if callable(key):
            for value in old_data:
                data[key(value)] = value
        else:
            for value in old_data:
                data[value] = value
        self._data = list(data.values())
        self._p = 0
        return self

    def clear(self):
        self._data = []
        self._p = 0
        return self

    def push(self, value):
        prev_value = None
        if len(self._data) < self._capacity:
            self._data.append(value)
        else:
            if self._p >= len(self._data):
                self._p = 0
            prev_value = self._data[self._p]
            self._data[self._p] = value
            self._p += 1
        return prev_value

    def __getitem__(self, item):
        return self._data[item]

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __bool__(self):
        return bool(self._data)


class Memory(Interface):
    @abstractmethod
    def load_state_dict(self, state_dict, load_config=False):
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self, size, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def push(self, *args, **kwargs):
        raise NotImplementedError


class ReplayMemory(Memory):
    def __init__(self, size, seed=None):
        self.memory = RoundRobinQueue(size)
        self.randomizer = Randomizer(seed)

    def load_state_dict(self, state_dict, load_config=False):
        if isinstance(state_dict, ReplayMemory):
            return self.load_state_dict(state_dict.state_dict(), load_config)
        if 'memory' in state_dict:
            self.memory.load_state_dict(state_dict['memory'], load_config)
        if 'randomizer' in state_dict:
            self.randomizer.load_state_dict(state_dict['randomizer'])

    def state_dict(self):
        return {
            'memory': self.memory.state_dict(),
            'randomizer': self.randomizer.state_dict(),
        }

    def push(self, value):
        self.memory.push(value)

    def sample(self, size, preserve=0):
        if preserve <= 0:
            return self.randomizer.sample(self.memory, k=min(size, len(self.memory)))
        if size <= preserve:
            return self.memory[-preserve:]
        res = self.randomizer.sample(self.memory[:-preserve], k=min(size - preserve, len(self.memory)))
        res.extend(self.memory[-preserve:])
        return res

    def clear(self):
        self.memory.clear()
        self.randomizer.reset()


class PriorityMemory(ReplayMemory):
    def __init__(self, size, priority_size, seed=None):
        super().__init__(size, seed)
        self._pqs = {}
        self._priority_size = priority_size

    def load_state_dict(self, state_dict, load_config=False):
        if 'memory' in state_dict:
            self.memory.load_state_dict(state_dict['memory'], load_config)
        if 'randomizer' in state_dict:
            self.randomizer.load_state_dict(state_dict['randomizer'])
        if 'pqs' in state_dict:
            _pqs = state_dict['pqs']
            self._pqs = {}
            for k, v in _pqs.items():
                pq = PriorityQueue()
                pq.queue = v
                self._pqs[k] = pq
        if 'priority_size' in state_dict:
            self._priority_size = state_dict['priority_size']

    def state_dict(self):
        return {
            'memory': self.memory.state_dict(),
            'randomizer': self.randomizer.state_dict(),
            'pqs': {k : v.queue for k, v in self._pqs.items()},
            'priority_size': self._priority_size,
        }

    def push_priority_queue(self, sample_set, value, index):
        pq = self._pqs.setdefault(index, PriorityQueue())
        value_tuple = (-value, time.time(), sample_set)
        pq.put_nowait(value_tuple) # the heap top represents the largest value
        if pq.qsize() > self._priority_size:
            pq.get_nowait()

    def clear(self):
        super().clear()
        self._pqs.clear()

    def sample(self, size, preserve=0):
        if preserve <= 0:
            return self.randomizer.sample(self.memory, k=min(size, len(self.memory)))
        if size <= preserve:
            size = preserve
        preserve_pool = []
        for pq in self._pqs.values():
            for _, _, s in pq.queue:
                preserve_pool.extend(s)
        preserve = min(preserve, len(preserve_pool))
        psv = self.randomizer.sample(preserve_pool, k=preserve)
        size = size - preserve
        if size > 0:
            psv.extend(self.randomizer.sample(self.memory, k=min(size, len(self.memory))))
        return psv


class DictPairMemory(Memory):
    def __init__(self, size=128, seed=None):
        self._memories = {}
        self._size = size
        self.randomizer = Randomizer(seed)

    def load_state_dict(self, state_dict, load_config=False):
        if 'memories' in state_dict:
            _memories = state_dict['memories']
            self._memories = {}
            for k, v in _memories.items():
                self._memories[k] = v
        if 'randomizer' in state_dict:
            self.randomizer.load_state_dict(state_dict['randomizer'])
        if 'size' in state_dict:
            self._size = state_dict['size']

    def state_dict(self):
        return {
            'memories': self._memories,
            'randomizer': self.randomizer.state_dict(),
            'size': self._size,
        }

    def push(self, value, key=None):
        memory = self._memories.get(key, None)
        if memory is None:
            memory = ReplayMemory(size=self._size)
            self._memories[key] = memory
        memory.push(value)

    def clear(self):
        self._memories.clear()
        self.randomizer.reset()

    def sample(self, size, preserve=0, tuple_size=2):
        # return *size* pairs of samples
        keys = tuple(self._memories.keys())
        keys = Counter(self.randomizer.choices(keys, k=size))
        res = []
        for k, num_pairs in keys.items():
            memory = self._memories[k]
            _res = memory.sample(num_pairs * tuple_size)
            num_pairs = len(_res) // tuple_size # can be less than the original num_pairs
            _new_res = (_res[i : i + tuple_size] for i in range(0, num_pairs * tuple_size, tuple_size))#zip(_res[:num_pairs], _res[num_pairs:])
            res.extend(_new_res)
        return res

_hash = hash
class HashPairMemory(Memory):
    def __init__(self, seed=None):
        self._memories = {}
        self.randomizer = Randomizer(seed)

    def load_state_dict(self, state_dict, load_config=False):
        if 'memories' in state_dict:
            _memories = state_dict['memories']
            self._memories = {}
            for k, v in _memories.items():
                self._memories[k] = v
        if 'randomizer' in state_dict:
            self.randomizer.load_state_dict(state_dict['randomizer'])

    def state_dict(self):
        return {
            'memories': self._memories,
            'randomizer': self.randomizer.state_dict(),
        }

    def push(self, value, key=None, hash=None):
        memory = self._memories.get(key, None)
        if memory is None:
            memory = {}
            self._memories[key] = memory
        if hash is None:
            hash = _hash(value)
        memory[hash] = value

    def clear(self):
        self._memories.clear()
        self.randomizer.reset()

    def sample(self, size, preserve=0, tuple_size=2):
        # return *size* pairs of samples
        keys = tuple(self._memories.keys())
        keys = Counter(self.randomizer.choices(keys, k=size))
        res = []
        for k, num_pairs in keys.items():
            memory = self._memories[k]
            sample_num = num_pairs * tuple_size
            if sample_num > len(memory):
                _res = tuple(memory.values())
            else:
                _res = self.randomizer.sample(tuple(memory.values()), k=sample_num)#memory.sample(num_pairs * tuple_size)
            num_pairs = len(_res) // tuple_size # can be less than the original num_pairs
            _new_res = (_res[i : i + tuple_size] for i in range(0, num_pairs * tuple_size, tuple_size))#zip(_res[:num_pairs], _res[num_pairs:])
            res.extend(_new_res)
        return res


class BestCache:
    def __init__(self, large_best=False):
        self._large_best = large_best
        self._cache = {}

    def __setitem__(self, key, value):
        assert value is not None
        prev_value = self._cache.get(key, None)
        if prev_value is not None:
            if self._large_best:
                if value > prev_value:
                    self._cache[key] = value
            else:
                if value < prev_value:
                    self._cache[key] = value
        else:
            self._cache[key] = value

    def __getitem__(self, key):
        return self._cache[key]

    def __contains__(self, item):
        return item in self._cache

    def get(self, key, default=None):
        return self._cache.get(key, default)

    def clear(self):
        self._cache.clear()

    def state_dict(self):
        return {
            'cache': self._cache,
            'large_best': self._large_best,
        }

    def load_state_dict(self, state_dict):
        if 'cache' in state_dict:
            self._cache = state_dict['cache']
        if 'large_best' in state_dict:
            self._large_best = state_dict['large_best']
