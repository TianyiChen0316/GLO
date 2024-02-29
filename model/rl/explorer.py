import random


class Explorer:
    def __init__(self, seed=None):
        self.count = 0
        if seed is None:
            seed = random.randrange(65536)
        self.randomizer = random.Random(seed)

    def reset(self):
        self.count = 0

    def step(self):
        self.count += 1

    def state_dict(self):
        return {
            'count': self.count,
            'rand_state': self.randomizer.getstate(),
        }

    def load_state_dict(self, state_dict: dict):
        self.count = state_dict.get('count', self.count)
        if 'rand_state' in state_dict:
            self.randomizer.setstate(state_dict['rand_state'])

    @property
    def prob(self):
        return 0.

    def explore(self, prob=None):
        if prob is None:
            prob = self.prob
        return self.randomizer.random() < prob


class HalfTimeExplorer(Explorer):
    def __init__(self, start, end, half_steps, seed=None):
        super().__init__(seed)
        self.start = start
        self.end = end
        self.half_steps = half_steps

    @property
    def prob(self):
        process = 0.5 ** (self.count / self.half_steps)
        return self.start + (self.end - self.start) * (1 - process)

    def state_dict(self):
        res = super().state_dict()
        res.update(
            start=self.start,
            end=self.end,
            half_steps=self.half_steps,
        )
        return res

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.start = state_dict.get('start', self.start)
        self.end = state_dict.get('end', self.end)
        self.half_steps = state_dict.get('half_steps', self.half_steps)


class LinearExplorer(Explorer):
    def __init__(self, start, end, steps, seed=None):
        super().__init__(seed)
        self.start = start
        self.end = end
        self.steps = steps

    @property
    def prob(self):
        if self.count > self.steps:
            return self.end
        return self.start + (self.end - self.start) * self.count / self.steps

    def state_dict(self):
        res = super().state_dict()
        res.update(
            start=self.start,
            end=self.end,
            steps=self.steps,
        )
        return res

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.start = state_dict.get('start', self.start)
        self.end = state_dict.get('end', self.end)
        self.steps = state_dict.get('steps', self.steps)
