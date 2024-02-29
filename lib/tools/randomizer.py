import random


class Randomizer:
    def __init__(self, seed):
        self._seed = seed
        self._randomizer = random.Random(seed)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.reset()

    def reset(self):
        self._randomizer.seed(self._seed)

    def __getattr__(self, item):
        if item in (
            'getrandbits', 'random',
            'randbytes', 'randrange', 'randint',
            'choice', 'shuffle', 'sample', 'choices',
            'uniform', 'triangular', 'normalvariate',
            'gauss', 'expovariate', 'vonmisesvariate',
            'gammavariate', 'betavariate', 'paretovariate',
            'weibullvariate',
        ):
            return getattr(self._randomizer, item)
        return super().__getattribute__(item)

    def state_dict(self):
        return {
            'seed': self._seed,
            'random_state': self._randomizer.getstate(),
        }

    def load_state_dict(self, state_dict):
        if 'seed' in state_dict:
            self._seed = state_dict['seed']
        if 'random_state' in state_dict:
            self._randomizer.setstate(state_dict['random_state'])


__all__ = ['Randomizer']
