import numpy as np


class Signal:
    def __init__(self, values, sample_rate):
        assert not isinstance(values, Signal)
        self._values = np.array(values)
        self.sampleRate = sample_rate

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return self._values.__iter__()

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Signal(self._values[key], self.sampleRate)
        return self._values[key]

    def __setitem__(self, key, item):
        self._values[key] = item

    def __add__(self, x):
        return Signal(x + self._values, self.sampleRate)

    def __sub__(self, x):
        return Signal(self._values - x, self.sampleRate)

    def __mul__(self, x):
        return Signal(x * self._values, self.sampleRate)

    def __div__(self, x):
        return Signal(self._values / x, self.sampleRate)

    def min(self):
        return self._values.min()

    def max(self):
        return self._values.max()

    def mean(self):
        return self._values.mean()

    def asnparray(self):
        return self._values

    def __abs__(self):
        return Signal(abs(self._values), self.sampleRate)

    def copy(self):
        return Signal(self._values[:], self.sampleRate)
