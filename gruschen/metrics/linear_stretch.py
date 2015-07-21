import numpy as np
from scipy.interpolate import interp1d


def get_metric(signal1, signal2):
    if len(signal1) < len(signal2):
        signal1 = _stretch_signal(signal1, len(signal2))
    elif len(signal2) < len(signal1):
        signal2 = _stretch_signal(signal2, len(signal1))
    norm_sum = 0
    for i in range(len(signal1)):
        norm_sum += np.linalg.norm(signal1[i] - signal2[i])
    return norm_sum


def _stretch_signal(signal, new_length):
    assert len(signal) < new_length
    f = interp1d([_ for _ in range(len(signal))], signal, kind='cubic', axis=0)
    return np.array([f(t) for t in np.linspace(0, len(signal)-1, new_length)])
