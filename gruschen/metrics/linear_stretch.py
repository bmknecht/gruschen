import numpy as np


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
    return np.array([_interpolate_value(signal,
                                        i,
                                        new_length)
                     for i in range(new_length)])


def _interpolate_value(signal, new_length_index, new_length):
    assert new_length_index < new_length
    assert len(signal) < new_length
    old_length_exact_index = (new_length_index / (new_length-1) *
                              (len(signal)-1))
    old_length_index = int(old_length_exact_index)
    if old_length_index != len(signal) - 1:
        length_factor = old_length_exact_index - old_length_index
        return ((1 - length_factor) * signal[old_length_index] +
                length_factor * signal[old_length_index+1])
    else:
        return signal[-1]
