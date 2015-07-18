import math
import unittest

import numpy as np

import dynamic_time_warping


# unit test
class MetricsTest(unittest.TestCase):
    testSize = 100

    def sine(self):
        return np.array([math.sin(x) for x in range(self.testSize)])

    def test_dynamic_time_warping_sqr(self):
        dtwm = dynamic_time_warping.norm_sqr
        sine = np.array([self.sine().tolist(),
                         self.sine().tolist(),
                         self.sine().tolist()])
        self.assertLess(dtwm(sine, sine), 1)
        self.assertLess(dtwm(sine, sine+0.9),
                        dtwm(sine, sine+1))
        self.assertLess(dtwm(np.zeros((self.testSize, self.testSize)),
                             sine),
                        dtwm(np.zeros((self.testSize, self.testSize)),
                             sine*1.1))
        self.assertLess(dtwm(np.zeros((self.testSize, self.testSize)),
                             np.zeros((2*self.testSize, self.testSize))+1),
                        dtwm(np.zeros((self.testSize, self.testSize)),
                             np.zeros((2*self.testSize, self.testSize))+1.1))
        self.assertAlmostEqual(dtwm(sine, sine*1.1),
                               dtwm(sine*1.1, sine))

    def test_stretch_signal(self):
        signal = [0, 2, 4]
        stretched = _stretch_signal(signal, 5)
        for i in range(5):
            self.assertEqual(stretched[i], i)

        signal = np.array([_ for _ in range(0, 20, 2)])
        assert len(signal) == 10
        stretched = _stretch_signal(signal, 19)
        for i in range(19):
            self.assertEqual(stretched[i], i)

    def test_linear_interpolation_stretching(self):
        signal1 = np.array([_ for _ in range(0, 20, 2)])
        signal2 = np.array([_ for _ in range(0, 19)])

        self.assertAlmostEqual(linear_interpolation_stretching(signal1,
                                                               signal2), 0)


def linear_interpolation_stretching(signal1, signal2):
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
