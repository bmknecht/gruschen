import math
import unittest

import numpy as np

from . import dynamic_time_warping
from . import linear_stretch


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
        stretched = linear_stretch._stretch_signal(signal, 5)
        for i in range(5):
            self.assertEqual(stretched[i], i)

        signal = np.array([_ for _ in range(0, 20, 2)])
        assert len(signal) == 10
        stretched = linear_stretch._stretch_signal(signal, 19)
        for i in range(19):
            self.assertEqual(stretched[i], i)

    def test_linear_interpolation_stretching(self):
        signal1 = np.array([_ for _ in range(0, 20, 2)])
        signal2 = np.array([_ for _ in range(0, 19)])

        self.assertAlmostEqual(linear_stretch.get_metric(signal1, signal2), 0)
