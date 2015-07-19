import math
import unittest

import numpy as np

from . import dynamic_time_warping


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
