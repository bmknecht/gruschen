import math
import unittest

import numpy as np

from . import dynamic_time_warping as dtw
from . import linear_stretch


# unit test
class MetricsTest(unittest.TestCase):
    testSize = 100

    def sine(self):
        return np.array([math.sin(x) for x in range(self.testSize)])

    def test_dtw_get_metric(self):
        testfunc = dtw.get_metric
        sine = np.array([self.sine().tolist(),
                         self.sine().tolist(),
                         self.sine().tolist()])
        self.assertLess(testfunc(sine, sine), 1)
        self.assertLess(testfunc(sine, sine+0.9),
                        testfunc(sine, sine+1))
        self.assertLess(testfunc(np.zeros((self.testSize, self.testSize)),
                                 sine),
                        testfunc(np.zeros((self.testSize, self.testSize)),
                                 sine*1.1))
        self.assertLess(testfunc(np.zeros((self.testSize, self.testSize)),
                                 np.zeros((2*self.testSize, self.testSize))+1),
                        testfunc(np.zeros((self.testSize, self.testSize)),
                                 np.zeros((2*self.testSize,
                                           self.testSize))+1.1))
        self.assertAlmostEqual(testfunc(sine, sine*1.1),
                               testfunc(sine*1.1, sine))

    def test_dtw_get_dtw_matrix(self):
        testfunc = dtw._get_dtw_matrix
        sine = np.array([self.sine().tolist(),
                         self.sine().tolist(),
                         self.sine().tolist()])
        self.assertAlmostEqual(testfunc(sine, sine)[-1, -1],
                               dtw.get_metric(sine, sine))
        self.assertTrue((sine == self.sine()).all())

    def test_dtw_cost_vector_euclidean(self):
        vec = [np.array([1, 2]), np.array([2, -3])]
        dist_sqr = [0, 0]
        dtw._cost_vector_euclidean(dist_sqr, 0, vec)
        self.assertEqual(dist_sqr[0], math.sqrt(5))
        self.assertEqual(dist_sqr[1], math.sqrt(13))

    def test_dtw_row_prediction(self):
        cost = np.array([11 for _ in range(6)])
        vec = np.array([1, 2, 3, 4, 5, 6, 7])
        res = np.empty(len(vec)-1)
        dtw._row_prediction(res, cost, vec)
        self.assertTrue((res - np.array([12, 13, 14, 15, 16, 17]) < 1e-7).all())

    def test_dtw_row_adjustment(self):
        cost = np.array([1, 2, -1, 4, 11, -9, -1])
        row = np.array([1, -4, 1, 8, 7, -13, 2])
        dtw._row_adjustment(row, cost)
        self.assertTrue((row == np.array([1, -4, -2, -3, 1, -13, -22])).all())

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
