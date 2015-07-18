import math

import numpy as np


# dynamic time warping - squared
def norm_sqr(s, t):
    n = len(s)
    m = len(t)
    dtw = np.zeros((n+1, m+1))
    dtw[1:, 0] = float("inf")
    dtw[0, 1:] = float("inf")
    dtw[0, 0] = 0

    for i in range(1, n+1):
        prev_row = dtw[i-1, :]
        row = dtw[i, :]
        diff = s[i-1] - t
        cost = _cost_vector(diff, m)
        row[1:] = _row_prediction(cost, prev_row)
        _row_adjustment(row, cost, m)
    return dtw[-1, -1]


def _cost_vector(diff, m):
    return np.fromiter((np.inner(d, d) for d in diff), np.float, m)


def _row_prediction(cost, prev_row):
    return cost + np.minimum(prev_row[0:-1], prev_row[1:])


def _row_adjustment(row, cost, m):
    for j in range(1, m+1):
        if row[j-1] + cost[j-1] < row[j]:   # is true 1 out of 5 times
            row[j] = row[j-1] + cost[j-1]
