import math

import numpy as np


# dynamic time warping - squared
def get_metric(s, t):
    return _get_dtw_matrix(s, t, _point_cost_sqr)[-1, -1]


def _get_dtw_matrix(s, t, point_norm):
    n = len(s)
    m = len(t)
    dtw = np.empty((n, m))
    cost = np.empty(m)
    row = np.empty(m)

    _cost_vector(cost, s[0], t, point_norm)
    dtw[0, 0] = cost[0]
    for i in range(1, m):
        dtw[0, i] = dtw[0, i-1] + cost[i]
    for i in range(1, n):
        dtw[i, 0] = point_norm(s[i] - t[0]) + dtw[i-1, 0]

    for i in range(1, n):
        row = dtw[i, :]
        _cost_vector(cost, s[i], t[1:], point_norm)
        _row_prediction(row, cost, dtw[i-1, :])
        _row_adjustment(row, cost)
    return dtw


def _point_cost_sqr(v):
    return np.inner(v, v)


def _cost_vector(cost, ref, mov, point_norm):
    for i in range(len(mov)):
        diff = ref - mov[i]
        cost[i] = point_norm(diff)


def _row_prediction(row, cost, prev_row):
    for i in range(1, len(row)):
        row[i] = cost[i-1] + min(prev_row[i-1], prev_row[i])


def _row_adjustment(row, cost):
    for j in range(1, len(row)):
        if row[j-1] + cost[j-1] < row[j]:   # is true more often than not
            row[j] = row[j-1] + cost[j-1]
