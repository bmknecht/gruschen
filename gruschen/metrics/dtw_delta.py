import math

import numpy as np


# own dynamic time warping - squared
def get_metric(s, t):
    return _get_dtw_matrix(s, t)[-1, -1]


def _get_dtw_matrix(s, t):
    ds = s[1:] - s[0:-1]
    s = s[1:]
    dt = t[1:] - t[0:-1]
    t = t[1:]

    n = len(s)
    m = len(t)
    dtw = np.empty((n, m))
    cost = np.empty(m)
    dcost = np.empty(m)

    _cost_vector_euclidean(cost, s[0], t)
    _cost_vector_euclidean(dcost, ds[0], dt)
    dtw[0, 0] = cost[0] + dcost[0]
    for i in range(1, m):
        dtw[0, i] = dtw[0, i-1] + cost[i] + dcost[i]
    for i in range(1, n):
        dtw[i, 0] = (_point_cost(s[i] - t[0]) + _point_cost(ds[i] - dt[0]) +
                     dtw[i-1, 0])

    for i in range(1, n):
        _cost_vector_euclidean(cost, s[i], t)
        _cost_vector_euclidean(dcost, ds[i], dt)
        _row_prediction(dtw[i, 1:], cost+dcost, dtw[i-1, :])
        _row_adjustment(dtw[i, 1:], cost+dcost)
    return dtw


def _point_cost(v):
    return math.sqrt(np.inner(v, v))


def _cost_vector_euclidean(cost, ref, mov):
    for i in range(len(mov)):
        diff = ref - mov[i]
        cost[i] = _point_cost(diff)


def _row_prediction(row, cost, prev_row):
    for i in range(len(row)-1):
        row[i+1] = cost[i] + min(prev_row[i], prev_row[i+1])


def _row_adjustment(row, cost):
    for j in range(1, len(row)):
        if row[j-1] + cost[j-1] < row[j]:   # is true 1 out of 5 times
            row[j] = row[j-1] + cost[j-1]
