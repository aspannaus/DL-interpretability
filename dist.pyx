#!/usr/bin/env python3
#cython: language_level=3

import numpy as np
cimport numpy as cnp
from numpy.linalg import norm

cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport floor
from libc.math cimport fabs
from libc.math cimport log
from libc.math cimport pow
from libc.math cimport sqrt


ctypedef fused my_type:
    int
    double
    long long


DTYPE = np.int_
ctypedef cnp.int_t DTYPE_t


cdef inline double float_max(double a, double b): return a if a >= b else b
cdef inline double float_min(double a, double b): return a if a <= b else b

cdef double random():
    cdef double r = rand()
    return r / float(RAND_MAX)


@cython.boundscheck(False)
@cython.wraparound(False)
def sample_wo_replacement(int N, int n, cnp.ndarray[my_type, ndim=1] samples):
    cdef int t = 0
    cdef Py_ssize_t m = 0
    cdef double u

    if my_type is int:
        dtype = np.int_
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong

    for m in range(n):
        u = random()
        if (N-t) * u >= (n-m):
            t += 1
        else:
            samples[m] = t
            t == 1
    return samples


@cython.boundscheck(False)
@cython.wraparound(False)
def hausdorff_dist(cnp.ndarray[double, ndim=2] u, cnp.ndarray[double, ndim=2] dist,
                   int N, double g, Py_ssize_t n_cols):
    cdef Py_ssize_t i, j, k, idx, idx1
    cdef int const = 10
    cdef double mi, mj
    cdef double eta = 0.001
    cdef int m = int(floor(n_cols / (pow(log(n_cols)/log(const), 1 + eta))))
    m = min(m, n_cols-1)

    cdef double val = 0.0
    cdef double delta, lip, r
    delta = 0.0
    lip = 0.0

    cdef cnp.ndarray[DTYPE_t, ndim=1] samples = np.zeros(m, dtype=DTYPE)
    for i in range(N):
        samples = sample_wo_replacement(n_cols, m, samples)
        val = 0
        for j in range(n_cols):
            idx = samples[0]
            mj = dist[j, idx]
            if dist[j, 0] < 1e-10:
                mi = 0
            else:
                mi = my_norm(u[j], u[0], n_cols) / dist[j, 0]
            for k in range(1, m):
                idx1 = samples[k]
                mj = float_min(mj, dist[j, idx1])
            for k in range(j+1, n_cols):
                if dist[j, k] < 1e-10:
                    mi = float_max(mi, 0.0)
                else:
                    mi = float_max(mi, my_norm(u[j], u[k], n_cols) / dist[j, k])
            lip = float_max(lip, mi)
            val = float_max(val, mj)
        delta += val

    delta = delta / float(N)
    r = (lip * delta) / g
    return tuple((lip, delta, g, r))


@cython.boundscheck(False)
@cython.wraparound(False)
def my_norm(cnp.ndarray[double, ndim=1] u, cnp.ndarray[double, ndim=1] v, Py_ssize_t n_cols):
    cdef double res = 0.0
    cdef double tmp
    cdef Py_ssize_t i

    for i in range(n_cols):
        tmp = fabs(u[i] - v[i])
        res += (tmp * tmp)

    return sqrt(res)
