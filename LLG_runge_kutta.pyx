# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate

cimport numpy as np
cimport cython
from libc.math cimport pow


ctypedef np.int_t INT_t

ctypedef np.float64_t DOUBLE_t

ctypedef np.complex128_t COMP_t


cdef cross_product(np.ndarray[DOUBLE_t, ndim=3] vx, np.ndarray[DOUBLE_t, ndim=3] vy, int sys_size):
    cdef np.ndarray[DOUBLE_t, ndim= 3] vout
    vout = np.array([np.identity(sys_size), np.identity(
        sys_size), np.identity(sys_size)])
    vout[0, :, :] = vx[1, :, :] * vy[2, :, :] - vx[2, :, :] * vy[1, :, :]
    vout[1, :, :] = vx[2, :, :] * vy[0, :, :] - vx[0, :, :] * vy[2, :, :]
    vout[2, :, :] = vx[0, :, :] * vy[1, :, :] - vx[1, :, :] * vy[0, :, :]
    return vout


cdef imposePBC(np.ndarray[DOUBLE_t, ndim=2] system_sz, np.ndarray[COMP_t, ndim=2] system_inplane, int sys_size):

    system_sz[0, :] = system_sz[sys_size, :]
    system_inplane[0, :] = system_inplane[sys_size, :]
    system_sz[sys_size + 1, :] = system_sz[1, :]
    system_inplane[sys_size + 1, :] = system_inplane[1, :]
    system_sz[:, 0] = system_sz[:, sys_size]
    system_inplane[:, 0] = system_inplane[:, sys_size]
    system_sz[:, sys_size + 1] = system_sz[:, 1]
    system_inplane[:, sys_size + 1] = system_inplane[:, 1]

    return system_sz, system_inplane


def numeric(int sys_size, np.ndarray[DOUBLE_t, ndim=2] system_sz, np.ndarray[COMP_t, ndim=2] system_inplane, double alpha, list Bext, double J, double step, double current_time, int fintemp, double Temperature):

    cdef np.ndarray[DOUBLE_t, ndim = 2] k1_sz, k2_sz, k3_sz, k4_sz, NORM
    cdef np.ndarray[COMP_t, ndim= 2] k1_inplane, k2_inplane, k3_inplane, k4_inplane
    cdef np.ndarray[DOUBLE_t, ndim= 2] T, predictor_sz
    cdef np.ndarray[COMP_t, ndim= 2] predictor_inplane
    cdef np.ndarray[DOUBLE_t, ndim= 3] TEMP
    cdef np.ndarray[DOUBLE_t, ndim= 3] Gxi, Gxi2

    if fintemp == False:  # zero temperature calculation, 4th-order Runge-Kutta
        k1_sz, k1_inplane = Kfunc(sys_size, system_sz, system_inplane, np.identity(
            sys_size + 2) * 0 * np.exp(-0j), np.identity(sys_size + 2) * 0, current_time, alpha, Bext, J)

        k2_sz, k2_inplane = Kfunc(sys_size, system_sz, system_inplane, k1_inplane *
                                  step / 2., k1_sz * step / 2., current_time + step / 2., alpha, Bext, J)

        k3_sz, k3_inplane = Kfunc(sys_size, system_sz, system_inplane, k2_inplane *
                                  step / 2., k2_sz * step / 2., current_time + step / 2., alpha, Bext, J)

        k4_sz, k4_inplane = Kfunc(sys_size, system_sz, system_inplane,
                                  k3_inplane * step, k3_sz * step, current_time + step, alpha, Bext, J)

        system_sz += step * \
            np.real((k1_sz + 2 * k2_sz + 2 * k3_sz + k4_sz)) / 6.0
        system_inplane += step * \
            (k1_inplane + 2 * k2_inplane + 2 * k3_inplane + k4_inplane) / 6.0

    else:  # finite temperature calculation, Heum method

        TEMP = np.random.randn(3, sys_size + 2, sys_size + 2) * \
            np.sqrt(2 * alpha * step * Temperature)

        k1_sz, k1_inplane = Kfunc(sys_size, system_sz, system_inplane, np.identity(
            sys_size + 2) * 0 * np.exp(-0j), np.identity(sys_size + 2) * 0, current_time, alpha, Bext, J)

        Gxi = Gfunc(sys_size, system_sz, system_inplane, alpha, step, TEMP)

        predictor_sz = k1_sz * step + Gxi[2, :, :]
        predictor_inplane = k1_inplane * step + \
            (Gxi[0, :, :] + 1j * Gxi[1, :, :])

        Gxi2 = Gfunc(sys_size, system_sz + predictor_sz,
                     system_inplane + predictor_inplane, alpha, step, TEMP)

        k2_sz, k2_inplane = Kfunc(sys_size, system_sz, system_inplane,
                                  predictor_inplane, predictor_sz, current_time + step, alpha, Bext, J)

        system_sz += step * np.real((k1_sz + k2_sz)) / \
            2.0 + (Gxi[2, :, :] + Gxi2[2, :, :]) / 2
        system_inplane += step * (k1_inplane + k2_inplane) / 2.0 + (
            Gxi[0, :, :] + 1j * Gxi[1, :, :] + Gxi2[0, :, :] + 1j * Gxi2[1, :, :]) / 2

    system_sz, system_inplane = imposePBC(system_sz, system_inplane, sys_size)

    NORM = np.sqrt(np.real(system_inplane)**2 +
                   np.imag(system_inplane)**2 + np.absolute(system_sz)**2)

    system_inplane = system_inplane / NORM
    system_sz = system_sz / NORM

    return system_sz, system_inplane


cdef Kfunc(int sys_size, np.ndarray[DOUBLE_t, ndim=2] sys_sz, np.ndarray[COMP_t, ndim=2] sys_inplane, np.ndarray[COMP_t, ndim=2] kmat_inplane, np.ndarray[DOUBLE_t, ndim=2] kmat_sz, double time, double alpha, list Bext, double J):

    cdef np.ndarray[COMP_t, ndim = 2] sys_inplane_new
    cdef np.ndarray[DOUBLE_t, ndim = 2] sys_sz_new
    cdef np.ndarray[DOUBLE_t, ndim = 3] v, vu, vd, vr, vl, Dr, Du, H_field, eff_field, vect, vout, system_vect, vnn

    cdef int  li, lj

    kmat_sz, kmat_inplane = imposePBC(kmat_sz, kmat_inplane, sys_size)

    sys_inplane_new = sys_inplane + kmat_inplane
    sys_sz_new = sys_sz + kmat_sz
    system_vect = np.array(
        [np.real(sys_inplane_new), np.imag(sys_inplane_new), np.real(sys_sz_new)])

    v = system_vect[:, 1:sys_size + 1, 1:sys_size + 1]
    vl = system_vect[:, 1:sys_size + 1, 0:sys_size]
    vr = system_vect[:, 1:sys_size + 1, 2:sys_size + 2]
    vu = system_vect[:, 0:sys_size, 1:sys_size + 1]
    vd = system_vect[:, 2:sys_size + 2, 1:sys_size + 1]

    vnn = -J * (vu + vd + vl + vr)

    H_field = np.array([Bext[0] * np.ones((sys_size, sys_size)), Bext[1] *
                        np.ones((sys_size, sys_size)),  Bext[2] * np.ones((sys_size, sys_size))])

    eff_field = H_field + vnn

    vect = (-1. / (1 + pow(alpha, 2))) * (cross_product(v, alpha * cross_product(v, eff_field, sys_size) + eff_field, sys_size))

    vout = np.array([np.real(sys_inplane_new), np.real(
        sys_inplane_new), np.real(sys_inplane_new)])

    vout[:, 1:sys_size + 1, 1:sys_size + 1] = vect

    return vout[2], vout[0] + 1j * vout[1]

cdef Gfunc(int sys_size, np.ndarray[DOUBLE_t, ndim=2] sys_sz, np.ndarray[COMP_t, ndim=2] sys_inplane, double alpha, double step, np.ndarray[DOUBLE_t, ndim=3] TEMP):

    cdef np.ndarray[COMP_t, ndim = 2] sys_inplane_new
    cdef np.ndarray[DOUBLE_t, ndim = 2] sys_sz_new
    cdef np.ndarray[DOUBLE_t, ndim = 3] v, vect, vout, system_vect

    system_vect = np.array(
        [np.real(sys_inplane), np.imag(sys_inplane), np.real(sys_sz)])

    v = system_vect[:, 1:sys_size + 1, 1:sys_size + 1]

    vect = (-1. / (1 + pow(alpha, 2))) * (cross_product(v, alpha * cross_product(v, TEMP[:, 1:sys_size + 1, 1:sys_size + 1], sys_size) + TEMP[:, 1:sys_size + 1, 1:sys_size + 1], sys_size))

    vout = np.array([np.real(sys_inplane), np.real(
        sys_inplane), np.real(sys_inplane)])

    vout[:, 1:sys_size + 1, 1:sys_size + 1] = vect

    return vout
