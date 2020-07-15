# -*- coding: utf-8 -*-
"""
voxelwise T1 mapping in C++

contributors: Yoon-Chul Kim, Khu Rai Kim, Hyelee Lee
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
import copy
import pickle
import skimage.morphology as sm
import time
import pydicom as dicom

sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'x64/Release'))

import t1_mapping as t1  # t1_mapping.pyd

from skimage import data, io, filters
from numpy import arange, sin, pi
from scipy.optimize import curve_fit
from math import floor, sqrt


def func_orig(x, a, b, c):
    return a*(1-np.exp(-b*x)) + c


def calculate_T1map_cpp_rd(ir_img, inversiontime, multicore_flag, zoom=2, zoom_lenz=16):
    '''
    implementation of Barral's method
    '''
    if inversiontime[-1] == 0:
        inversiontime = inversiontime[0:-1]
        if ir_img.shape[2] > inversiontime.shape[0]:
            ir_img = ir_img[:,:,0:ir_img.shape[2]-1]

    zoom_lenz = 16
    shape = ir_img.shape

    flat_t1map = t1.fit_t1_barral(ir_img.flatten().astype(np.float64),
                                  np.array(inversiontime, dtype=np.float64),
                                  shape[0], shape[1], shape[2],
                                  zoom, zoom_lenz, multicore_flag)

    t1_map = np.reshape(flat_t1map, [shape[0], shape[1], 3])

    return t1_map


def calculate_T1map_cpp_lm(ir_img, inversiontime, multicore_flag):

    nx, ny, nti = ir_img.shape
    y = np.zeros(nti)

    prtno = 1  # post
    if prtno == 0:  # pre
        p0_initial = [350, 0.001, -150]
    else:  # post
        p0_initial = [350, 0.005, -150]

    if inversiontime[-1] == 0:
        nTI = 7
        inversiontime = inversiontime[0:-1]
        y = y[0:-1]
    else:
        nTI = 8

    err_tol = 1e-7
    ir_img = ir_img[:,:, :nTI]
    shape = ir_img.shape
    flat_t1map = t1.fit_t1(ir_img.flatten(),
                           np.array(inversiontime),
                           np.array(p0_initial),
                           shape[0], shape[1], shape[2],
                           err_tol, multicore_flag)

    t1_map = np.reshape(flat_t1map, [nx, ny, 3])

    return t1_map

