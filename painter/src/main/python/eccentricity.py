"""
Modified from scikit-image, as calling the scikit-image eccentricity method was giving "zsh: bus error" for certain input.
See also https://github.com/Abe404/root_painter/issues/22

This code is a modified from code in scikit-image
Which can be viewed online at:
https://github.com/scikit-image/scikit-image/blob/1188c33c3defefdc0f672870b5bcc36beaec967f/skimage/measure/_regionprops.py
https://github.com/scikit-image/scikit-image/blob/1188c33c3defefdc0f672870b5bcc36beaec967f/skimage/measure/_moments.py

Original Work : Copyright (C) 2019, the scikit-image team
Modified work : Copyright (C) 2019 Abraham Smith

All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of skimage nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import os
from math import sqrt
import itertools

import numpy as np
from skimage import measure
from skimage.measure import _moments
from PIL import Image



def eccentricity_for_region(region):

    image = region.image.astype(np.uint8)
    # compute centroid for image

    M = _moments.moments_central(image, center=(0,) * image.ndim, order=1)

    center = (M[tuple(np.eye(image.ndim, dtype=int))]  # array of weighted sums
                                                       # for each axis
              / M[(0,) * image.ndim])  # weighted sum of all points
    print('after get center', center)
    order = 3
    calc = image.astype(float)
    print('after make float')
    for dim, dim_length in enumerate(image.shape):
        delta = np.arange(dim_length, dtype=float) - center[dim]
        powers_of_delta = delta[:, np.newaxis] ** np.arange(order + 1)
        calc = np.rollaxis(calc, dim, image.ndim)
        calc = np.dot(calc, powers_of_delta)
        calc = np.rollaxis(calc, -1, dim)
    M = calc # moments
    #M = _moments.moments(region_image, 3)
    print('M')
    local_centroid = tuple(M[tuple(np.eye(region._ndim, dtype=int))] /  M[(0,) * region._ndim])
    print('local_centroid')
    mu = _moments.moments_central(region.image.astype(np.uint8), local_centroid, order=3)
    print('mu')
    inertia_tensor =  _moments.inertia_tensor(region.image, mu)
    print('intertia_tensor')
    inertia_tensor_eigvals = _moments.inertia_tensor_eigvals(region.image, T=inertia_tensor)
    print('intertia_tensor_eigvals')
    l1, l2 = inertia_tensor_eigvals
    print('l1 l2')
    if l1 == 0:
        return 0
    return sqrt(1 - l2 / l1)


def centroid(image):
    M = moments_central(image, center=(0,) * image.ndim, order=1)
    center = (M[tuple(np.eye(image.ndim, dtype=int))]  # array of weighted sums
                                                       # for each axis
              / M[(0,) * image.ndim])  # weighted sum of all points
    return center


def moments_central(image, center=None, order=3):
    if center is None:
        center = centroid(image)
    calc = image.astype(float)
    for dim, dim_length in enumerate(image.shape):
        delta = np.arange(dim_length, dtype=float) - center[dim]
        powers_of_delta = delta[:, np.newaxis] ** np.arange(order + 1)
        calc = np.rollaxis(calc, dim, image.ndim)
        calc = np.dot(calc, powers_of_delta)
        calc = np.rollaxis(calc, -1, dim)
    return calc

def inertia_tensor(image, mu=None):
    if mu is None:
        mu = moments_central(image, order=2)  # don't need higher-order moments
    mu0 = mu[(0,) * image.ndim]
    result = np.zeros((image.ndim, image.ndim))
    corners2 = tuple(2 * np.eye(image.ndim, dtype=int))
    d = np.diag(result)
    d.flags.writeable = True
    d[:] = (np.sum(mu[corners2]) - mu[corners2]) / mu0
    for dims in itertools.combinations(range(image.ndim), 2):
        mu_index = np.zeros(image.ndim, dtype=int)
        mu_index[list(dims)] = 1
        result[dims] = -mu[tuple(mu_index)] / mu0
        result.T[dims] = -mu[tuple(mu_index)] / mu0
    return result

def inertia_tensor_eigvals(image, mu=None, T=None):
    if T is None:
        T = inertia_tensor(image, mu)
    eigvals = np.linalg.eigvalsh(T)
    eigvals = np.clip(eigvals, 0, None, out=eigvals)
    return sorted(eigvals, reverse=True)

def get_inertia_tensor_eigvals(region):
    return inertia_tensor_eigvals(region.image, T=inertia_tensor(region.image))

def eccentricity2(region):
    l1, l2 = get_inertia_tensor_eigvals(region)
    if l1 == 0:
        return 0
    return sqrt(1 - l2 / l1)
