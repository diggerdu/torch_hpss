#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spectrogram decomposition
=========================
.. autosummary::
    :toctree: generated/

    hpss
"""

import numpy as np

import scipy.sparse
from scipy.ndimage import median_filter

def softmask(X, X_ref, power=1, split_zeros=False):

    if X.shape != X_ref.shape:
        raise ParameterError('Shape mismatch: {}!={}'.format(X.shape,
                                                             X_ref.shape))

    if np.any(X < 0) or np.any(X_ref < 0):
        raise ParameterError('X and X_ref must be non-negative')

    if power <= 0:
        raise ParameterError('power must be strictly positive')

    # We're working with ints, cast to float.
    dtype = X.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32

    # Re-scale the input arrays relative to the larger value
    Z = np.maximum(X, X_ref).astype(dtype)
    bad_idx = (Z < np.finfo(dtype).tiny)
    Z[bad_idx] = 1

    # For finite power, compute the softmask
    if np.isfinite(power):
        mask = (X / Z)**power
        ref_mask = (X_ref / Z)**power
        good_idx = ~bad_idx
        mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]
        # Wherever energy is below energy in both inputs, split the mask
        if split_zeros:
            mask[bad_idx] = 0.5
        else:
            mask[bad_idx] = 0.0
    else:
        # Otherwise, compute the hard mask
        mask = X > X_ref

    return mask
def magphase(D, power=1):
    mag = np.abs(D)
    mag **= power
    phase = np.exp(1.j * np.angle(D))
    return mag, phase


def hpss(S, kernel_size=31, power=2.0, mask=False, margin=1.0):
    if np.iscomplexobj(S):
        S, phase = core.magphase(S)
    else:
        phase = 1

    if np.isscalar(kernel_size):
        win_harm = kernel_size
        win_perc = kernel_size
    else:
        win_harm = kernel_size[0]
        win_perc = kernel_size[1]

    if np.isscalar(margin):
        margin_harm = margin
        margin_perc = margin
    else:
        margin_harm = margin[0]
        margin_perc = margin[1]

    # margin minimum is 1.0
    if margin_harm < 1 or margin_perc < 1:
        raise ParameterError("Margins must be >= 1.0. "
                             "A typical range is between 1 and 10.")

    # Compute median filters. Pre-allocation here preserves memory layout.
    harm = np.empty_like(S)
    harm[:] = median_filter(S, size=(1, win_harm), mode='reflect')

    perc = np.empty_like(S)
    perc[:] = median_filter(S, size=(win_perc, 1), mode='reflect')

    split_zeros = (margin_harm == 1 and margin_perc == 1)

    mask_harm = softmask(harm, perc * margin_harm,
                              power=power,
                              split_zeros=split_zeros)

    mask_perc = softmask(perc, harm * margin_perc,
                              power=power,
                              split_zeros=split_zeros)

    if mask:
        return mask_harm, mask_perc

    return ((S * mask_harm) * phase, (S * mask_perc) * phase)



