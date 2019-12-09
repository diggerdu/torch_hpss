#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Du Xingjian
# E-Mail:diggerdu97@gmail.com
########################

import numpy as np

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmask(X, X_ref, power=1, split_zeros=False):

    if X.shape != X_ref.shape:
        raise ParameterError('Shape mismatch: {}!={}'.format(X.shape,
                                                             X_ref.shape))
    if torch.any(X < 0) or torch.any(X_ref < 0):
        raise ParameterError('X and X_ref must be non-negative')

    if power <= 0:
        raise ParameterError('power must be strictly positive')

    # We're working with ints, cast to float.
    dtype = X.dtype
    try:
        assert dtype in [torch.float16, torch.float32, torch.float64]
    except:
        raise ParameterError('data type error')
    
    # Re-scale the input arrays relative to the larger value
    Z = torch.max(X, X_ref)
    bad_idx = (Z < torch.finfo(dtype).tiny)
    if  bad_idx.sum() > 0: 
        Z[bad_idx] = 1
        __import__('pdb').set_trace() 

    # For finite power, compute the softmask
    if np.isfinite(power):
        mask = (X / Z)**power
        ref_mask = (X_ref / Z)**power
        good_idx = ~bad_idx
        #mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]
        mask /= mask + ref_mask
        # Wherever energy is below energy in both inputs, split the mask
        '''
        if split_zeros:
            mask[bad_idx] = 0.5
        else:
            mask[bad_idx] = 0.0
        '''
    else:
        # Otherwise, compute the hard mask
        mask = X > X_ref

    return mask

class HPSS(nn.Module):
    def __init__(self, kernel_size, power=2.0, mask=False, margin=1.0):
        super(HPSS, self).__init__()
        win_harm = win_perc = kernel_size
        self.harm_median_filter = MedianBlur(kernel_size=(1, win_harm))
        self.perc_median_filter = MedianBlur(kernel_size=(win_perc, 1))
     

    def forward(self, S, power=2.0, mask=False, margin=1.0):
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
        harm = self.harm_median_filter(S)
        perc = self.perc_median_filter(S)


        split_zeros = (margin_harm == 1 and margin_perc == 1)

        mask_harm = softmask(harm, perc * margin_harm,
                                  power=power,
                                  split_zeros=split_zeros)

        mask_perc = softmask(perc, harm * margin_perc,
                                  power=power,
                                  split_zeros=split_zeros)

        if mask:
            return mask_harm, mask_perc

        return {"harm_spec" : S * mask_harm, "perc_spec" : S * mask_perc}



def get_binary_kernel2d(window_size: Tuple[int, int]) -> torch.Tensor:
    r"""Creates a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])

def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: Tuple[int, ...] = tuple([(k - 1) // 2 for k in kernel_size])
    return computed[0], computed[1]


class MedianBlur(nn.Module):
    r"""Blurs an image using the median filter.

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = kornia.filters.MedianBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super(MedianBlur, self).__init__()
        self.kernel: torch.Tensor = get_binary_kernel2d(kernel_size)
        self.padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    def forward(self, input: torch.Tensor):  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # map the local window to single vector
        features: torch.Tensor = F.conv2d(
            input, kernel, padding=self.padding, stride=1, groups=c)
        features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

        # compute the median along the feature axis
        median: torch.Tensor = torch.median(features, dim=2)[0]
        return median


# functiona api


def median_blur(input: torch.Tensor,
                kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Blurs an image using the median filter.

    See :class:`~kornia.filters.MedianBlur` for details.
    """
    return MedianBlur(kernel_size)(input)


if __name__ == "__main__":
    import librosa
    y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)              
    D = librosa.stft(y)
    S = torch.from_numpy(np.abs(D)[None, None, ...])
    hpss_module = HPSS(kernel_size=31)
    res = hpss_module(S)


