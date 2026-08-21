"""Centered 2D FFT helpers and zero padding / cropping operators.

These are the frequency-domain building blocks of the imaging model
(Section 2.1 of the paper):

    M   : circulant convolution with the PSF k, i.e.  M = F^H diag(F k) F
    M^H : adjoint of M, i.e.  M^H = F^H diag(conj(F k)) F
    C   : cropping operator;  C^H : zero-padding operator (its adjoint)

All convolutions involved in the reconstruction algorithms are computed in
the frequency domain for efficiency (see Supplement 1, Eqs. (S6)-(S7)).
"""

import torch
import torch.fft as fft
import torch.nn.functional as nf


def toFreq(img, norm="backward"):
    """Spatial domain -> centered frequency domain."""
    return fft.fftshift(fft.fft2(fft.ifftshift(img, dim=(-2, -1)), norm=norm), dim=(-2, -1))


def fromFreq(img, norm="backward"):
    """Centered frequency domain -> spatial domain."""
    return fft.fftshift(fft.ifft2(fft.ifftshift(img, dim=(-2, -1)), norm=norm), dim=(-2, -1))


def pad(img, pad_size=None):
    """Centered zero padding: (..., H, W) -> (..., H + pad_size[0], W + pad_size[1]).

    Note: a negative pad_size crops the image instead, which is used to
    implement the cropping operators C1 / C2 with the adjoint of nf.pad.
    """
    if pad_size is None:
        pad_size = (img.shape[-2] - 1, img.shape[-1] - 1)
    pad = (pad_size[-1] // 2, (pad_size[-1] + 1) // 2,
           pad_size[-2] // 2, (pad_size[-2] + 1) // 2)

    return nf.pad(img, pad=pad, mode='constant', value=0).contiguous()


def crop(img, crop_size=None):
    """Centered crop: (..., H, W) -> (..., crop_size[0], crop_size[1])."""
    if crop_size is None:
        crop_size = ((img.shape[-2] + 1) // 2, (img.shape[-1] + 1) // 2)

    top = (img.shape[-2] - crop_size[-2]) // 2
    left = (img.shape[-1] - crop_size[-1]) // 2

    return img[..., top:top + crop_size[-2], left:left + crop_size[-1]].contiguous()
