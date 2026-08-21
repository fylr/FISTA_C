"""FISTA_C+U: FISTA_C preliminary reconstruction + U-Net perceptual enhancement (proposed).

This is the proposed method in the paper (Section 2.3, Tables 1-2).  The
pipeline consists of two stages:

1. **FISTA_C** (``algorithms.reconstruction.LenslessReconstructor.fista_c``):
   model-based iterative reconstruction using the optimized imaging model
   with TV regularization (Eq. 6).  This produces a stable but potentially
   noisy preliminary reconstruction.

2. **U-Net** (``net.unet.UNet``):  a lightweight U-Net that refines the
   FISTA_C output to improve perceptual quality (Eq. 7 loss).

At inference time the two stages are applied sequentially.  During training,
the FISTA_C outputs are typically pre-computed offline and the U-Net is
trained to map them to the ground-truth scenes (training code is not
included in this repository).
"""

import torch
import torch.nn as nn

from algorithms.reconstruction import LenslessReconstructor
from net.unet import UNet


class FISTA_C_U(nn.Module):
    """FISTA_C + U-Net perceptual enhancement pipeline.

    Args:
        psf_path:    path to the PSF image.
        obj_size:    (H, W) of the effective scene region.
        blur_size:   (H, W) of the cropped blur image.
        conv_size:   (H, W) of the padded convolution domain.
        sensor_size: (H, W) of the full sensor image (defaults to conv_size).
        weight:      PSF brightness scaling factor.
        fista_lamda: TV regularization weight for FISTA_C.
        fista_iters: maximum FISTA_C iterations.
        in_chans:    number of input/output channels.
        nn_upsample: use bilinear upsampling in U-Net decoder.
    """

    def __init__(self, psf_path: str, obj_size: tuple = (96, 96),
                 blur_size: tuple = (112, 112), conv_size: tuple = (480, 480),
                 sensor_size: tuple = None, weight: float = 1.0,
                 fista_lamda: float = 1.0e-4, fista_iters: int = 1500,
                 in_chans: int = 3, nn_upsample: bool = False):
        super().__init__()
        self.obj_size = tuple(obj_size)
        self.blur_size = tuple(blur_size)
        self.fista_lamda = fista_lamda
        self.fista_iters = fista_iters

        # When the input is a pre-cropped blur_crop (the typical case),
        # sensor_size must equal blur_size so that set_measurement() correctly
        # round-trips the crop through pad -> crop.
        if sensor_size is None:
            sensor_size = self.blur_size

        # Stage 1: FISTA_C reconstructor (model-based, non-trainable).
        self.reconstructor = LenslessReconstructor(
            psf_path=psf_path, obj_size=obj_size, blur_size=blur_size,
            conv_size=conv_size, sensor_size=sensor_size, weight=weight,
        )

        # Stage 2: U-Net perceptual enhancement (trainable).
        self.unet = UNet(in_chans=in_chans, nn_upsample=nn_upsample,
                         negative_slope=0.01, obj_size=self.obj_size)

    def fista_c_reconstruct(self, blur: torch.Tensor, blur_tl: tuple = None):
        """Run FISTA_C on a blur_crop image and return the preliminary result.

        Args:
            blur:    cropped blur image tensor (B, C, H, W) in [0, 1].
            blur_tl: top-left corner of the blur crop (None = center).

        Returns:
            rec: FISTA_C reconstruction (B, C, obj_H, obj_W) in [0, 1].
        """
        self.reconstructor.set_measurement(blur, blur_tl=blur_tl)
        rec, _ = self.reconstructor.fista_c(
            lamda=self.fista_lamda, iters=self.fista_iters,
        )
        return rec

    def forward(self, blur: torch.Tensor, blur_tl: tuple = None) -> torch.Tensor:
        """Full pipeline: FISTA_C -> U-Net.

        Args:
            blur:    cropped blur image (B, C, H, W) in [0, 1].
            blur_tl: top-left corner of the blur crop (None = center).

        Returns:
            Enhanced reconstruction (B, C, obj_H, obj_W) in [0, 1].
        """
        with torch.no_grad():
            rec_fista = self.fista_c_reconstruct(blur, blur_tl=blur_tl)
        return self.unet(rec_fista)

    def forward_from_fista(self, rec_fista: torch.Tensor) -> torch.Tensor:
        """Bypass FISTA_C and run only U-Net on a pre-computed reconstruction.

        Useful when FISTA_C outputs have been pre-computed and cached.

        Args:
            rec_fista: pre-computed FISTA_C reconstruction (B, C, H, W).

        Returns:
            Enhanced reconstruction (B, C, obj_H, obj_W) in [0, 1].
        """
        return self.unet(rec_fista)
