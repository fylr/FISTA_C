"""FlatNet-gen-C: Wiener inversion followed by U-Net refinement (Khan et al., 2022).

This is one of the comparison methods in the paper (Section 3.2, Table 1).
The network first applies a trainable Wiener filter in the frequency domain to
obtain a preliminary reconstruction, then refines it with a U-Net decoder.

The Wiener filter parameters (PSF, noise-to-signal ratio, brightness weight)
are registered as learnable parameters and are optimized during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as nf
from torchvision.transforms import v2

from algorithms.fourier import toFreq, fromFreq, pad
from net.unet import UNet
from utils.image_loader import opencv_loader


class FlatNetGenC(nn.Module):
    """FlatNet-gen-C: Wiener inversion + U-Net.

    Args:
        psf_path:   path to the PSF image.
        in_chans:   number of input/output channels (3 for RGB).
        obj_size:   (H, W) of the output effective scene region.
        blur_size:  (H, W) of the input cropped blur image.
        conv_size:  (H, W) of the padded convolution domain.
    """

    def __init__(self, psf_path: str, in_chans: int = 3,
                 obj_size: tuple = (96, 96), blur_size: tuple = (112, 112),
                 conv_size: tuple = (480, 480)):
        super().__init__()
        self.obj_size = tuple(obj_size)
        self.blur_size = tuple(blur_size)
        self.conv_size = tuple(conv_size)

        # Centered pad / crop operators for the scene and sensor domains.
        self._setup_operators()

        # Load PSF and compute initial Wiener filter parameters.
        psf_pad, nsr_k = self._init_wiener(psf_path)
        self.psf_pad = nn.Parameter(psf_pad, requires_grad=True)
        self.nsr_k = nn.Parameter(nsr_k, requires_grad=True)
        self.brightness = nn.Parameter(torch.tensor([1000.0]), requires_grad=True)

        # U-Net refinement network.
        self.unet = UNet(in_chans=in_chans, nn_upsample=False,
                         negative_slope=0.01, obj_size=self.obj_size)

    def _setup_operators(self):
        """Compute centered pad/crop LRTB tuples for scene and blur."""
        obj_tl = ((self.conv_size[0] - self.obj_size[0]) // 2,
                  (self.conv_size[1] - self.obj_size[1]) // 2)
        self.obj_pad_lrtb = (obj_tl[1], self.conv_size[1] - obj_tl[1] - self.obj_size[1],
                             obj_tl[0], self.conv_size[0] - obj_tl[0] - self.obj_size[0])
        self.obj_crop_lrtb = tuple(-v for v in self.obj_pad_lrtb)

        blur_tl = ((self.conv_size[0] - self.blur_size[0]) // 2,
                   (self.conv_size[1] - self.blur_size[1]) // 2)
        self.blur_pad_lrtb = (blur_tl[1], self.conv_size[1] - blur_tl[1] - self.blur_size[1],
                              blur_tl[0], self.conv_size[0] - blur_tl[0] - self.blur_size[0])

    def _init_wiener(self, psf_path: str):
        """Load PSF, pad to conv_size, and compute the initial NSR."""
        psf_img = opencv_loader(psf_path)
        my_trans = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        psf = my_trans(psf_img).unsqueeze(0)
        psf_pad = pad(psf, pad_size=(self.conv_size[0] - psf.shape[-2],
                                     self.conv_size[1] - psf.shape[-1]))
        psf_freq = toFreq(psf_pad)
        nsr_k = psf_freq.abs().mean() * psf_freq.abs().median()
        return psf_pad, nsr_k

    def _blur_pad(self, img):
        return nf.pad(img, pad=self.blur_pad_lrtb, mode="replicate", value=0)

    def _obj_crop(self, img):
        return nf.pad(img, pad=self.obj_crop_lrtb, mode="replicate", value=0)

    def forward(self, blur: torch.Tensor) -> torch.Tensor:
        """Wiener inversion -> brightness scaling -> U-Net refinement."""
        # Frequency-domain Wiener deconvolution.
        psf_freq = toFreq(self.psf_pad)
        wiener_h = psf_freq.conj() / (psf_freq.abs().pow(2) + self.nsr_k)
        blur_pad = self._blur_pad(blur)
        blur_freq = toFreq(blur_pad)
        rec_pad = fromFreq(blur_freq * wiener_h).real
        rec = self._obj_crop(rec_pad) * self.brightness

        # U-Net perceptual refinement.
        return self.unet(rec)
