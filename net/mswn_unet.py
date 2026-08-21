"""MWDN-CPSF: Multi-Scale Wiener Deconvolution Network with Coded PSF (Li et al., 2023).

This is one of the comparison methods in the paper (Section 3.2, Table 1).
The network embeds trainable Wiener deconvolution modules at multiple scales
of the U-Net encoder, using a separate PSF encoder that downsamples the PSF
in parallel with the main encoder.  Each skip-connection feature map is
Wiener-deconvolved with the corresponding downsampled PSF before being passed
to the decoder.
"""

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as v2f
from torchvision.transforms import v2

from algorithms.fourier import toFreq, fromFreq, pad, crop
from net.unet import DoubleConv, Down, Up
from utils.image_loader import opencv_loader


class MWDN_CPSF(nn.Module):
    """Multi-Scale Wiener Deconvolution Network (MWDN-CPSF).

    Args:
        psf_path:   path to the PSF image.
        in_chans:   number of input/output channels (3 for RGB).
        nn_upsample: use bilinear upsampling in the decoder.
        obj_size:   (H, W) of the output effective scene region.
        conv_size:  (H, W) of the padded convolution domain.
    """

    def __init__(self, psf_path: str, in_chans: int = 3,
                 nn_upsample: bool = False, obj_size: tuple = (96, 96),
                 conv_size: tuple = (480, 480)):
        super().__init__()
        self.obj_size = tuple(obj_size)
        self.conv_size = tuple(conv_size)

        # Load and pad PSF.
        psf = self._load_psf(psf_path)
        self.psf_img = nn.Buffer(psf)

        # Learnable intensity weight and per-scale SNR inverse parameters.
        self.intensity_weight = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.snr_inv = nn.Parameter(torch.tensor([0.01] * 4), requires_grad=True)

        # Main encoder.
        self.in_conv = DoubleConv(in_chans, 24)
        self.down1 = Down(24, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        # PSF encoder (parallel downsampling path).
        self.in_conv_psf = DoubleConv(in_chans, 24)
        self.down_psf1 = Down(24, 64)
        self.down_psf2 = Down(64, 128)
        self.down_psf3 = Down(128, 256)

        # Decoder.
        self.up4 = Up(512, 256, nn_upsample=nn_upsample)
        self.up3 = Up(256, 128, nn_upsample=nn_upsample)
        self.up2 = Up(128, 64, nn_upsample=nn_upsample)
        self.up1 = Up(64, 24, nn_upsample=nn_upsample)
        self.out_conv = nn.Conv2d(24, in_chans, kernel_size=1)

    def _load_psf(self, psf_path: str) -> torch.Tensor:
        psf_img = opencv_loader(psf_path)
        my_trans = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        psf = my_trans(psf_img).unsqueeze(0)
        return pad(psf, pad_size=(self.conv_size[0] - psf.shape[-2],
                                  self.conv_size[1] - psf.shape[-1]))

    @staticmethod
    def wiener_deconv(blur_img: torch.Tensor, psf_img: torch.Tensor,
                      snr_inv: torch.Tensor) -> torch.Tensor:
        """Wiener deconvolution in the frequency domain.

        Args:
            blur_img: input feature map (B, C, H, W).
            psf_img:  PSF at the same scale (B, C, H_psf, W_psf).
            snr_inv:  inverse SNR (regularization strength).
        """
        psf_freq = toFreq(psf_img)
        # Pad blur to PSF size for circular convolution.
        blur_pad = pad(blur_img, pad_size=(psf_img.shape[-2] - blur_img.shape[-2],
                                           psf_img.shape[-1] - blur_img.shape[-1]))
        blur_freq = toFreq(blur_pad)
        rec_freq = blur_freq * psf_freq.conj() / (psf_freq.abs().pow(2) + snr_inv)
        rec_pad = fromFreq(rec_freq).real
        return crop(rec_pad, crop_size=(blur_img.shape[-2], blur_img.shape[-1]))

    def weight_constraint(self):
        """Clamp learnable parameters to positive values (call after optimizer step)."""
        self.intensity_weight.data.clamp_(min=1e-8)
        self.snr_inv.data.clamp_(min=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main encoder.
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # PSF encoder (parallel path).
        psf1 = self.in_conv_psf(self.psf_img * self.intensity_weight)
        psf2 = self.down_psf1(psf1)
        psf3 = self.down_psf2(psf2)
        psf4 = self.down_psf3(psf3)

        # Multi-scale Wiener deconvolution on skip connections.
        x4 = self.wiener_deconv(x4, psf4, self.snr_inv[3])
        x3 = self.wiener_deconv(x3, psf3, self.snr_inv[2])
        x2 = self.wiener_deconv(x2, psf2, self.snr_inv[1])
        x1 = self.wiener_deconv(x1, psf1, self.snr_inv[0])

        # Decoder with skip connections.
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.out_conv(x)

        # Center crop to target view size and clip to [0, 1].
        return v2f.center_crop(x, self.obj_size).clip(min=0, max=1)
