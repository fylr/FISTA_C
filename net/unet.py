"""U-Net architecture for lensless image reconstruction (Table 1 of the paper).

This U-Net is used as:
  * the perceptual enhancement network in FISTA_C+U (proposed, Section 2.3);
  * the pure deep-learning baseline U-Net (comparison, Section 3.2);
  * the decoder backbone in FlatNet-gen-C and MWDN-CPSF (comparisons).

The network takes a blur_crop image as input and outputs the reconstructed
effective scene region (view).  A center crop is applied at the output so
that the result matches the target view size exactly.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as nf
import torchvision.transforms.v2.functional as v2f


class DoubleConv(nn.Module):
    """(Conv2d -> Norm -> LeakyReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int,
                 mid_channels: Optional[int] = None,
                 negative_slope: float = 0.01,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            relu,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            relu,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool downsampling followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int,
                 negative_slope: float = 0.01,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        self.down = nn.MaxPool2d(2, 2)
        self.conv = DoubleConv(in_channels, out_channels,
                               negative_slope=negative_slope, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.down(x))


class Up(nn.Module):
    """Transposed-conv upsampling, skip concatenation, then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int,
                 skip_channels: Optional[int] = None,
                 nn_upsample: bool = False,
                 negative_slope: float = 0.01,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if skip_channels is None:
            skip_channels = out_channels

        if nn_upsample:
            self.up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, skip_channels, kernel_size=1),
            )
        else:
            self.up_conv = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(out_channels + skip_channels, out_channels,
                               negative_slope=negative_slope, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        # Handle size mismatch from odd input dimensions.
        if x.shape[-2:] != x_skip.shape[-2:]:
            diffH = x_skip.shape[-2] - x.shape[-2]
            diffW = x_skip.shape[-1] - x.shape[-1]
            x = nf.pad(x, [diffW // 2, diffW - diffW // 2,
                           diffH // 2, diffH - diffH // 2])
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for lensless reconstruction (Table 1).

    Args:
        in_chans:     number of input channels (3 for RGB).
        nn_upsample:  if True, use bilinear upsampling + 1x1 conv instead of
                      transposed convolution (useful for odd input sizes).
        negative_slope: LeakyReLU negative slope.
        norm_layer:   normalization layer class (default BatchNorm2d).
        obj_size:     (H, W) of the output effective scene region.  A center
                      crop of this size is applied to the network output.
    """

    def __init__(self, in_chans: int = 3, nn_upsample: bool = False,
                 negative_slope: float = 0.01,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 obj_size: tuple = (96, 96)):
        super().__init__()
        self.obj_size = tuple(obj_size)

        self.in_conv = DoubleConv(in_chans, 24, negative_slope, norm_layer)
        self.down1 = Down(24, 64, negative_slope, norm_layer)
        self.down2 = Down(64, 128, negative_slope, norm_layer)
        self.down3 = Down(128, 256, negative_slope, norm_layer)
        self.down4 = Down(256, 512, negative_slope, norm_layer)

        self.up4 = Up(512, 256, nn_upsample=nn_upsample, negative_slope=negative_slope, norm_layer=norm_layer)
        self.up3 = Up(256, 128, nn_upsample=nn_upsample, negative_slope=negative_slope, norm_layer=norm_layer)
        self.up2 = Up(128, 64, nn_upsample=nn_upsample, negative_slope=negative_slope, norm_layer=norm_layer)
        self.up1 = Up(64, 24, nn_upsample=nn_upsample, negative_slope=negative_slope, norm_layer=norm_layer)

        self.out_conv = nn.Conv2d(24, in_chans, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.out_conv(x)

        # Center crop to the target view size and clip to [0, 1].
        x = v2f.center_crop(x, self.obj_size).clip(min=0, max=1)
        return x
