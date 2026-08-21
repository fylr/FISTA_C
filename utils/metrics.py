"""Image quality metrics: PSNR, SSIM, and a lightweight LPIPS wrapper.

These metrics are used to evaluate reconstruction quality in the paper
(Section 3, Tables 1-2).  LPIPS requires the ``lpips`` package; if it is
not installed the corresponding function raises a clear error.
"""

import torch
import torch.nn.functional as nf


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio.

    Args:
        pred:       predicted image tensor, same shape as ``target``.
        target:     ground-truth image tensor.
        data_range: dynamic range of the images (1.0 for [0, 1], 255 for [0, 255]).
    """
    mse = nf.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 10.0 * torch.log10((data_range ** 2) / mse)


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device, dtype):
    """Create a 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_1d = g.unsqueeze(1)
    kernel_2d = kernel_1d @ kernel_1d.t()
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11,
         data_range: float = 1.0, size_average: bool = True) -> torch.Tensor:
    """Structural Similarity Index (Wang et al., 2004).

    Args:
        pred:         predicted image (B, C, H, W) or (C, H, W).
        target:       ground-truth image, same shape.
        window_size:  size of the Gaussian window (odd number).
        data_range:   dynamic range.
        size_average: if True, return mean SSIM over batch and channels.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    C = pred.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5, C, pred.device, pred.dtype)

    mu1 = nf.conv2d(pred, kernel, groups=C, padding=window_size // 2)
    mu2 = nf.conv2d(target, kernel, groups=C, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nf.conv2d(pred ** 2, kernel, groups=C, padding=window_size // 2) - mu1_sq
    sigma2_sq = nf.conv2d(target ** 2, kernel, groups=C, padding=window_size // 2) - mu2_sq
    sigma12 = nf.conv2d(pred * target, kernel, groups=C, padding=window_size // 2) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=(1, 2, 3))


class LPIPSMetric:
    """Learned Perceptual Image Patch Similarity (Zhang et al., 2018).

    Uses AlexNet as the backbone network, as in the paper (Eq. 7).

    Usage::

        lpips_fn = LPIPSMetric()
        score = lpips_fn(pred, target)
    """

    def __init__(self, net: str = "alex", verbose: bool = False):
        try:
            import lpips
        except ImportError:
            raise ImportError(
                "LPIPS requires the 'lpips' package. Install it with: pip install lpips"
            )
        self._model = lpips.LPIPS(net=net, verbose=verbose)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

    def to(self, device):
        self._model = self._model.to(device)
        return self

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS.  Inputs should be in [0, 1]; they are rescaled to [-1, 1]."""
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        pred = pred * 2.0 - 1.0
        target = target * 2.0 - 1.0
        return self._model(pred, target).mean()
