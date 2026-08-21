"""Model-based reconstruction algorithms for miniaturized lensless cameras.

Implements the three traditional reconstruction algorithms of the paper
"High-quality reconstruction method for miniaturized lensless cameras based
on imaging model optimization":

1. Wiener (Eq. (4)): the analytical solution corresponding to the optimized
   imaging model.
2. FISTA_O (Eq. (5)): FISTA for the original imaging model
       argmin  1/2 ||C M v_pad - b_crop||^2 + lamda ||v_pad||_1
   with sparse (soft-thresholding) regularization.
3. FISTA_C (Eq. (6), proposed): FISTA for the optimized imaging model
       argmin  1/2 ||C1 M C2^H v - b_crop||^2 + lamda (||Dx v||_1 + ||Dy v||_1)
   with gradient-sparse (TV) regularization on the effective scene region.

Both FISTA variants follow Algorithm 1 of the paper:
   - backtracking line search for an adaptive step size mu;
   - an approximate TV proximal mapping via gradient-domain soft
     thresholding (fast alternative to Chambolle's projection);
   - all convolutions computed in the frequency domain (see Supplement 1).

Terminology / sizes:
   - obj_size   : size of the effective scene region "view" (e.g. 96x96);
   - blur_size  : size of the cropped encoded image "blur_crop" (e.g. 112x112);
   - conv_size  : size of the padded domain for circular convolution
                  (e.g. 480x480);
   - sensor_size: size of the recorded blur image (e.g. 480x480).
"""

import numpy as np
import torch
import torch.nn.functional as nf
import torchvision.transforms.v2.functional as v2f
from torch import nn

from algorithms.fourier import toFreq, fromFreq, pad
from utils.image_loader import opencv_loader
from torchvision.transforms import v2


def SoftThresh(x, tau):
    """Soft-thresholding operator, the proximal mapping of the l1 norm."""
    return torch.sign(x) * torch.maximum(torch.abs(x) - tau, torch.zeros_like(x))


def anisotropic_tv_fast_approx_map(x, tau):
    """Fast approximate proximal mapping of the anisotropic TV regularizer.

    Instead of Chambolle's projection (iterative and slow), soft thresholding
    is applied independently in the gradient domain, and the divergence
    operator approximately reconstructs the image from the processed
    gradients (Section 2.2 of the paper).
    """
    Dh = torch.roll(x, 1, dims=-2) - x
    Dh_st = SoftThresh(Dh, tau)
    Dh_st_DhH = torch.roll(Dh_st, -1, dims=-2) - Dh_st
    Dw = torch.roll(x, 1, dims=-1) - x
    Dw_st = SoftThresh(Dw, tau)
    Dw_st_DwH = torch.roll(Dw_st, -1, dims=-1) - Dw_st

    return x - tau * (Dh_st_DhH + Dw_st_DwH)


def _center_lrtb(full_size, size):
    """Centered pad / crop amounts expressed as nf.pad (left, right, top, bottom).

    The negative version implements the adjoint operation: pad <-> crop.
    """
    tl = ((full_size[0] - size[0]) // 2, (full_size[1] - size[1]) // 2)
    pad_lrtb = (tl[1], full_size[1] - tl[1] - size[1],
                tl[0], full_size[0] - tl[0] - size[0])
    crop_lrtb = (-pad_lrtb[0], -pad_lrtb[1], -pad_lrtb[2], -pad_lrtb[3])
    return tl, pad_lrtb, crop_lrtb


class LenslessReconstructor(nn.Module):
    """Reconstruction module for a fixed PSF and geometry.

    Args:
        psf_path:    path of the PSF image (png/tif, grayscale or color).
        obj_size:    (H, W) of the effective scene region "view".
        blur_size:   (H, W) of the cropped encoded image "blur_crop".
        conv_size:   (H, W) of the padded convolution domain.
        sensor_size: (H, W) of the recorded blur image; defaults to conv_size.
        weight:      brightness scaling of the PSF (camera/dataset dependent;
                     larger -> darker reconstruction).
        dtype:       torch.float32 or torch.float64.
        is_quant:    whether to simulate quantization (for simulation only).
    """

    def __init__(self, psf_path, obj_size, blur_size, conv_size, sensor_size=None,
                 weight=1.0, dtype=torch.float32, norm_type='backward', is_quant=False):
        super().__init__()
        self.obj_size = tuple(obj_size)
        self.blur_size = tuple(blur_size)
        self.conv_size = tuple(conv_size)
        self.sensor_size = tuple(sensor_size) if sensor_size is not None else self.conv_size
        self.weight = weight
        self.dtype = dtype
        self.norm_type = norm_type
        self.is_quant = is_quant
        self.eps = 1e-8

        # Cropping / padding operators for the scene (C2) and the sensor (C1).
        _, self.obj_pad_lrtb, self.obj_crop_lrtb = _center_lrtb(self.conv_size, self.obj_size)
        _, self.sensor_pad_lrtb, self.sensor_crop_lrtb = _center_lrtb(self.conv_size, self.sensor_size)

        # PSF spectrum (energy-normalized and scaled).
        self.psf_freq = nn.Buffer(self._load_psf_freq(psf_path))

        self.blur_tl = None
        self.blur_crop_lrtb = None

    # ------------------------------------------------------------------ #
    # Operators
    # ------------------------------------------------------------------ #
    def obj_pad_func(self, img):
        """C2^H: zero-pad the effective scene region to conv_size."""
        return nf.pad(img, pad=self.obj_pad_lrtb, mode='constant', value=0).contiguous()

    def obj_crop_func(self, img):
        """C2: crop the effective scene region from conv_size."""
        return nf.pad(img, pad=self.obj_crop_lrtb, mode='constant', value=0).contiguous()

    def sensor_pad_func(self, img):
        """C1^H on the sensor domain (pads, or crops if sensor > conv_size)."""
        return nf.pad(img, pad=self.sensor_pad_lrtb, mode='constant', value=0).contiguous()

    def blur_pad_func(self, img):
        """C1^H: zero-pad blur_crop back to conv_size at the cropping position."""
        return nf.pad(img, pad=self.blur_pad_lrtb, mode='constant', value=0).contiguous()

    def blur_crop_func(self, img):
        """C1: crop blur_crop out of the full encoded image at blur_tl."""
        return nf.pad(img, pad=self.blur_crop_lrtb, mode='constant', value=0).contiguous()

    def set_blur_tl(self, blur_tl=None):
        """Set the top-left corner of the blur cropping region (None = center)."""
        if blur_tl is None:
            blur_tl = ((self.conv_size[0] - self.blur_size[0]) // 2,
                       (self.conv_size[1] - self.blur_size[1]) // 2)
        self.blur_tl = tuple(blur_tl)
        self.blur_pad_lrtb = (self.blur_tl[1], self.conv_size[1] - self.blur_tl[1] - self.blur_size[1],
                              self.blur_tl[0], self.conv_size[0] - self.blur_tl[0] - self.blur_size[0])
        self.blur_crop_lrtb = (-self.blur_pad_lrtb[0], -self.blur_pad_lrtb[1],
                               -self.blur_pad_lrtb[2], -self.blur_pad_lrtb[3])

    # Forward / adjoint operators of the two imaging models.
    def A_C1M(self, vk):
        """Original model, forward: A v_pad = C1 M v_pad."""
        vk_freq = toFreq(vk, self.norm_type)
        blur_rec = fromFreq(self.psf_freq * vk_freq, self.norm_type).abs()
        return self.blur_crop_func(blur_rec)

    def AH_MHC1H(self, blur):
        """Original model, adjoint: A^H u = M^H C1^H u."""
        blur_pad = self.blur_pad_func(blur)
        blur_pad_freq = toFreq(blur_pad, self.norm_type)
        return fromFreq(self.psf_freq.conj() * blur_pad_freq, self.norm_type).real

    def A_C1MC2H(self, vk):
        """Optimized model, forward: A v = C1 M C2^H v."""
        vk_pad = self.obj_pad_func(vk)
        vk_pad_freq = toFreq(vk_pad, self.norm_type)
        blur_rec = fromFreq(self.psf_freq * vk_pad_freq, self.norm_type).abs()
        return self.blur_crop_func(blur_rec)

    def AH_C2MHC1H(self, blur):
        """Optimized model, adjoint: A^H u = C2 M^H C1^H u."""
        blur_pad = self.blur_pad_func(blur)
        blur_pad_freq = toFreq(blur_pad, self.norm_type)
        rec_pad = fromFreq(self.psf_freq.conj() * blur_pad_freq, self.norm_type).real
        return self.obj_crop_func(rec_pad)

    # ------------------------------------------------------------------ #
    # Measurements: simulation or real data
    # ------------------------------------------------------------------ #
    def simulate_measurement(self, obj, scale_factor=15000, noise_std=0.01, blur_tl=None):
        """Simulate blur_crop from a scene image (Fig. 2, lower pipeline).

        Args:
            obj:          scene image tensor (..., C, H, W) in [0, 1] or uint8.
            scale_factor: full-well capacity controlling shot noise strength.
            noise_std:    std of the Gaussian read noise.
            blur_tl:      top-left corner of the blur crop (None = center).

        Returns:
            blur_crop, blur (full encoded image before cropping).
        """
        self.set_blur_tl(blur_tl)

        obj = v2f.resize(obj.to(self.dtype), size=list(self.obj_size))
        self.obj = obj
        obj_pad_freq = toFreq(self.obj_pad_func(obj), self.norm_type)
        blur = fromFreq(obj_pad_freq * self.psf_freq, self.norm_type).abs().clip(min=0, max=1)
        self.blur = blur

        blur_crop = self.blur_crop_func(blur)
        # Shot noise (Poisson) + read noise (Gaussian).
        scale_factor = scale_factor * 0.8
        if scale_factor > 0:
            blur_crop = torch.poisson(blur_crop * scale_factor) / scale_factor
        blur_crop = blur_crop + torch.normal(mean=0., std=noise_std, size=blur_crop.shape,
                                             dtype=self.dtype, device=blur_crop.device)
        blur_crop = blur_crop.clip(min=0, max=1)
        if self.is_quant:  # quantization (12-bit as in the paper's simulation)
            blur_crop = (blur_crop * 4095).round() / 4095
        self.blur_crop = blur_crop

        return blur_crop, blur

    def set_measurement(self, blur_sensor, obj_sensor=None, blur_tl=None):
        """Use a real recorded blur image (and its ground truth, if given).

        Args:
            blur_sensor: recorded blur image (..., C, H, W) of sensor_size.
            obj_sensor:  ground-truth scene image of sensor_size (optional,
                         used only for evaluation).
            blur_tl:     top-left corner of the blur crop (None = center).
        """
        self.set_blur_tl(blur_tl)

        blur_sensor = blur_sensor.to(self.dtype)
        self.blur = self.sensor_pad_func(blur_sensor)
        self.blur_crop = self.blur_crop_func(self.blur)

        if obj_sensor is not None:
            obj_sensor = obj_sensor.to(self.dtype)
            self.obj_pad = self.sensor_pad_func(obj_sensor)
            self.obj = self.obj_crop_func(self.obj_pad)

    # ------------------------------------------------------------------ #
    # 1. Wiener (Eq. (4))
    # ------------------------------------------------------------------ #
    def wiener(self):
        """Wiener solution of the optimized imaging model.

        v_hat = crop{ F^-1[ conj(F(k)) * F[pad(b_crop)] / (|F(k)|^2 + lamda2) ] }
        """
        blur_crop_pad = self.blur_pad_func(self.blur_crop)
        blur_freq_rec = toFreq(blur_crop_pad, self.norm_type)

        # Regularization term lamda2 (Sec. 3.2: derived from the PSF spectrum).
        nsr_k = self.psf_freq.abs().mean() * self.psf_freq.abs().median()
        wiener_h = self.psf_freq.conj() / (self.psf_freq.abs().pow(2) + nsr_k)

        rec_pad = fromFreq(blur_freq_rec * wiener_h, self.norm_type).real
        self.rec_pad = rec_pad.clip(min=0, max=1)
        self.rec = self.obj_crop_func(self.rec_pad)
        return self.rec, self.rec_pad

    # ------------------------------------------------------------------ #
    # 2/3. FISTA with backtracking line search (Algorithm 1)
    # ------------------------------------------------------------------ #
    def _fista(self, A_func, AH_func, v0, lamda=1.0e-4, iters=1500, is_tv=True,
               is_bt=True, mu_eta=0.75, err_threshold=1.0e-8, disp_interval=0):
        """Generic FISTA loop shared by FISTA_O and FISTA_C.

        Args:
            A_func / AH_func: forward / adjoint operators of the model.
            v0:               initialization (C1^H b for FISTA_O,
                              C2 C1^H b for FISTA_C, see Algorithm 1 line 1).
            lamda:            regularization weight.
            iters:            maximum iterations (1500 in the paper).
            is_tv:            True -> TV regularization (FISTA_C),
                              False -> sparse regularization (FISTA_O).
            is_bt:            enable backtracking line search.
            mu_eta:           step size decay factor eta.
            err_threshold:    convergence threshold on the data fidelity.
        """
        approx_map_func = anisotropic_tv_fast_approx_map if is_tv else SoftThresh

        # Initial step size: mu = max(1 / ||F(k)||_inf^2, 1.0) (Algorithm 1).
        mu = 1.0 / self.psf_freq.abs().pow(2).max().item()
        if is_bt:
            mu = max(mu, 1.0)
        tau = mu * lamda

        vk = v0
        vk_prev = vk
        tk = 1.0
        fidelity = None

        for it in range(iters):
            # Convergence criterion (Algorithm 1 line 3).
            if fidelity is not None and fidelity.item() < err_threshold:
                print(f"iter {it:4d}: data fidelity {fidelity.item():.4e} < "
                      f"{err_threshold:.4e}, converged.")
                break

            if disp_interval > 0 and it % disp_interval == 0:
                print(f"iter {it:4d}, data fidelity: "
                      f"{0.5 * (A_func(vk) - self.blur_crop).pow(2).sum().item():.4e}")

            # Momentum update.
            tk_prev = tk
            tk = (1. + np.sqrt(1. + 4. * tk_prev ** 2)) / 2.
            # Extrapolation.
            yk = vk + (tk_prev - 1.) / tk * (vk - vk_prev)
            # Gradient of the data fidelity term: A^H (A y_k - b).
            blur_crop_diff = A_func(yk) - self.blur_crop
            gradient = AH_func(blur_crop_diff)

            vk_prev = vk
            if is_bt:  # Backtracking line search over the step size mu.
                while True:
                    zk = yk - mu * gradient
                    vk_new = approx_map_func(zk, tau=tau).clip(min=0, max=1)
                    fvk_new = (A_func(vk_new) - self.blur_crop).pow(2).sum() / 2.
                    fyk1 = blur_crop_diff.pow(2).sum() / 2.
                    fyk2 = (gradient * (vk_new - yk)).sum()
                    fyk3 = (vk_new - yk).pow(2).sum() / (2 * mu)
                    if fvk_new <= fyk1 + fyk2 + fyk3:
                        vk = vk_new
                        fidelity = fvk_new
                        break
                    else:
                        mu = mu * mu_eta
                        tau = mu * lamda
            else:
                zk = yk - mu * gradient
                vk = approx_map_func(zk, tau=tau).clip(min=0, max=1)
                fidelity = (A_func(vk) - self.blur_crop).pow(2).sum() / 2.

        return vk

    def fista_o(self, lamda=1.0e-2, iters=1500, is_bt=True, mu_eta=0.75,
                err_threshold=1.0e-8, disp_interval=0):
        """FISTA_O: original imaging model with sparse regularization (Eq. (5)).

        The unknown is the padded scene v_pad (conv_size); initialization
        v0 = C1^H b_crop as in Algorithm 1.
        """
        v0 = self.blur_pad_func(self.blur_crop)
        rec_pad = self._fista(self.A_C1M, self.AH_MHC1H, v0, lamda=lamda, iters=iters,
                              is_tv=False, is_bt=is_bt, mu_eta=mu_eta,
                              err_threshold=err_threshold, disp_interval=disp_interval)
        self.rec_pad = rec_pad.clip(min=0, max=1)
        self.rec = self.obj_crop_func(self.rec_pad)
        return self.rec, self.rec_pad

    def fista_c(self, lamda=1.0e-4, iters=1500, is_bt=True, mu_eta=0.75,
                err_threshold=1.0e-8, disp_interval=0):
        """FISTA_C (proposed): optimized imaging model with TV regularization
        (Eq. (6)).

        The unknown is the effective scene region v (obj_size); initialization
        v0 = C2 C1^H b_crop as in Algorithm 1.
        """
        v0 = self.obj_crop_func(self.blur_pad_func(self.blur_crop))
        rec = self._fista(self.A_C1MC2H, self.AH_C2MHC1H, v0, lamda=lamda, iters=iters,
                          is_tv=True, is_bt=is_bt, mu_eta=mu_eta,
                          err_threshold=err_threshold, disp_interval=disp_interval)
        self.rec = rec.clip(min=0, max=1)
        self.rec_pad = self.obj_pad_func(self.rec)
        return self.rec, self.rec_pad

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _load_psf_freq(self, psf_path):
        psf_img = opencv_loader(psf_path)
        my_trans = v2.Compose([v2.ToImage(), v2.ToDtype(self.dtype, scale=True)])
        psf = my_trans(psf_img).unsqueeze(0)

        self.psf = psf / psf.max()  # for display only
        # Energy normalization (then scaled by a camera-dependent weight).
        psf = psf / psf.sum() * psf.shape[-3] * self.weight
        psf_pad = pad(psf, pad_size=(self.conv_size[0] - psf.shape[-2],
                                     self.conv_size[1] - psf.shape[-1]))

        return toFreq(psf_pad, self.norm_type)
