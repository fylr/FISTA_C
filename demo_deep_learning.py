"""Demo: deep learning reconstruction algorithms (U-Net, FlatNet-gen-C, MWDN-CPSF, FISTA_C+U).

This script demonstrates the four deep-learning-based methods compared in
the paper (Section 3.2, Tables 1-2):

  * U-Net          -- pure end-to-end deep learning baseline
  * FlatNet-gen-C  -- Wiener inversion + U-Net (Khan et al., 2022)
  * MWDN-CPSF      -- multi-scale Wiener deconvolution + U-Net (Li et al., 2023)
  * FISTA_C+U      -- FISTA_C preliminary reconstruction + U-Net (proposed)

Pretrained model weights should be downloaded and placed under ``checkpoints/``.
If weights are not available, the networks run with random initialization to
verify the forward pass.

Usage:
    # PhlatCam sample (PSF and data included in the repo)
    python demo_deep_learning.py \\
        --psf samples/psf/PhlatCam_psf.png \\
        --blur samples/OPhlatCam/blur/n01440764/n01440764_1154.png \\
        --gt samples/OPhlatCam/gt/n01440764/n01440764_1154.png \\
        --obj_size 192 192 --blur_size 224 224 --conv_size 640 704 \\
        --methods unet flatnet mswn fista_c_u

    # PhlatCam sample with pretrained weights
    python demo_deep_learning.py \\
        --psf samples/psf/PhlatCam_psf.png \\
        --blur samples/OPhlatCam/blur/n01440764/n01440764_1154.png \\
        --gt samples/OPhlatCam/gt/n01440764/n01440764_1154.png \\
        --obj_size 192 192 --blur_size 224 224 --conv_size 640 704 \\
        --methods unet flatnet mswn fista_c_u \\
        --checkpoints unet=checkpoints/OPhlatCam-unet-rec-blur2gt_old.pth flatnet=checkpoints/OPhlatCam-flatnet-rec-blur2gt_old.pth mswn=checkpoints/OPhlatCam-mswn-unet-rec-blur2gt_old.pth fista_c_u=checkpoints/OPhlatCam-unet-rec-fista2gt.pth

    # OCIFAR100 sample (PhlatCam PSF and pretrained weights)
    python demo_deep_learning.py \\
        --psf samples/psf/PhlatCam_psf.png \\
        --blur samples/OCIFAR100/blur/test/apple/00-000.png \\
        --gt samples/OCIFAR100/gt/test/apple/00-000.png \\
        --obj_size 96 96 --blur_size 112 112 --conv_size 480 480 \\
        --methods unet flatnet mswn fista_c_u \\
        --checkpoints unet=checkpoints/OCIFAR100-unet-rec-blur2gt_full_old.pth flatnet=checkpoints/OCIFAR100-flatnet-rec-blur2gt_full_old.pth mswn=checkpoints/OCIFAR100-mswn-unet-rec-blur2gt_full_old.pth fista_c_u=checkpoints/OCIFAR100-unet-rec-fista2gt_full.pth

    # OCIFAR100_sim sample (PhlatCam simulated PSF). Pretrained weights for
    # the simulated dataset are not included; the network runs with random
    # initialization to verify the forward pass.
    python demo_deep_learning.py \\
        --psf samples/psf/PhlatCam_psf_sim.png \\
        --blur samples/OCIFAR100/blur_sim/test/apple/00-000.png \\
        --gt samples/OCIFAR100/gt/test/apple/00-000.png \\
        --methods fista_c_u
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net.unet import UNet
from net.flatnet import FlatNetGenC
from net.mswn_unet import MWDN_CPSF
from net.fista_c_u import FISTA_C_U
from utils.image_loader import opencv_loader
from utils.metrics import psnr, ssim


def load_image_tensor(path, size=None, dtype=torch.float32):
    img = opencv_loader(path)
    transforms = [v2.ToImage(), v2.ToDtype(dtype, scale=True)]
    if size is not None:
        transforms.append(v2.Resize(size))
    return v2.Compose(transforms)(img).unsqueeze(0)


def build_model(method, args, device):
    """Instantiate the requested model."""
    common = dict(
        psf_path=args.psf,
        obj_size=tuple(args.obj_size),
        blur_size=tuple(args.blur_size),
        conv_size=tuple(args.conv_size),
    )

    if method == "unet":
        model = UNet(in_chans=3, nn_upsample=False, obj_size=tuple(args.obj_size))
    elif method == "flatnet":
        model = FlatNetGenC(**common)
    elif method == "mswn":
        model = MWDN_CPSF(psf_path=args.psf, in_chans=3,
                          obj_size=tuple(args.obj_size), conv_size=tuple(args.conv_size))
    elif method == "fista_c_u":
        model = FISTA_C_U(
            psf_path=args.psf, obj_size=tuple(args.obj_size),
            blur_size=tuple(args.blur_size), conv_size=tuple(args.conv_size),
            fista_lamda=args.fista_c_lambda, fista_iters=args.fista_iters,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return model.to(device).eval()


def load_weights(model, checkpoint_path):
    """Load pretrained weights if available."""
    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = state.get("model", state)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from: {checkpoint_path}")
    else:
        print("Warning: no pretrained weights found, using random initialization.")


def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load input images.
    blur = load_image_tensor(args.blur, size=tuple(args.blur_size)).to(device)
    gt = None
    if args.gt and os.path.isfile(args.gt):
        gt = load_image_tensor(args.gt, size=tuple(args.obj_size)).to(device)

    methods = args.methods if args.methods else ["unet", "flatnet", "mswn", "fista_c_u"]
    results = {}

    for method in methods:
        print(f"\n--- Running {method} ---")
        model = build_model(method, args, device)

        ckpt = args.checkpoints.get(method) if args.checkpoints else None
        load_weights(model, ckpt)

        with torch.no_grad():
            if method == "fista_c_u":
                rec = model(blur)
            else:
                rec = model(blur)

        results[method] = rec

        if gt is not None:
            p = psnr(rec, gt).item()
            s = ssim(rec, gt).item()
            print(f"  PSNR: {p:.2f} dB, SSIM: {s:.4f}")

    # --- Summary table ---
    if gt is not None:
        print("\n" + "=" * 55)
        print(f"{'Method':<16} {'PSNR (dB)':>10} {'SSIM':>10}")
        print("-" * 55)
        for name, rec in results.items():
            p = psnr(rec, gt).item()
            s = ssim(rec, gt).item()
            print(f"{name:<16} {p:>10.2f} {s:>10.4f}")
        print("=" * 55)

    # --- Visualization ---
    n_cols = 1 + len(results) + (1 if gt is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    def to_numpy(t):
        return t.squeeze(0).permute(1, 2, 0).cpu().clip(0, 1).numpy()

    col = 0
    axes[col].imshow(to_numpy(blur))
    axes[col].set_title("Blur (input)")
    col += 1
    if gt is not None:
        axes[col].imshow(to_numpy(gt))
        axes[col].set_title("Ground Truth")
        col += 1
    for name, rec in results.items():
        axes[col].imshow(to_numpy(rec))
        axes[col].set_title(name)
        col += 1

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    out_path = args.output or "deep_learning_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nResults saved to: {out_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Deep learning reconstruction demo")
    parser.add_argument("--psf", type=str, required=True, help="Path to PSF image")
    parser.add_argument("--blur", type=str, required=True, help="Path to blur image")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground-truth image (optional)")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["unet", "flatnet", "mswn", "fista_c_u"],
                        choices=["unet", "flatnet", "mswn", "fista_c_u"],
                        help="Methods to evaluate")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Checkpoint paths as method=path pairs, e.g. unet=ckpt/unet.pth")
    parser.add_argument("--obj_size", type=int, nargs=2, default=[96, 96])
    parser.add_argument("--blur_size", type=int, nargs=2, default=[112, 112])
    parser.add_argument("--conv_size", type=int, nargs=2, default=[480, 480])
    parser.add_argument("--fista_c_lambda", type=float, default=1e-4)
    parser.add_argument("--fista_iters", type=int, default=1500)
    parser.add_argument("--output", type=str, default="deep_learning_results.png")
    args = parser.parse_args()

    # Parse method=path checkpoint pairs.
    if args.checkpoints:
        ckpt_dict = {}
        for item in args.checkpoints:
            if "=" in item:
                k, v = item.split("=", 1)
                ckpt_dict[k.strip()] = v.strip()
        args.checkpoints = ckpt_dict
    return args


if __name__ == "__main__":
    run_demo(parse_args())
