"""Demo: traditional model-based reconstruction algorithms (Wiener, FISTA_O, FISTA_C).

This script demonstrates the three traditional optimization algorithms from
the paper (Section 2.2):

  * Wiener    -- analytical solution of the optimized imaging model (Eq. 4)
  * FISTA_O   -- FISTA on the original imaging model with sparse regularization (Eq. 5)
  * FISTA_C   -- FISTA on the optimized imaging model with TV regularization (Eq. 6, proposed)

Usage:
    # PhlatCam sample (PSF and data included in the repo)
    python demo_traditional.py \\
        --psf samples/psf/PhlatCam_psf.png \\
        --blur samples/OPhlatCam/blur/n01440764/n01440764_1154.png \\
        --gt samples/OPhlatCam/gt/n01440764/n01440764_1154.png \\
        --obj_size 192 192 --blur_size 224 224 --conv_size 640 704

    # OCIFAR100 sample (PhlatCam PSF)
    python demo_traditional.py \\
        --psf samples/psf/PhlatCam_psf.png \\
        --blur samples/OCIFAR100/blur/test/apple/00-000.png \\
        --gt samples/OCIFAR100/gt/test/apple/00-000.png \\
        --obj_size 96 96 --blur_size 112 112 --conv_size 480 480

    # OCIFAR100_sim sample (PhlatCam simulated PSF)
    python demo_traditional.py \\
        --psf samples/psf/PhlatCam_psf_sim.png \\
        --blur samples/OCIFAR100/blur_sim/test/apple/00-000.png \\
        --gt samples/OCIFAR100/gt/test/apple/00-000.png \\
        --obj_size 96 96 --blur_size 112 112 --conv_size 480 480

If no blur image is provided, a synthetic measurement is generated from a
random or built-in test pattern using the forward imaging model.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2

# Ensure the project root is on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.reconstruction import LenslessReconstructor
from utils.image_loader import opencv_loader
from utils.metrics import psnr, ssim


def load_image_tensor(path, size=None, dtype=torch.float32):
    """Load an image and convert to a (1, C, H, W) tensor in [0, 1]."""
    img = opencv_loader(path)
    transforms = [v2.ToImage(), v2.ToDtype(dtype, scale=True)]
    if size is not None:
        transforms.append(v2.Resize(size))
    return v2.Compose(transforms)(img).unsqueeze(0)


def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Geometry parameters (OCIFAR100 setup from the paper).
    obj_size = tuple(args.obj_size)
    blur_size = tuple(args.blur_size)
    conv_size = tuple(args.conv_size)

    reconstructor = LenslessReconstructor(
        psf_path=args.psf,
        obj_size=obj_size,
        blur_size=blur_size,
        conv_size=conv_size,
        sensor_size=blur_size,  # input is a pre-cropped blur_crop
        weight=args.psf_weight,
    ).to(device)

    # Load or simulate the measurement.
    if args.blur and os.path.isfile(args.blur):
        blur = load_image_tensor(args.blur, size=blur_size).to(device)
        reconstructor.set_measurement(blur)
        gt = None
        if args.gt and os.path.isfile(args.gt):
            gt = load_image_tensor(args.gt, size=obj_size).to(device)
    else:
        print("No blur image provided, generating synthetic measurement...")
        if args.gt and os.path.isfile(args.gt):
            obj = load_image_tensor(args.gt, size=obj_size).to(device)
        else:
            # Synthetic test pattern: radial gradient + edges.
            y, x = torch.meshgrid(torch.linspace(-1, 1, obj_size[0]),
                                  torch.linspace(-1, 1, obj_size[1]), indexing="ij")
            obj = (0.5 + 0.5 * torch.sin(4 * x) * torch.cos(4 * y)).unsqueeze(0).unsqueeze(0)
            obj = obj.repeat(1, 3, 1, 1).to(device)
        blur_crop, blur_full = reconstructor.simulate_measurement(obj)
        blur = blur_crop
        gt = obj

    # --- Wiener ---
    print("\n[1/3] Running Wiener reconstruction...")
    rec_wiener, _ = reconstructor.wiener()

    # --- FISTA_O ---
    print("[2/3] Running FISTA_O reconstruction...")
    rec_fista_o, _ = reconstructor.fista_o(
        lamda=args.fista_o_lambda, iters=args.iters,
    )

    # --- FISTA_C ---
    print("[3/3] Running FISTA_C reconstruction...")
    rec_fista_c, _ = reconstructor.fista_c(
        lamda=args.fista_c_lambda, iters=args.iters,
    )

    # --- Evaluation ---
    results = {
        "Wiener": rec_wiener,
        "FISTA_O": rec_fista_o,
        "FISTA_C": rec_fista_c,
    }

    print("\n" + "=" * 60)
    print(f"{'Method':<12} {'PSNR (dB)':>10} {'SSIM':>10}")
    print("-" * 60)
    for name, rec in results.items():
        if gt is not None:
            p = psnr(rec, gt).item()
            s = ssim(rec, gt).item()
            print(f"{name:<12} {p:>10.2f} {s:>10.4f}")
        else:
            print(f"{name:<12} {'(no GT)':>10} {'(no GT)':>10}")
    print("=" * 60)

    # --- Visualization ---
    n_cols = 3 + (1 if gt is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    def to_numpy(t):
        return t.squeeze(0).permute(1, 2, 0).cpu().clip(0, 1).numpy()

    col = 0
    if gt is not None:
        axes[col].imshow(to_numpy(gt))
        axes[col].set_title("Ground Truth")
        col += 1
    axes[col].imshow(to_numpy(rec_wiener))
    axes[col].set_title("Wiener")
    col += 1
    axes[col].imshow(to_numpy(rec_fista_o))
    axes[col].set_title("FISTA_O")
    col += 1
    axes[col].imshow(to_numpy(rec_fista_c))
    axes[col].set_title("FISTA_C (proposed)")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    out_path = args.output or "traditional_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nResults saved to: {out_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Traditional reconstruction demo")
    parser.add_argument("--psf", type=str, required=True, help="Path to PSF image")
    parser.add_argument("--blur", type=str, default=None, help="Path to blur image (optional)")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground-truth image (optional)")
    parser.add_argument("--obj_size", type=int, nargs=2, default=[96, 96], help="Effective scene size (H W)")
    parser.add_argument("--blur_size", type=int, nargs=2, default=[112, 112], help="Blur crop size (H W)")
    parser.add_argument("--conv_size", type=int, nargs=2, default=[480, 480], help="Convolution domain size (H W)")
    parser.add_argument("--psf_weight", type=float, default=1.0, help="PSF brightness scaling")
    parser.add_argument("--fista_o_lambda", type=float, default=1e-2, help="FISTA_O sparse regularization weight")
    parser.add_argument("--fista_c_lambda", type=float, default=1e-4, help="FISTA_C TV regularization weight")
    parser.add_argument("--iters", type=int, default=1500, help="Maximum FISTA iterations")
    parser.add_argument("--output", type=str, default="traditional_results.png", help="Output figure path")
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
