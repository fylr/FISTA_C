"""Evaluate reconstruction methods on a dataset test set.

This script loads a dataset (OCIFAR100 / OCIFAR100_sim / OPhlatCam), runs a
selected reconstruction method on the test split, and reports PSNR / SSIM /
LPIPS metrics.

Usage:
    # Traditional FISTA_C on OCIFAR100 test set (PSF: PhlatCam_psf.png)
    python evaluate.py --dataset ocifar100 --ds_path samples/OCIFAR100 \\
        --psf samples/psf/PhlatCam_psf.png --method fista_c

    # Traditional FISTA_C on OCIFAR100_sim test set (PSF: PhlatCam_psf_sim.png)
    python evaluate.py --dataset ocifar100_sim --ds_path samples/OCIFAR100 \\
        --psf samples/psf/PhlatCam_psf_sim.png --method fista_c

    # Deep learning FISTA_C+U on PhlatCam test set
    python evaluate.py --dataset ophlatcam --ds_path /path/to/OPhlatCam \\
        --psf samples/psf/OPhlatCam_psf.png --method fista_c_u \\
        --checkpoint checkpoints/OPhlatCam-unet-rec-fista2gt.pth
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import OCIFAR100, OCIFAR100Sim, OPhlatCam
from algorithms.reconstruction import LenslessReconstructor
from net.unet import UNet
from net.flatnet import FlatNetGenC
from net.mswn_unet import MWDN_CPSF
from net.fista_c_u import FISTA_C_U
from utils.metrics import psnr, ssim, LPIPSMetric


def build_dataset(name, ds_path, blur_size, obj_size):
    """Instantiate the test set for the given dataset."""
    if name == "ocifar100":
        return OCIFAR100(ds_path, split="test", blur_type="blur",
                         blur_size=blur_size, obj_size=obj_size, is_train=False)
    elif name == "ocifar100_sim":
        return OCIFAR100Sim(ds_path, split="test",
                            blur_size=blur_size, obj_size=obj_size, is_train=False)
    elif name == "ophlatcam":
        # PhlatCam has no predefined split; use full dataset as test for simplicity.
        return OPhlatCam(ds_path, blur_size=blur_size, obj_size=obj_size, is_train=False)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def build_traditional(method, reconstructor, lamda, iters):
    """Return a callable that runs a traditional algorithm on a blur tensor."""
    if method == "wiener":
        def fn(blur):
            reconstructor.set_measurement(blur)
            rec, _ = reconstructor.wiener()
            return rec
        return fn
    elif method == "fista_o":
        def fn(blur):
            reconstructor.set_measurement(blur)
            rec, _ = reconstructor.fista_o(lamda=lamda, iters=iters)
            return rec
        return fn
    elif method == "fista_c":
        def fn(blur):
            reconstructor.set_measurement(blur)
            rec, _ = reconstructor.fista_c(lamda=lamda, iters=iters)
            return rec
        return fn
    else:
        raise ValueError(f"Unknown traditional method: {method}")


def build_dl_model(method, args, device):
    """Instantiate a deep learning model and load weights if provided."""
    common = dict(psf_path=args.psf, obj_size=tuple(args.obj_size),
                  blur_size=tuple(args.blur_size), conv_size=tuple(args.conv_size))
    if method == "unet":
        model = UNet(in_chans=3, obj_size=tuple(args.obj_size))
    elif method == "flatnet":
        model = FlatNetGenC(**common)
    elif method == "mswn":
        model = MWDN_CPSF(psf_path=args.psf, obj_size=tuple(args.obj_size),
                          conv_size=tuple(args.conv_size))
    elif method == "fista_c_u":
        model = FISTA_C_U(psf_path=args.psf, obj_size=tuple(args.obj_size),
                          blur_size=tuple(args.blur_size),
                          conv_size=tuple(args.conv_size),
                          fista_lamda=args.lamda, fista_iters=args.iters)
    else:
        raise ValueError(f"Unknown deep learning method: {method}")

    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    return model.to(device).eval()


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}  |  Method: {args.method}")

    # Dataset
    dataset = build_dataset(args.dataset, args.ds_path,
                            tuple(args.blur_size), tuple(args.obj_size))
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)
    print(f"Test samples: {len(dataset)}")

    # Metrics
    lpips_fn = LPIPSMetric().to(device) if args.use_lpips else None

    # Build reconstruction function
    traditional_methods = {"wiener", "fista_o", "fista_c"}
    if args.method in traditional_methods:
        reconstructor = LenslessReconstructor(
            psf_path=args.psf, obj_size=tuple(args.obj_size),
            blur_size=tuple(args.blur_size), conv_size=tuple(args.conv_size),
            sensor_size=tuple(args.blur_size), weight=args.psf_weight,
        ).to(device)
        rec_fn = build_traditional(args.method, reconstructor, args.lamda, args.iters)
    else:
        model = build_dl_model(args.method, args, device)
        def rec_fn(blur):
            return model(blur)

    # Evaluation loop
    psnr_sum, ssim_sum, lpips_sum = 0.0, 0.0, 0.0
    n_samples = 0
    t0 = time.time()

    for batch_idx, (blur, gt, _) in enumerate(loader):
        blur = blur.to(device)
        gt = gt.to(device)

        rec = rec_fn(blur)

        # Ensure rec matches gt size (center crop if needed).
        if rec.shape[-2:] != gt.shape[-2:]:
            import torchvision.transforms.v2.functional as v2f
            rec = v2f.center_crop(rec, gt.shape[-2:])

        bs = gt.shape[0]
        for i in range(bs):
            psnr_sum += psnr(rec[i], gt[i]).item()
            ssim_sum += ssim(rec[i], gt[i]).item()
            if lpips_fn is not None:
                lpips_sum += lpips_fn(rec[i], gt[i]).item()
        n_samples += bs

        if (batch_idx + 1) % args.log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}]  "
                  f"PSNR={psnr_sum/n_samples:.2f}  "
                  f"SSIM={ssim_sum/n_samples:.4f}" +
                  (f"  LPIPS={lpips_sum/n_samples:.4f}" if lpips_fn else ""))

    elapsed = time.time() - t0
    print("\n" + "=" * 55)
    print(f"Results on {args.dataset} test set ({n_samples} samples, {elapsed:.1f}s)")
    print(f"  PSNR:  {psnr_sum / n_samples:.2f} dB")
    print(f"  SSIM:  {ssim_sum / n_samples:.4f}")
    if lpips_fn:
        print(f"  LPIPS: {lpips_sum / n_samples:.4f}")
    print("=" * 55)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction on dataset test set")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ocifar100", "ocifar100_sim", "ophlatcam"])
    parser.add_argument("--ds_path", type=str, required=True, help="Root path of the dataset")
    parser.add_argument("--psf", type=str, required=True, help="Path to PSF image")
    parser.add_argument("--method", type=str, required=True,
                        choices=["wiener", "fista_o", "fista_c",
                                 "unet", "flatnet", "mswn", "fista_c_u"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained model weights (for DL methods)")
    parser.add_argument("--obj_size", type=int, nargs=2, default=[96, 96])
    parser.add_argument("--blur_size", type=int, nargs=2, default=[112, 112])
    parser.add_argument("--conv_size", type=int, nargs=2, default=[480, 480])
    parser.add_argument("--psf_weight", type=float, default=1.0)
    parser.add_argument("--lamda", type=float, default=1e-4,
                        help="Regularization weight for FISTA methods")
    parser.add_argument("--iters", type=int, default=1500,
                        help="Maximum FISTA iterations")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--use_lpips", action="store_true",
                        help="Compute LPIPS metric (requires lpips package)")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
