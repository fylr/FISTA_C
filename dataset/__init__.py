"""Dataset loaders for lensless reconstruction experiments.

Available datasets:
    - OCIFAR100:     captured lensless dataset (shared gt; blur + blur_sim, 100 classes)
    - OCIFAR100Sim:  simulated lensless dataset, shares the same gt/ root as OCIFAR100
                     (uses blur_sim/ blur, PSF: PhlatCam_psf_sim.png)
    - OPhlatCam:     public PhlatCam dataset (gt / blur, 1000 classes, 99:1 split)
"""

from dataset.base import PairedImageDataset, make_transforms
from dataset.ocifar100 import OCIFAR100, get_ocifar100_loaders
from dataset.ocifar100_sim import OCIFAR100Sim, get_ocifar100sim_loaders
from dataset.ophlatcam import OPhlatCam, get_ophlatcam_loaders

__all__ = [
    "PairedImageDataset",
    "make_transforms",
    "OCIFAR100",
    "get_ocifar100_loaders",
    "OCIFAR100Sim",
    "get_ocifar100sim_loaders",
    "OPhlatCam",
    "get_ophlatcam_loaders",
]
