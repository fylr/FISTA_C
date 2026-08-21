"""Base class for paired blur / ground-truth image datasets.

All datasets in this project follow an ImageFolder-style layout where blur and
ground-truth images share identical subdirectory structures (same class
folders, same filenames), allowing samples to be matched by index.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms import v2

from utils.image_loader import opencv_loader


class PairedImageDataset(Dataset):
    """Load paired (blur, gt) images from two parallel ImageFolder directories.

    Args:
        blur_dir:       root directory of blur images (ImageFolder layout).
        gt_dir:         root directory of ground-truth images (same layout).
        blur_transform: transform applied to blur images.
        gt_transform:   transform applied to gt images.
        loader:         image loading function (returns NumPy RGB array).
    """

    def __init__(self, blur_dir: Union[str, Path], gt_dir: Union[str, Path],
                 blur_transform: Optional[Callable] = None,
                 gt_transform: Optional[Callable] = None,
                 loader: Callable = opencv_loader):
        self.blur_dir = Path(blur_dir)
        self.gt_dir = Path(gt_dir)
        self.loader = loader

        # Use ImageFolder to traverse directories and get sorted sample lists.
        blur_if = ImageFolder(str(self.blur_dir))
        gt_if = ImageFolder(str(self.gt_dir))

        assert len(blur_if.samples) == len(gt_if.samples), (
            f"blur ({len(blur_if.samples)}) and gt ({len(gt_if.samples)}) "
            f"must have the same number of images."
        )

        self.blur_paths = [s[0] for s in blur_if.samples]
        self.gt_paths = [s[0] for s in gt_if.samples]
        self.labels = [s[1] for s in blur_if.samples]
        self.classes = blur_if.classes
        self.class_to_idx = blur_if.class_to_idx

        self.blur_transform = blur_transform
        self.gt_transform = gt_transform

    def __len__(self) -> int:
        return len(self.blur_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (blur_tensor, gt_tensor, class_label)."""
        blur_img = self.loader(self.blur_paths[idx])
        gt_img = self.loader(self.gt_paths[idx])

        if self.blur_transform is not None:
            blur = self.blur_transform(blur_img)
        else:
            blur = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(blur_img)

        if self.gt_transform is not None:
            gt = self.gt_transform(gt_img)
        else:
            gt = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(gt_img)

        label = self.labels[idx]
        return blur, gt, label


def make_transforms(crop_size: tuple, is_train: bool = False,
                    additional: Optional[list] = None) -> v2.Compose:
    """Build a standard transform pipeline: ToImage -> ToDtype -> CenterCrop.

    Args:
        crop_size:  (H, W) center crop size.
        is_train:   if True, add random horizontal flip (for training only).
        additional: optional list of extra transforms to append.
    """
    tfms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.CenterCrop(size=list(crop_size)),
    ]
    if is_train:
        tfms.append(v2.RandomHorizontalFlip(p=0.5))
    if additional:
        tfms.extend(additional)
    return v2.Compose(tfms)
