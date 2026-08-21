"""OPhlatCam dataset loader (public PhlatCam lensless dataset).

Directory layout::

    ophlatcam/
    ├── gt/
    │   ├── class_0000/   *.png
    │   ├── class_0001/   *.png
    │   └── ...           (1000 classes, ~10 images each = 10k total)
    └── blur/
        ├── class_0000/   *.png
        ├── class_0001/   *.png
        └── ...           (1000 classes)

Unlike OCIFAR100, PhlatCam does not have predefined train/test subdirectories.
Following the original paper (Khan et al., 2022), the 10,000 image pairs are
split 99:1 into train / test programmatically.  Images are downsampled to
640 x 704, with the effective scene region (view) of size 192 x 192.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Subset

from dataset.base import PairedImageDataset, make_transforms


class OPhlatCam(PairedImageDataset):
    """PhlatCam public dataset.

    Args:
        root:           path to the ``ophlatcam/`` directory.
        blur_size:      (H, W) center crop size for blur images.
        obj_size:       (H, W) center crop size for gt images.
        is_train:       if True, apply training augmentation.
        blur_transform: optional custom transform for blur.
        gt_transform:   optional custom transform for gt.
    """

    def __init__(self, root: Union[str, Path],
                 blur_size: tuple = (224, 224), obj_size: tuple = (192, 192),
                 is_train: bool = False,
                 blur_transform: Optional[Callable] = None,
                 gt_transform: Optional[Callable] = None):
        root = Path(root)
        blur_dir = root / "blur"
        gt_dir = root / "gt"

        if blur_transform is None:
            blur_transform = make_transforms(blur_size, is_train=is_train)
        if gt_transform is None:
            gt_transform = make_transforms(obj_size, is_train=is_train)

        super().__init__(blur_dir=blur_dir, gt_dir=gt_dir,
                         blur_transform=blur_transform,
                         gt_transform=gt_transform)


def get_ophlatcam_loaders(root: Union[str, Path],
                          blur_size: tuple = (224, 224),
                          obj_size: tuple = (192, 192),
                          batch_size: int = 8,
                          num_workers: int = 4,
                          train_ratio: float = 0.99,
                          seed: int = 0) -> Tuple[torch.utils.data.DataLoader,
                                                   torch.utils.data.DataLoader]:
    """Create train/test DataLoaders for PhlatCam.

    The full dataset (10,000 pairs) is split into train / test according to
    ``train_ratio`` (default 99:1, matching the original paper).

    Args:
        root:        path to the ``ophlatcam/`` directory.
        blur_size:   blur center crop size.
        obj_size:    gt center crop size.
        batch_size:  batch size.
        num_workers: number of data loading workers.
        train_ratio: fraction of data used for training (rest = test).
        seed:        random seed for the split.

    Returns:
        (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    full_dataset = OPhlatCam(root, blur_size=blur_size, obj_size=obj_size,
                             is_train=True)

    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_set = Subset(full_dataset, train_indices)

    # Test set uses no augmentation.
    test_full = OPhlatCam(root, blur_size=blur_size, obj_size=obj_size,
                          is_train=False)
    test_set = Subset(test_full, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
