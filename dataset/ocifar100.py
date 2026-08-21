"""OCIFAR100 dataset loader (captured lensless dataset).

OCIFAR100 and OCIFAR100_sim share the *same* root directory and the *same*
``gt/`` ground-truth images.  They differ only in the blur source, selected via
``blur_type``:

    ocifar100/                 # common root for OCIFAR100 AND OCIFAR100_sim
    ├── gt/                    # ground-truth scenes (SHARED by both datasets)
    │   ├── train/             # 50,000 images in 100 class folders
    │   └── test/              # 10,000 images in 100 class folders
    ├── blur/                  # real captured blur   (OCIFAR100, blur_type="blur")
    │   ├── train/
    │   └── test/
    └── blur_sim/              # simulated blur        (OCIFAR100_sim, blur_type="blur_sim")
        ├── train/
        └── test/

Corresponding PSFs:
    * OCIFAR100     -> PhlatCam_psf.png
    * OCIFAR100_sim -> PhlatCam_psf_sim.png

The CIFAR-100 source has 50k train / 10k test images; each class folder
contains the corresponding images.  ``blur_type`` selects whether to load the
real captured blur or the simulated blur.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch

from dataset.base import PairedImageDataset, make_transforms


class OCIFAR100(PairedImageDataset):
    """OCIFAR100 captured dataset.

    Args:
        root:           path to the ``ocifar100/`` directory.
        split:          ``'train'`` or ``'test'``.
        blur_type:      ``'blur'`` (real captured) or ``'blur_sim'`` (simulated).
        blur_size:      (H, W) center crop size for blur images.
        obj_size:       (H, W) center crop size for gt images.
        is_train:       if True, apply training augmentation (random flip).
        blur_transform: optional custom transform for blur (overrides default).
        gt_transform:   optional custom transform for gt (overrides default).
    """

    def __init__(self, root: Union[str, Path], split: str = "test",
                 blur_type: str = "blur",
                 blur_size: tuple = (112, 112), obj_size: tuple = (96, 96),
                 is_train: bool = False,
                 blur_transform: Optional[Callable] = None,
                 gt_transform: Optional[Callable] = None):
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split}"
        assert blur_type in ("blur", "blur_sim"), \
            f"blur_type must be 'blur' or 'blur_sim', got {blur_type}"

        root = Path(root)
        blur_dir = root / blur_type / split
        gt_dir = root / "gt" / split

        if blur_transform is None:
            blur_transform = make_transforms(blur_size, is_train=is_train)
        if gt_transform is None:
            gt_transform = make_transforms(obj_size, is_train=is_train)

        super().__init__(blur_dir=blur_dir, gt_dir=gt_dir,
                         blur_transform=blur_transform,
                         gt_transform=gt_transform)


def get_ocifar100_loaders(root: Union[str, Path],
                          blur_type: str = "blur",
                          blur_size: tuple = (112, 112),
                          obj_size: tuple = (96, 96),
                          batch_size: int = 16,
                          num_workers: int = 4,
                          train_ratio: float = 0.9,
                          seed: int = 0) -> Tuple[torch.utils.data.DataLoader,
                                                   torch.utils.data.DataLoader]:
    """Create train/validation DataLoaders for OCIFAR100.

    The original CIFAR-100 train set (50k images) is split into train /
    validation according to ``train_ratio``.  The test set (10k images) is
    returned as the validation loader when ``train_ratio`` is 1.0.

    Args:
        root:        path to the ``ocifar100/`` directory.
        blur_type:   ``'blur'`` or ``'blur_sim'``.
        blur_size:   blur center crop size.
        obj_size:    gt center crop size.
        batch_size:  batch size.
        num_workers: number of data loading workers.
        train_ratio: fraction of the train split used for training (rest = val).
        seed:        random seed for the train/val split.

    Returns:
        (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, random_split

    full_train = OCIFAR100(root, split="train", blur_type=blur_type,
                           blur_size=blur_size, obj_size=obj_size, is_train=True)
    test_set = OCIFAR100(root, split="test", blur_type=blur_type,
                         blur_size=blur_size, obj_size=obj_size, is_train=False)

    if train_ratio < 1.0:
        n_train = int(len(full_train) * train_ratio)
        n_val = len(full_train) - n_train
        train_set, val_set = random_split(
            full_train, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        # Validation set should not use training augmentation.
        val_set.dataset.is_train = False
        val_set.dataset.blur_transform = make_transforms(blur_size, is_train=False)
        val_set.dataset.gt_transform = make_transforms(obj_size, is_train=False)
    else:
        train_set = full_train
        val_set = test_set

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
