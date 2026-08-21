"""Image loading utilities supporting grayscale / color, 8-bit / 16-bit PNG and TIFF.

The ``opencv_loader`` returns a NumPy array in RGB order (not BGR), which is
what ``torchvision.transforms.v2.ToImage`` expects.
"""

import cv2
import numpy as np
from PIL import Image


def opencv_loader(path: str, chans: int = 3) -> np.ndarray:
    """Load an image with OpenCV and return an RGB NumPy array.

    Args:
        path:  image file path (png / tif / jpg, 8-bit or 16-bit).
        chans: 1 for grayscale, 3 for RGB.

    Returns:
        NumPy array of shape (H, W) or (H, W, 3), dtype matching the source.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if chans == 1:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif chans == 3:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"chans must be 1 or 3, got {chans}")

    return img


def pil_loader(path: str, chans: int = 3) -> Image.Image:
    """Load an image with PIL (fallback for formats OpenCV may struggle with)."""
    with open(path, "rb") as f:
        img = Image.open(f)
        if chans == 1:
            return img.convert("L")
        elif chans == 3:
            return img.convert("RGB")
        else:
            raise ValueError(f"chans must be 1 or 3, got {chans}")
