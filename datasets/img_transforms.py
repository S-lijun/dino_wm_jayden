from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms


def default_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def letterbox_resize_and_normalize(
    images: torch.Tensor, img_h: int, img_w: int
) -> torch.Tensor:
    """Resize to img_h x img_w (480:640 and 192:256 share 3:4 — no padding needed)."""
    if images.shape[-2] != img_h or images.shape[-1] != img_w:
        images = F.interpolate(
            images, size=(img_h, img_w), mode="bilinear", align_corners=False
        )
    return (images - 0.5) / 0.5


class LetterboxTransform:
  """Resize to fixed HxW preserving 480:640 aspect when img_h:img_w = 3:4."""

  def __init__(self, img_h: int = 192, img_w: int = 256):
      self.img_h = int(img_h)
      self.img_w = int(img_w)

  def __call__(self, images: torch.Tensor) -> torch.Tensor:
      return letterbox_resize_and_normalize(images, self.img_h, self.img_w)


def letterbox_transform(img_h: int = 192, img_w: int = 256) -> LetterboxTransform:
    """Hydra entry; name kept for config compatibility — direct resize, no pad."""
    return LetterboxTransform(img_h=img_h, img_w=img_w)


def get_transform_hw(
    transform: Optional[Callable], default: Tuple[int, int] = (192, 256)
) -> Tuple[int, int]:
    if transform is None:
        return default
    if hasattr(transform, "img_h") and hasattr(transform, "img_w"):
        return int(transform.img_h), int(transform.img_w)
    if isinstance(transform, LetterboxTransform):
        return transform.img_h, transform.img_w
    for step in getattr(transform, "transforms", []):
        if isinstance(step, transforms.Resize):
            size = step.size
            if isinstance(size, int):
                return size, size
            return int(size[0]), int(size[1])
    return default
