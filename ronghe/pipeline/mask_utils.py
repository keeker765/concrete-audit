"""mask_utils.py

Utilities for applying masks to keypoints.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def filter_keypoints_by_valid_mask(kpts: torch.Tensor, valid_mask_u8: Optional[np.ndarray]):
    """Keep only keypoints that fall inside valid_mask (mask>0).

    Args:
      kpts: [N,2] (x,y)
      valid_mask_u8: HxW uint8 mask with 1 for valid region

    Returns:
      keep_idx (torch.LongTensor), filtered_kpts
    """
    if valid_mask_u8 is None or kpts is None or len(kpts) == 0:
        keep = torch.arange(0, 0 if kpts is None else len(kpts), dtype=torch.long)
        return keep, kpts

    h, w = valid_mask_u8.shape[:2]
    pts = kpts.detach().cpu().numpy()
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    good = valid_mask_u8[ys, xs] > 0
    keep_idx = torch.from_numpy(np.where(good)[0].astype(np.int64))
    return keep_idx, kpts[keep_idx]
