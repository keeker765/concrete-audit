"""yolo_mask.py

Use Ultralytics YOLO segmentation model (best(1).pt) to obtain a binary mask
for the main concrete block region.

This is adapted from daima/one_click_cnn.py logic.

The mask is intended to be used as a *valid region* constraint for keypoint
selection/matching (not the sticker-exclude mask).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class YoloSegMasker:
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = device
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        from ultralytics import YOLO  # lazy import
        model = YOLO(str(self.model_path))
        model.model.eval()
        model.model.to("cpu")  # keep consistent with existing scripts
        self._model = model

    def segment_main_object(self, img_bgr) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Return (mask_u8, box_xyxy) in original image coordinates.

        mask_u8: 0/1 uint8 mask, same HxW as img.
        """
        self._lazy_load()
        results = self._model.predict(source=img_bgr, device="cpu", verbose=False)
        result = results[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return None, None

        scores = result.boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(scores))

        mask = result.masks.data[best_idx].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)

        # resize to image size
        import cv2
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        box_xyxy = None
        if result.boxes.xyxy is not None and len(result.boxes.xyxy) > best_idx:
            box_xyxy = tuple(result.boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist())

        return mask, box_xyxy
