"""pairing_abc.py

Parse pairs for datasets organized like:
  sample/<batch_id>/{A1,A2,A3,B1,B2,B3,...}.{jpg|jpeg|png|webp}

Default pairing strategy:
  A<k> <-> B<k> for each k.

Returns list of tuples:
  (batch_id, img_left_path, img_right_path, specimen_no)

Notes:
- This module is independent of QR/OCR pairing.
- It is designed for the user's "A1,B1,..." naming convention.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


_NAME_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _index_images(dir_path: Path) -> Dict[Tuple[str, str], Path]:
    """Map (group, idx) -> file path."""
    out: Dict[Tuple[str, str], Path] = {}
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        m = _NAME_RE.match(p.stem)
        if not m:
            continue
        group = m.group(1).upper()
        idx = m.group(2)
        out[(group, idx)] = p
    return out


def parse_abc_pairs(samples_root: Path, left_group: str = "A", right_group: str = "B") -> List[Tuple[str, Path, Optional[Path], int]]:
    """Parse A/B pairs inside each batch directory.

    If a right image is missing, right_path is None (will become INVALID upstream).
    specimen_no is the numeric index.
    """
    left_group = left_group.upper()
    right_group = right_group.upper()

    pairs: List[Tuple[str, Path, Optional[Path], int]] = []

    batch_dirs = sorted([d for d in samples_root.iterdir() if d.is_dir()])
    for batch_dir in batch_dirs:
        batch_id = batch_dir.name
        idx_map = _index_images(batch_dir)

        # collect indices existing in left group
        left_indices = sorted({idx for (g, idx) in idx_map.keys() if g == left_group}, key=lambda x: int(x))
        for idx in left_indices:
            left_path = idx_map[(left_group, idx)]
            right_path = idx_map.get((right_group, idx))
            pairs.append((batch_id, left_path, right_path, int(idx)))

    return pairs
