"""
hexiga.junc.pad_qfspt
=====================

Cyclically pad a sequence
of quad-face points (a ``(3, n)`` matrix) on its left and/or right side by
repeating the boundary point(s).

Signature::

    QFSpt_cpad = padQFSpt(QFSpt, padFactor)

Notes
-----

* ``padFactor``: ``0`` pad both sides, ``1`` right side, ``-1`` left side.
* Unrecognised case prints a message and returns an empty ``(3, 0)`` array
  (MATLAB returns ``[]``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["padQFSpt"]

def padQFSpt(QFSpt: np.ndarray, padFactor: int) -> np.ndarray:
    """Cyclically pad a ``(3, n)`` point matrix on one or both sides."""
    QFSpt = np.asarray(QFSpt, dtype=float)
    if padFactor == 0:
        return np.hstack([QFSpt[:, -1:], QFSpt, QFSpt[:, :1]])
    if padFactor == 1:
        return np.hstack([QFSpt, QFSpt[:, :1]])
    if padFactor == -1:
        return np.hstack([QFSpt[:, -1:], QFSpt])
    print(" * padQFSpt: Unrecognised Case! - Return empty.")
    return np.empty((3, 0), dtype=float)
