"""
hexiga.junc.split_qfspt
=======================

For each quad-face
point, produce the "split" point on the segment to its right neighbour by
scaling the (centred) endpoints by ``sqrt(2)/2`` and re-adding the centre.

Signature::

    QFSpt_split = splitQFSpt(QFSpt, C)

Notes
-----

* The original ``mean(...)`` line is commented out in the MATLAB source; the
  active line uses ``(sqrt(2)/2)*P(:,jj) + (sqrt(2)/2)*P(:,jj+1)`` after
  right-padding (``padQFSpt(..., 1)``) and centring about ``C``.
* ``C`` is a ``(3,1)`` centre vector broadcast across columns.
"""

from __future__ import annotations

import numpy as np

from .pad_qfspt import padQFSpt

__all__ = ["splitQFSpt"]

def splitQFSpt(QFSpt: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Compute the split points for a ``(3, n)`` point matrix about centre ``C``."""
    QFSpt = np.asarray(QFSpt, dtype=float)
    C = np.asarray(C, dtype=float).reshape(-1)

    QFSpt_pad = padQFSpt(QFSpt, 1)
    QFSpt_pad = QFSpt_pad - C[:, None]

    n = QFSpt.shape[1]
    QFSpt_split = np.zeros_like(QFSpt)
    for jj in range(n):
        QFSpt_split[:, jj] = (np.sqrt(2) / 2) * QFSpt_pad[:, jj] + (np.sqrt(2) / 2) * QFSpt_pad[:, jj + 1]

    QFSpt_split = QFSpt_split + C[:, None]
    return QFSpt_split
