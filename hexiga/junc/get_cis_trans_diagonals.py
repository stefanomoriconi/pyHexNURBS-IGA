"""
hexiga.junc.get_cis_trans_diagonals
===================================

Retrieve both
the Cis and Trans possible diagonals of a single QFS side defined by the
4-points in ``QFSpt``.

Signature::

    [VXcis, VXtrans, VXcisAvg, VXtransAvg] = getCisTransDiagonals(QFSpt)   % QFSpt: 3x4
"""

from __future__ import annotations

import numpy as np

from .uvect import uvect

__all__ = ["getCisTransDiagonals"]

def getCisTransDiagonals(QFSpt: np.ndarray):
    """Retrieve the Cis/Trans diagonals of the QFS side ``QFSpt`` (3 x 4)."""
    QFSpt = np.asarray(QFSpt, dtype=float)

    VXcis = QFSpt[:, 2] - QFSpt[:, 0]
    VXcis = uvect(VXcis)
    VXcisAvg = np.mean(np.column_stack([QFSpt[:, 0], QFSpt[:, 2]]), axis=1)

    VXtrans = QFSpt[:, 3] - QFSpt[:, 1]
    VXtrans = uvect(VXtrans)
    VXtransAvg = np.mean(np.column_stack([QFSpt[:, 1], QFSpt[:, 3]]), axis=1)

    return VXcis, VXtrans, VXcisAvg, VXtransAvg
