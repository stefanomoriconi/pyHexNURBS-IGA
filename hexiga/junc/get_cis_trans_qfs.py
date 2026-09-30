"""
hexiga.junc.get_cis_trans_qfs
=============================

Build the Cis and
Trans quad-face (QFS) index columns that insert a new midpoint ``M`` into the
four-point QFS ``(V, A, X, B)``.

``QFSfcs`` values are **1-based** labels (MATLAB convention).

Signature::

    [cQFSfc, tQFSfc, QFSpts] = getCisTransQFSs(QFSpt, M)   % QFSpt:3x4, M:3x1
"""

from __future__ import annotations

import numpy as np

__all__ = ["getCisTransQFSs"]

def getCisTransQFSs(QFSpt: np.ndarray, M: np.ndarray):
    """Build the Cis/Trans QFS index columns inserting midpoint ``M``.

    ``QFSpt`` (3 x 4) = ``[V, A, X, B]``; ``M`` (3 x 1) is the new partition
    point.  Returns ``cQFSfc`` (4 x 2), ``tQFSfc`` (4 x 2) with 1-based face
    labels, and the extended point set ``QFSpts`` (3 x 5) = ``[V, A, M, B, X]``.
    """
    QFSpt = np.asarray(QFSpt, dtype=float)
    M = np.asarray(M, dtype=float)

    V = QFSpt[:, 0]
    A = QFSpt[:, 1]
    X = QFSpt[:, 2]
    B = QFSpt[:, 3]

    QFSpts = np.column_stack([V, A, M, B, X])

    # Cis
    cQFSfc = np.array([[1, 2, 5, 3], [1, 3, 5, 4]]).T
    # Trans
    tQFSfc = np.array([[2, 5, 4, 3], [2, 3, 4, 1]]).T

    return cQFSfc, tQFSfc, QFSpts
