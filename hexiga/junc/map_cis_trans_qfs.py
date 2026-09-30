"""
hexiga.junc.map_cis_trans_qfs
=============================

Map a single 4-point
QFS face (labels) together with a new midpoint label ``Mlbl`` into the two
Cis or Trans quad-face index columns.

All face labels are **1-based** (MATLAB convention).

Signature::

    QFSfcs = mapCisTransQFSs(QFSfc, Mlbl, FlagCIS)   % QFSfc: 4x1, QFSfcs: 4x2
"""

from __future__ import annotations

import numpy as np

__all__ = ["mapCisTransQFSs"]

def mapCisTransQFSs(QFSfc: np.ndarray, Mlbl: int, FlagCIS: bool) -> np.ndarray:
    """Map a QFS face ``(V, A, X, B)`` + midpoint label into two Cis/Trans faces."""
    QFSfc = np.asarray(QFSfc, dtype=int).ravel()

    Vlbl = QFSfc[0]
    Albl = QFSfc[1]
    Xlbl = QFSfc[2]
    Blbl = QFSfc[3]

    if FlagCIS:  # Cis-configuration
        # MATLAB: [ [V;A;X;M], [V;M;X;B] ]  -> each face is one *column*.
        QFSfcs = np.array([Vlbl, Albl, Xlbl, Mlbl,
                           Vlbl, Mlbl, Xlbl, Blbl]).reshape(4, 2, order="F")
    else:        # Trans-configuration
        # MATLAB: [ [A;X;B;M], [A;M;B;V] ]  -> each face is one *column*.
        QFSfcs = np.array([Albl, Xlbl, Blbl, Mlbl,
                           Albl, Mlbl, Blbl, Vlbl]).reshape(4, 2, order="F")

    return QFSfcs
