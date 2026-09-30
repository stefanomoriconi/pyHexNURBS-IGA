"""
hexiga.smth.find_ctrl_pt
========================

Faithful port of MATLAB ``smth_utils/multipatch/findCtrlPt.m``.

Locate an exact (toleranced) match of a single 3D point against a bank of 3D
points using an L2 distance test.
"""

from __future__ import annotations

import numpy as np

__all__ = ["find_ctrl_pt"]

def find_ctrl_pt(pt3D, pts3D, errtol: float = 1e-9) -> np.ndarray:
    """Return the (1-based) indices of ``pts3D`` within ``errtol`` of ``pt3D``.

    Parameters
    ----------
    pt3D
        The target point, shape ``(3,)``.
    pts3D
        The bank of points, shape ``(3, N)``.
    errtol
        L2 tolerance (MATLAB default ``1e-9``).

    Returns
    -------
    np.ndarray
        1-based column indices whose points match, shape ``(M,)`` (may be
        empty).
    """
    pt3D = np.asarray(pt3D, dtype=float).reshape(3)
    pts3D = np.asarray(pts3D, dtype=float)
    if pts3D.size == 0:
        return np.zeros(0, dtype=int)
    # pts3D is (3, N); broadcast the target across columns.
    diff = np.repeat(pt3D[:, None], pts3D.shape[1], axis=1) - pts3D
    L2 = np.sqrt(np.sum(diff ** 2, axis=0))
    return np.nonzero(L2 < errtol)[0] + 1
