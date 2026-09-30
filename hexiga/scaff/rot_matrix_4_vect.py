"""
hexiga.scaff.rot_matrix_4_vect
==============================

Estimate the
3x3 rotation matrix that aligns the orthogonal basis ``(x0, y0)`` to the
orthogonal basis ``(x1, y1)`` via two sequential Rodrigues rotations
(first ``x0 -> x1``, then ``y0 -> y1`` in the rotated frame).

Signature: ``R = RotMatrix4Vect(x0, y0, x1, y1)``.

Notes
-----

* All inputs are unit vectors stored as **row** vectors of shape ``(1, 3)``;
  the result is a ``3 x 3`` matrix.
* ``tol = 1e-3`` guards the "equal / opposite / different" tests.
* The Rodrigues formula used is::

        R = I + Vxmat + (Vxmat^2) * ((1 - c) / s^2)

  where ``Vxmat`` is the skew-symmetric matrix of the axis
  ``cross(target, source)`` (target FIRST, source AFTER), ``s`` is the
  ``sin`` (axis norm) and ``c`` the ``cos`` (dot).
* The opposite-vector branch (``x0 = -x1``) builds a diagonal reflection
  matrix whose negative entries sit where the two vectors differ in sign.
* Zero-length axis (``s == 0``) divides by zero and yields ``NaN`` -- the
  same as in MATLAB (no guard added, preserving ground-truth behaviour).
"""

from __future__ import annotations

import numpy as np

__all__ = ["RotMatrix4Vect"]

def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix ``Vxmat`` of a 3-vector ``v``."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )

def _rodrigues_branch(
    src: np.ndarray, tgt: np.ndarray, opposite: bool
) -> np.ndarray:
    """Port of the single (source, target) rotation sub-branch."""
    src = np.asarray(src, dtype=float).reshape(3)
    tgt = np.asarray(tgt, dtype=float).reshape(3)

    if opposite:
        # MATLAB: R = eye(3); flip the diagonal where (src .* tgt) < 0.
        R = np.eye(3)
        pos = np.where((src * tgt) < 0)[0]
        for jj in pos:
            R[int(jj), int(jj)] = -1.0
        return R

    v = np.cross(tgt, src)
    s = float(np.sum(v ** 2) ** 0.5)
    c = float(np.dot(src, tgt))
    Vmat = _skew(v)
    return np.eye(3) + Vmat + (Vmat @ Vmat) * ((1.0 - c) / (s ** 2))

def RotMatrix4Vect(
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
) -> np.ndarray:
    """Rotation matrix aligning the basis ``(x0, y0)`` to ``(x1, y1)``.

    Parameters
    ----------
    x0, y0
        Source orthonormal basis vectors, each a row vector of shape
        ``(1, 3)``.
    x1, y1
        Target orthonormal basis vectors, each a row vector of shape
        ``(1, 3)``.

    Returns
    -------
    np.ndarray
        A ``3 x 3`` rotation matrix ``R`` such that ``x0*R`` aligns with
        ``x1`` and ``y0*R`` aligns with ``y1`` (row-vector convention).
    """
    tol = 1e-3

    x0 = np.asarray(x0, dtype=float).reshape(1, 3)
    y0 = np.asarray(y0, dtype=float).reshape(1, 3)
    x1 = np.asarray(x1, dtype=float).reshape(1, 3)
    y1 = np.asarray(y1, dtype=float).reshape(1, 3)

    # --- first rotation: x0 -> x1 -----------------------------------------
    if np.sum(np.abs(x0 - x1) > tol) > 0:
        # x0 and x1 are different
        if np.sum(np.abs(np.abs(x0) - np.abs(x1)) > tol) > 0:
            # not just opposite (sign)
            R1 = _rodrigues_branch(x0, x1, opposite=False)
        else:
            # x0 = -x1
            R1 = _rodrigues_branch(x0, x1, opposite=True)
    else:
        R1 = np.eye(3)

    y0r = y0 @ R1

    # --- second rotation: y0r -> y1 ---------------------------------------
    if np.sum(np.abs(y0r - y1) > tol) > 0:
        # y0r and y1 are different
        if np.sum(np.abs(np.abs(y0r) - np.abs(y1)) > tol) > 0:
            # not just opposite (sign)
            R2 = _rodrigues_branch(y0r, y1, opposite=False)
        else:
            # y0r = -y1
            R2 = _rodrigues_branch(y0r, y1, opposite=True)
    else:
        R2 = np.eye(3)

    return R1 @ R2
