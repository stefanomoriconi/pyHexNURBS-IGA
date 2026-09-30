"""
hexiga.junc.get_extrude_bevel_qfs
=================================

For each
directional tube index, extrude the internal quad-face points along the tube
direction by ``extrFactor`` after applying a ``bvlFactor`` bevel, then
orthogonalise the resulting quad about its centre to a canonical basis
``(e1, e2, tg)``.  Also returns the per-tube extrusion direction
``edT`` (with optional wall/magnitude scaling).

Signature::

    [ebQFSfcs, ebQFSpts, KAe, idT, edT] = ...
        getExtrudeBevelQFS(dirTubes, QFSfcs, QFSpts, idx, extrFactor, bvlFactor, varargin)

Notes
-----

* Nested MATLAB functions ``getInputs``, ``extrbvlPts``, ``orthogQuad``,
  ``makeQuadConvex`` and ``extrDirTub`` are inlined here as private helpers.
* ``KAe`` (a MATLAB struct array) is returned as a list of dicts with keys
  ``pt, tg, e1, e2`` -- structurally identical for downstream use.
* ``pdist2`` is replaced by a per-column Euclidean distance (equivalent
  for the ``(3, n)`` quad case).
* Option parsing mirrors the MATLAB ``varargin`` ``switch`` (upper-cased
  keys); unrecognised keys print the literal message and apply defaults.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from .proj import proj
from .uvect import uvect
from .get_intersecting_point_for2_segments import getIntersectingPointFor2Segments

__all__ = ["getExtrudeBevelQFS"]

# ---------------------------------------------------------------------------
# Nested-helper ports
# ---------------------------------------------------------------------------

def _makeQuadConvex(Pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Port of the MATLAB ``makeQuadConvex`` local function (in-place convex fix)."""
    Pts = np.array(Pts, dtype=float, copy=True)
    M = np.asarray(M, dtype=float).ravel()

    inSeg1 = float(np.dot(Pts[:, 0] - M, Pts[:, 2] - M)) < 0
    inSeg2 = float(np.dot(Pts[:, 1] - M, Pts[:, 3] - M)) < 0

    if not inSeg1:
        if np.linalg.norm(Pts[:, 0] - M) < np.linalg.norm(Pts[:, 2] - M):
            Pts[:, 0] = M - uvect(Pts[:, 0] - M) * np.linalg.norm(Pts[:, 0] - M)
        else:
            Pts[:, 2] = M - uvect(Pts[:, 2] - M) * np.linalg.norm(Pts[:, 2] - M)
    if not inSeg2:
        if np.linalg.norm(Pts[:, 1] - M) < np.linalg.norm(Pts[:, 3] - M):
            Pts[:, 1] = M - uvect(Pts[:, 1] - M) * np.linalg.norm(Pts[:, 1] - M)
        else:
            Pts[:, 3] = M - uvect(Pts[:, 3] - M) * np.linalg.norm(Pts[:, 3] - M)
    return Pts

def _orthogQuad(Pts: np.ndarray, tg: np.ndarray):
    """Port of the MATLAB ``orthogQuad`` local function."""
    Pts = np.asarray(Pts, dtype=float)
    tg = np.asarray(tg, dtype=float).ravel()

    M = getIntersectingPointFor2Segments(Pts[:, 0], Pts[:, 2], Pts[:, 1], Pts[:, 3])

    Pts = _makeQuadConvex(Pts, M)

    e1ini = uvect(Pts[:, 0] - M)
    e2ini = uvect(Pts[:, 1] - M)

    e1fin = uvect(np.cross(tg, e2ini))
    e2fin = uvect(np.cross(tg, e1ini))

    e1fin = e1fin * np.sign(np.dot(e1fin, e1ini))
    e2fin = e2fin * np.sign(np.dot(e2fin, e2ini))

    e1 = uvect(np.mean(np.column_stack([e1ini, e1fin]), axis=1))
    e2 = uvect(np.mean(np.column_stack([e2ini, e2fin]), axis=1))

    # Per-column Euclidean distance from M (equivalent to pdist2(Pts', M')').
    dd = np.linalg.norm(Pts - M[:, None], axis=0)

    basis = np.column_stack([e1, e2, -e1, -e2])  # (3, 4)
    oPts = M[:, None] + dd[None, :] * basis
    return oPts, e1, e2, M

def _extrbvlPts(Pts: np.ndarray, extrDir: np.ndarray, extrFactor: float, bvlFactor: float):
    """Port of the MATLAB ``extrbvlPts`` local function."""
    Pts = np.asarray(Pts, dtype=float)
    extrDir = np.asarray(extrDir, dtype=float).ravel()

    ebPts = np.zeros_like(Pts)

    # Central offset
    # MATLAB: C = repmat(mean(Pts,2),[1,size(Pts,2)])
    # mean(Pts,2) = row-wise mean (3x1); numpy equivalent is axis=1.
    C = np.mean(Pts, axis=1, keepdims=True)

    # Remove the central offset
    cPts = Pts - C

    for jj in range(Pts.shape[1]):
        bPts = cPts[:, jj] * bvlFactor
        pPts = bPts - proj(bPts, extrDir) * extrDir
        ePts = pPts + extrFactor * extrDir
        ebPts[:, jj] = ePts

    eboPts, e1, e2, M = _orthogQuad(ebPts, extrDir)

    eboPts = eboPts + C
    # MATLAB: M = M + mean(Pts,2)  → row-wise mean → (3,) vector.
    M = M + np.mean(Pts, axis=1, keepdims=True).ravel()
    return eboPts, e1, e2, M

def _extrDirTub(ebQFSpt: np.ndarray, isFlat: bool, dirMag: float, isWall: bool) -> np.ndarray:
    """Port of the MATLAB ``extrDirTub`` local function.

    MATLAB: ``edT = mean(ebQFSpt, 2)`` reduces over the 4-point axis (columns),
    producing a (3,) direction vector.  In numpy that is ``axis=1``.
    """
    ebQFSpt = np.asarray(ebQFSpt, dtype=float)
    edT = np.mean(ebQFSpt, axis=1)
    if not isFlat:
        if isWall:
            edT = edT + (dirMag * 1.2 * uvect(edT))
        else:
            edT = edT + (dirMag * uvect(edT))
    return edT

def _getInputs(opts: Mapping[str, Any], n: int):
    """Port of the MATLAB ``getInputs`` local function.

    ``opts`` is a mapping from *upper-cased* option names to their values,
    or ``None`` / empty when no options are supplied.
    """
    O = {"isFlat": np.ones(n, dtype=bool), "dirMag": np.ones(n, dtype=float), "isWall": False}

    if not opts:
        return O

    for key, val in opts.items():
        k = str(key).upper()
        if k == "ISFLAT":
            arr = np.asarray(val)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.shape[0] == n:
                O["isFlat"] = arr.astype(bool)
            elif arr.shape[0] == 1:
                O["isFlat"] = (arr.astype(bool)[0] * np.ones(n)).astype(bool)
            else:
                print(" * getExtrudeBevelQFS: INCONSISTENT isFlat parameter: expected scalar or same size as idx - Default: \"true\" applied")
                O["isFlat"] = np.ones(n, dtype=bool)
        elif k == "DIRMAG":
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.shape[0] == n:
                O["dirMag"] = arr
            elif arr.shape[0] == 1:
                O["dirMag"] = arr[0] * np.ones(n)
            else:
                print(" * getExtrudeBevelQFS: INCONSISTENT dirMag parameter: expected scalar or same size as idx - Default: \"1\" applied")
                O["dirMag"] = np.ones(n)
        elif k == "WALL":
            O["isWall"] = bool(np.asarray(val).reshape(-1)[0])
        else:
            print(f" * getExtrudeBevelQFS: Unrecognised parsed parameter: {key} -- Default applied.")
    return O

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def getExtrudeBevelQFS(
    dirTubes: np.ndarray,
    QFSfcs: np.ndarray,
    QFSpts: np.ndarray,
    idx: np.ndarray,
    extrFactor,
    bvlFactor,
    **opts: Any,
):
    """Port of ``getExtrudeBevelQFS`` (see module docstring)."""
    dirTubes = np.asarray(dirTubes, dtype=float)
    QFSfcs = np.asarray(QFSfcs, dtype=int)
    QFSpts = np.asarray(QFSpts, dtype=float)
    idx = np.asarray(idx, dtype=int)

    assert idx.max() == idx.size, "Directional clusters must be SPLIT beforehand!"

    O = _getInputs(opts, idx.size)

    ef = np.atleast_1d(np.asarray(extrFactor, dtype=float)).ravel()
    if ef.size == 1:
        ef = ef[0] * np.ones(idx.size)
    else:
        assert ef.size == idx.size, " * getExtrudeBevelQFS: extrFactor MUST be scalar or [1 x numTubes]!"

    bf = np.atleast_1d(np.asarray(bvlFactor, dtype=float)).ravel()
    if bf.size == 1:
        bf = bf[0] * np.ones(idx.size)
    else:
        assert bf.size == idx.size, " * getExtrudeBevelQFS: bvlFactor MUST be scalar or [1 x numTubes]!"

    ebQFSpts: Optional[np.ndarray] = None
    ebQFSfcs = np.zeros_like(QFSfcs)
    ebQFSfcsKernel = np.array([1, 2, 3, 4])  # 1-based, literal MATLAB convention

    KAe: List[dict] = []
    idT = np.zeros((3, dirTubes.shape[1]), dtype=float)
    edT = np.zeros((3, dirTubes.shape[1]), dtype=float)

    for jj in range(dirTubes.shape[1]):
        QFSfc = QFSfcs[:, idx[jj] - 1]  # idx is 1-based (MATLAB convention)
        QFSpt = QFSpts[:, QFSfc - 1]    # face values are 1-based -> 0-based for numpy
        extrDir = dirTubes[:, jj]

        ebQFSpt, e1, e2, pt = _extrbvlPts(QFSpt, extrDir, ef[idx[jj] - 1], bf[idx[jj] - 1])

        edT[:, jj] = _extrDirTub(ebQFSpt, bool(O["isFlat"][idx[jj] - 1]), float(O["dirMag"][idx[jj] - 1]), bool(O["isWall"]))

        KAe.append({"pt": pt, "tg": dirTubes[:, jj], "e1": e1, "e2": e2})

        ebQFSpts = ebQFSpt if ebQFSpts is None else np.hstack([ebQFSpts, ebQFSpt])

        ebQFSfc = ((jj) * ebQFSfcsKernel.max()) + ebQFSfcsKernel
        ebQFSfcs[:, idx[jj] - 1] = ebQFSfc

    return ebQFSfcs, ebQFSpts, KAe, idT, edT
