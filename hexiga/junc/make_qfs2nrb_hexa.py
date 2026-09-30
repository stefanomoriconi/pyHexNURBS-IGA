"""
hexiga.junc.make_qfs2nrb_hexa
=============================

Build one linear
hexahedral NURBS patch per internal quad-face edge (``intQFSpt``), extruded
toward the external quad-face edge (``extQFSpt``), with the internal/external
junction midpoints ``intQpt`` / ``extEpt``.

Signature::

    H = makeQFSs2nrbHexa(QFSfcs, QFSpts, Qpts, ebQFSfcs, ebQFSpts, Epts, idx, varargin)

Notes
-----

* Returns a *flat* list of :class:`~hexiga.nurbs.nrb.Nrb` hexahedra (MATLAB
  stores them in a struct array indexed by ``idx``; the port linearises the
  per-tube groups in order, matching ``H(idx(jj)).hexa = [selH(:).hexa]``).
* Reuses :func:`hexiga.scaff.getHexaCoeffsKnots` (already ported) for the
  control-point / knot assembly, and :func:`hexiga.nurbs.nrbmak` to build
  each patch.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.make import nrbmak
from ..scaff.get_hexa_coeffs_knots import getHexaCoeffsKnots
from .pad_qfspt import padQFSpt
from .split_qfspt import splitQFSpt

__all__ = ["makeQFSs2nrbHexa"]

def _nrbHexa(intQFSpt, intQpt, extQFSpt, extEpt) -> List[Nrb]:
    """Port of the MATLAB ``nrbHexa`` local function (returns a flat list)."""
    intQFSpt = np.asarray(intQFSpt, dtype=float)
    extQFSpt = np.asarray(extQFSpt, dtype=float)

    pad_intQFSpt = padQFSpt(intQFSpt, 1)
    intQFSpt_split = splitQFSpt(intQFSpt, intQpt)

    pad_extQFspt = padQFSpt(extQFSpt, 1)
    extQFSpt_split = splitQFSpt(extQFSpt, extEpt)

    out: List[Nrb] = []
    n = intQFSpt.shape[1]
    for jj in range(n):
        iA = pad_intQFSpt[:, jj]
        iB = intQFSpt_split[:, jj]
        iC = intQpt
        iD = pad_intQFSpt[:, jj + 1]

        eA = pad_extQFspt[:, jj]
        eB = extQFSpt_split[:, jj]
        eC = extEpt
        eD = pad_extQFspt[:, jj + 1]

        Hcfs, Hknt = getHexaCoeffsKnots(iA, iB, iC, iD, eA, eB, eC, eD)
        selHexa = nrbmak(Hcfs, Hknt)
        out.append(selHexa)
    return out

def makeQFSs2nrbHexa(QFSfcs, QFSpts, Qpts, ebQFSfcs, ebQFSpts, Epts, idx, *varargs) -> List[Nrb]:
    """Port of ``makeQFSs2nrbHexa`` (see module docstring)."""
    QFSfcs = np.asarray(QFSfcs, dtype=int)
    QFSpts = np.asarray(QFSpts, dtype=float)
    Qpts = np.asarray(Qpts, dtype=float)
    ebQFSfcs = np.asarray(ebQFSfcs, dtype=int)
    ebQFSpts = np.asarray(ebQFSpts, dtype=float)
    Epts = np.asarray(Epts, dtype=float)
    idx = np.asarray(idx, dtype=int)

    assert idx.max() == idx.size, "Indices are not UNIQUE!"
    assert QFSfcs.shape[1] == idx.size, "Size mismatch between QFSfcs and indixes (idx)!"
    assert Qpts.shape[1] == idx.size, "Size mismatch between Qpts and indixes (idx)!"
    assert ebQFSfcs.shape[1] == idx.size, "Size mismatch between ebQFSfcs and indixes (idx)!"
    assert Epts.shape[1] == idx.size, "Size mismatch between Epts and indixes (idx)!"

    H: List[Nrb] = []
    for jj in range(idx.size):
        intQFSfc = QFSfcs[:, idx[jj] - 1] - 1  # idx is 1-based; face indices 1-based -> 0-based
        intQFSpt = QFSpts[:, intQFSfc]
        intQpt = Qpts[:, jj]

        extQFSfc = ebQFSfcs[:, idx[jj] - 1] - 1  # face values are 1-based -> 0-based
        extQFSpt = ebQFSpts[:, extQFSfc]
        extEpt = Epts[:, jj]

        selH = _nrbHexa(intQFSpt, intQpt, extQFSpt, extEpt)
        H.extend(selH)
    return H
