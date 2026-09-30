"""
hexiga.junc.make_qfs2nrb_wall_hexa
=================================

Build the
"wall" hexahedra that sweep a tube cross-section (``intTpt`` -> ``extTpt``)
between its internal and external quad-faces, and -- for terminal tubes --
four extra capping hexahedra between ``iCpt`` / ``eCpt``.

Signature::

    [Hw, CapFlag] = makeQFSs2nrbWallHexa(intTfcs, intTpts, intQfcs, intQpts, Qpts,
                       extTfcs, extTpts, extQfcs, extQpts, Epts, Cpts, idx, ioFlags)

Notes
-----

* Returns ``(Hw, CapFlag)``: a flat list of :class:`~hexiga.nurbs.nrb.Nrb`
  patches and a matching flat list of booleans (``True`` for cap patches).
  The MATLAB per-tube struct grouping (``Hw(idx(jj)).hexa = [selH(:).hexa]``)
  is linearised in order.
* Reuses :func:`hexiga.scaff.getHexaCoeffsKnots` and
  :func:`hexiga.nurbs.nrbmak`.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.make import nrbmak
from ..scaff.get_hexa_coeffs_knots import getHexaCoeffsKnots
from .pad_qfspt import padQFSpt
from .split_qfspt import splitQFSpt

__all__ = ["makeQFSs2nrbWallHexa"]

def _interleave(pts: np.ndarray, split: np.ndarray) -> np.ndarray:
    """Build ``seq(:,1:2:2n)=pts; seq(:,2:2:2n)=split`` (0-based analogue)."""
    n = pts.shape[1]
    seq = np.empty((3, 2 * n), dtype=float)
    seq[:, 0::2] = pts
    seq[:, 1::2] = split
    return seq

def _nrbHexaWall(intTpt, intQpt, extTpt, extQpt, Qpt, Ept) -> List[Nrb]:
    """Port of the MATLAB ``nrbHexaWall`` local function (returns a flat list)."""
    intTpt = np.asarray(intTpt, dtype=float)
    extTpt = np.asarray(extTpt, dtype=float)
    intQpt = np.asarray(intQpt, dtype=float)
    extQpt = np.asarray(extQpt, dtype=float)

    intTpt_seq = padQFSpt(_interleave(intTpt, splitQFSpt(intTpt, Ept)), 1)
    intQpt_seq = padQFSpt(_interleave(intQpt, splitQFSpt(intQpt, Qpt)), 1)
    extTpt_seq = padQFSpt(_interleave(extTpt, splitQFSpt(extTpt, Ept)), 1)
    extQpt_seq = padQFSpt(_interleave(extQpt, splitQFSpt(extQpt, Qpt)), 1)

    out: List[Nrb] = []
    nseq = intTpt_seq.shape[1]
    for jj in range(nseq - 1):
        iA = intQpt_seq[:, jj]
        iB = intQpt_seq[:, jj + 1]
        iC = extQpt_seq[:, jj]
        iD = extQpt_seq[:, jj + 1]

        eA = intTpt_seq[:, jj]
        eB = intTpt_seq[:, jj + 1]
        eC = extTpt_seq[:, jj]
        eD = extTpt_seq[:, jj + 1]

        Hcfs, Hknt = getHexaCoeffsKnots(iA, iB, iC, iD, eA, eB, eC, eD)
        out.append(nrbmak(Hcfs, Hknt))
    return out

def _nrbHexaCap(intTpt, extTpt, Ept, iCpt, eCpt) -> List[Nrb]:
    """Port of the MATLAB ``nrbHexaCap`` local function (returns a flat list)."""
    intTpt = np.asarray(intTpt, dtype=float)
    extTpt = np.asarray(extTpt, dtype=float)
    iCpt = np.asarray(iCpt, dtype=float)
    eCpt = np.asarray(eCpt, dtype=float)

    intTpt_seq = padQFSpt(_interleave(intTpt, splitQFSpt(intTpt, Ept)), 1)
    extTpt_seq = padQFSpt(_interleave(extTpt, splitQFSpt(extTpt, Ept)), 1)

    out: List[Nrb] = []
    nseq = intTpt_seq.shape[1]
    # MATLAB: for jj = 1 : 2 : size(intTpt_seq,2)-1  with `jj` 1-BASED.
    # `jj` therefore takes the *values* {1, 3, ..., 2n-1}; MATLAB then reads
    # the 1-based columns jj, jj+1, jj+2.  The 0-based equivalents of those
    # column references are (jj-1), jj, (jj+1) respectively.  Reproducing that
    # here (a uniform +1 shift, NOT +2, is the faithful mapping):
    for jj in range(1, nseq, 2):
        iA = intTpt_seq[:, jj]
        iB = intTpt_seq[:, jj - 1]
        iC = intTpt_seq[:, jj + 1]
        iD = iCpt

        eA = extTpt_seq[:, jj]
        eB = extTpt_seq[:, jj - 1]
        eC = extTpt_seq[:, jj + 1]
        eD = eCpt

        Ccfs, Cknt = getHexaCoeffsKnots(iA, iB, iC, iD, eA, eB, eC, eD)
        out.append(nrbmak(Ccfs, Cknt))
    return out

def makeQFSs2nrbWallHexa(
    intTfcs, intTpts, intQfcs, intQpts, Qpts,
    extTfcs, extTpts, extQfcs, extQpts, Epts, Cpts, idx, ioFlags,
) -> Tuple[List[Nrb], List[bool]]:
    """Port of ``makeQFSs2nrbWallHexa`` (see module docstring)."""
    intTfcs = np.asarray(intTfcs, dtype=int)
    intTpts = np.asarray(intTpts, dtype=float)
    intQfcs = np.asarray(intQfcs, dtype=int)
    intQpts = np.asarray(intQpts, dtype=float)
    Qpts = np.asarray(Qpts, dtype=float)
    extTfcs = np.asarray(extTfcs, dtype=int)
    extTpts = np.asarray(extTpts, dtype=float)
    extQfcs = np.asarray(extQfcs, dtype=int)
    extQpts = np.asarray(extQpts, dtype=float)
    Epts = np.asarray(Epts, dtype=float)
    Cpts = np.asarray(Cpts, dtype=float)
    idx = np.asarray(idx, dtype=int)
    ioFlags = np.asarray(ioFlags, dtype=bool)

    Hw: List[Nrb] = []
    CapFlag: List[bool] = []

    for jj in range(idx.size):
        # MATLAB: intTfc = intTfcs(:, idx(jj)); intTpt = intTpts(:, intTfc);
        # idx is 1-based → -1 for column; fc values are 1-based → -1 for indexing
        intTfc = intTfcs[:, idx[jj] - 1] - 1
        intTpt = intTpts[:, intTfc]
        intQfc = intQfcs[:, idx[jj] - 1] - 1
        intQpt = intQpts[:, intQfc]

        extTfc = extTfcs[:, idx[jj] - 1] - 1
        extTpt = extTpts[:, extTfc]
        extQfc = extQfcs[:, idx[jj] - 1] - 1
        extQpt = extQpts[:, extQfc]

        Qpt = Qpts[:, jj]

        if ioFlags[idx[jj] - 1]:  # NOT terminal tube (MATLAB: ioFlags(idx(jj)))
            Ept = Epts[:, jj]
            selH = _nrbHexaWall(intTpt, intQpt, extTpt, extQpt, Qpt, Ept)
            Hw.extend(selH)
            CapFlag.extend([False] * len(selH))
        else:  # terminal tube
            Ept = Epts[:, jj]
            selH = _nrbHexaWall(intTpt, intQpt, extTpt, extQpt, Qpt, Ept)
            iCpt = Epts[:, jj]
            eCpt = Cpts[:, jj]
            selC = _nrbHexaCap(intTpt, extTpt, Ept, iCpt, eCpt)
            Hw.extend(selH)
            Hw.extend(selC)
            CapFlag.extend([False] * len(selH))
            CapFlag.extend([True] * len(selC))

    return Hw, CapFlag
