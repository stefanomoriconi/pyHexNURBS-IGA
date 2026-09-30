"""
hexiga.smth.append_smth_atrbs_new
=================================

-- the neighbour/label attachment
pass that decorates each aggregated control point (``mpCPT``) with its
neighbour id list (``nnIDs``), neighbour labels (``nnLbls``) and the
smoothing flag (``isSmth``).

All ``ptIDs`` are 1-based indices into the ``mpCPT`` list, exactly as in
MATLAB.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .cpt import CPT, sub2ind

__all__ = ["append_smth_atrbs_new"]

# ---------------------------------------------------------------------------
# Point-selection helper
# ---------------------------------------------------------------------------
def _get_pt_ids_to_smth(mpCPT: List[CPT], selShells, freezeShellBoundary: bool) -> np.ndarray:
    """Port of ``getptIDsToSmth`` (returns 1-based point ids)."""
    ptFrm = np.array([c.ptFrm for c in mpCPT])
    isShlB = np.array([c.isShlB for c in mpCPT])
    ptShl = np.array([c.ptShl for c in mpCPT])

    selShells = np.atleast_1d(selShells) if selShells is not None and len(selShells) else np.array([])
    if np.size(selShells) == 0:
        if freezeShellBoundary:
            chk = (ptFrm != 0) & (~isShlB)
        else:
            chk = ptFrm != 0
    else:
        if np.all(np.isfinite(selShells)):
            ismember = np.isin(ptShl, selShells)
        else:  # selShells == inf
            ismember = np.isin(ptShl, [np.max(ptShl)])
        if freezeShellBoundary:
            chk = ismember & (ptFrm != 0) & (~isShlB)
        else:
            chk = ismember & (ptFrm != 0)
    return np.nonzero(chk)[0] + 1

# ---------------------------------------------------------------------------
# Simplex neighbours
# ---------------------------------------------------------------------------
def _get_simplex6() -> np.ndarray:
    """3 x 6: (du,dv,dw) face-neighbour offsets."""
    return np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=int,
    ).T

def _get_simplex12() -> np.ndarray:
    """3 x 12: (du,dv,dw) edge-neighbour offsets."""
    return np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, 0, -1],
            [0, 1, -1],
            [-1, 0, -1],
            [0, -1, -1],
        ],
        dtype=int,
    ).T

def _get_neighbors(dim, idx: int, simplex: np.ndarray):
    """Port of ``getNeighbors``.

    ``dim`` = 1-based size tuple, ``idx`` = 1-based linear index.  Returns
    ``(nnIDs, nnLbl)`` where ``nnLbl`` is a 3x3x3 array with the 1-based
    ``sub2ind`` of a neighbour stored at internal offset position
    ``(du+1, dv+1, dw+1)`` and 0 elsewhere.
    """
    nnIDs: List[int] = []
    nnLbl = np.zeros((3, 3, 3), dtype=int)
    u, v, w = np.unravel_index(idx - 1, dim, order="F")
    u += 1
    v += 1
    w += 1
    for ss in range(simplex.shape[1]):
        nnU = u + simplex[0, ss]
        nnV = v + simplex[1, ss]
        nnW = w + simplex[2, ss]
        if 1 <= nnU <= dim[0] and 1 <= nnV <= dim[1] and 1 <= nnW <= dim[2]:
            lid = sub2ind(dim, nnU, nnV, nnW)
            nnIDs.append(lid)
            nnLbl[simplex[0, ss] + 1, simplex[1, ss] + 1, simplex[2, ss] + 1] = lid
    nnIDs = np.array(nnIDs, dtype=int)
    return nnIDs, nnLbl

# ---------------------------------------------------------------------------
# 12-<->6 tensor LUT
# ---------------------------------------------------------------------------
def _get_neighbors12_tensor6_lut() -> np.ndarray:
    """Port of ``getNeighbors12Tensor6LUT`` (3 x 12, verbatim)."""
    return np.array(
        [
            [2, 4, 6, 8, 10, 12, 16, 18, 20, 22, 24, 26],
            [11, 13, 15, 17, 13, 15, 13, 15, 11, 13, 15, 17],
            [5, 5, 5, 5, 11, 11, 17, 17, 23, 23, 23, 23],
        ],
        dtype=int,
    )

def _nn12is_tensor6(nn12idx: int, nn6Lbl: np.ndarray) -> bool:
    """Port of ``nn12isTensor6``.

    ``nn6Lbl`` is a 3 x 3 x 3 label array in MATLAB; the LUT's 2nd and 3rd
    rows hold 1-based column-major linear indices into it.  We resolve each
    index to (u,v,w) subscripts via F-order ``unravel_index``.
    """
    LUT = _get_neighbors12_tensor6_lut()
    idx = np.nonzero(LUT[0, :] == nn12idx)[0]
    if idx.size == 0:
        return False
    shape = nn6Lbl.shape
    i1 = int(LUT[1, idx[0]]) - 1
    i2 = int(LUT[2, idx[0]]) - 1
    u1, v1, w1 = np.unravel_index(i1, shape, order="F")
    u2, v2, w2 = np.unravel_index(i2, shape, order="F")
    return bool(nn6Lbl[u1, v1, w1] > 0 and nn6Lbl[u2, v2, w2] > 0)

# ---------------------------------------------------------------------------
# Shell neighbour enumeration (single patch)
# ---------------------------------------------------------------------------
def _get_shl_nn_ids(UVWsize, ptIdx3, ShlVal, mpCPT, ptcID, ptcIDsList):
    """Port of ``getShlNNIDs``.  Returns ``(NNptIDs, NNEDist, NNShlB)``."""
    nn6IDs, nn6Lbl = _get_neighbors(UVWsize, ptIdx3, _get_simplex6())
    nn12IDs, nn12Lbl = _get_neighbors(UVWsize, ptIdx3, _get_simplex12())

    NN6IDs = np.full(nn6IDs.shape, np.nan)
    NN6ShlB = np.zeros(nn6IDs.shape, dtype=bool)
    NN12IDs = np.full(nn12IDs.shape, np.nan)
    NN12ShlB = np.zeros(nn12IDs.shape, dtype=bool)

    # points belonging to this patch
    sel_pt_ids = np.array(
        [e.ptIDs for e in ptcIDsList if e.ptcID == ptcID], dtype=object
    )
    if len(sel_pt_ids) == 0:
        sel_pt_ids = np.zeros(0, dtype=int)
    else:
        sel_pt_ids = np.concatenate([np.atleast_1d(p) for p in sel_pt_ids])

    # Fast-path membership sets (Python ``int``).  The two ``np.isin(...).any()``
    # gates below are set-membership tests over small integer arrays; using
    # hash sets is bit-identical to ``np.isin`` for integer membership (the
    # values are compared as exact integers) while removing the per-point
    # ``np.isin`` overhead.  This is a faithful, accuracy-preserving
    # optimisation of the same predicate -- no change to the accepted
    # neighbour / shell bookkeeping.
    _nn6_set = {int(x) for x in nn6IDs}
    _nn12_set = {int(x) for x in nn12IDs}

    for pt in sel_pt_ids:
        PTCmsk = mpCPT[int(pt) - 1].ptcIDs == ptcID
        _ptIdx = mpCPT[int(pt) - 1].ptIdx3[PTCmsk]
        # 6-neighbours
        if any(int(x) in _nn6_set for x in _ptIdx):
            if mpCPT[int(pt) - 1].ptShl == ShlVal:
                PT6msk = nn6IDs == _ptIdx[0]
                NN6IDs[PT6msk] = pt
                NN6ShlB[PT6msk] = mpCPT[int(pt) - 1].isShlB
        # 12-neighbours
        if any(int(x) in _nn12_set for x in _ptIdx):
            if mpCPT[int(pt) - 1].ptShl == ShlVal:
                PT12msk = nn12IDs == _ptIdx[0]
                NN12IDs[PT12msk] = pt
                NN12ShlB[PT12msk] = mpCPT[int(pt) - 1].isShlB

    NNptIDs: List[int] = []
    NNEDist: List[float] = []
    NNShlB: List[bool] = []
    for nn6 in range(len(nn6IDs)):
        if not np.isnan(NN6IDs[nn6]):
            NNptIDs.append(int(NN6IDs[nn6]))
            NNEDist.append(1.0)
            NNShlB.append(bool(NN6ShlB[nn6]))
        else:
            nn6Lbl[nn6Lbl == nn6IDs[nn6]] = 0

    for nn12 in range(len(nn12IDs)):
        if not np.isnan(NN12IDs[nn12]):
            # MATLAB: ``nn12idx = find(nn12Lbl == nn12IDs(nn12));`` is a
            # column-major (F-order) linear index into the 3 x 3 x 3 label
            # array; ``nn12isTensor6`` expects exactly that.  ``np.nonzero``
            # on a C-contiguous array returns C-order linear indices, so we
            # re-index in F-order to match MATLAB.
            nn12idx = np.flatnonzero(nn12Lbl.ravel(order="F") == nn12IDs[nn12])
            if nn12idx.size and _nn12is_tensor6(int(nn12idx[0]) + 1, nn6Lbl):
                NNptIDs.append(int(NN12IDs[nn12]))
                NNEDist.append(np.sqrt(2))
                NNShlB.append(bool(NN12ShlB[nn12]))

    return np.array(NNptIDs, dtype=int), np.array(NNEDist, dtype=float), np.array(NNShlB, dtype=bool)

def _get_shl_nn_ids_medial(UVWsize, ptIdx3, mpCPT, ptcID, ptcIDsList):
    """Port of ``getShlNNIDsMedialLocus``.  Returns ``NNptIDs`` (1-based)."""
    nn6IDs, _ = _get_neighbors(UVWsize, ptIdx3, _get_simplex6())
    NN6IDs = np.full(nn6IDs.shape, np.nan)

    sel_pt_ids = np.array(
        [e.ptIDs for e in ptcIDsList if e.ptcID == ptcID], dtype=object
    )
    if len(sel_pt_ids) == 0:
        sel_pt_ids = np.zeros(0, dtype=int)
    else:
        sel_pt_ids = np.concatenate([np.atleast_1d(p) for p in sel_pt_ids])

    for pt in sel_pt_ids:
        PTCmsk = mpCPT[int(pt) - 1].ptcIDs == ptcID
        if np.isin(mpCPT[int(pt) - 1].ptIdx3[PTCmsk], nn6IDs).any():
            PT6msk = nn6IDs == mpCPT[int(pt) - 1].ptIdx3[PTCmsk][0]
            NN6IDs[PT6msk] = pt
    return NN6IDs[~np.isnan(NN6IDs)].astype(int)

# ---------------------------------------------------------------------------
# Unique neighbour gathering (multi-patch)
# ---------------------------------------------------------------------------
def _get_mp_unique_neighbors_generic(mpCPT, mpCPTatrb, ptID, ptcIDsList):
    """Port of ``getMPUniqueNeighborsGenericPt``."""
    c = mpCPT[ptID - 1]
    NNptIDs_all = np.zeros(0, dtype=int)
    NNEDist_all = np.zeros(0)
    NNShlB_all = np.zeros(0, dtype=bool)
    for ptc in c.ptcIDs:
        ptc = int(ptc)
        number = tuple(int(x) for x in mpCPTatrb[ptc - 1].number)
        ptIdx3 = int(np.asarray(c.ptIdx3)[np.nonzero(c.ptcIDs == ptc)[0][0]])
        nn = _get_shl_nn_ids(number, ptIdx3, c.ptShl, mpCPT, ptc, ptcIDsList)
        NNptIDs_all = np.concatenate([NNptIDs_all, nn[0]])
        NNEDist_all = np.concatenate([NNEDist_all, nn[1]])
        NNShlB_all = np.concatenate([NNShlB_all, nn[2]])

    if c.n_patches > 1:
        NNptIDs, uniq_idx = np.unique(NNptIDs_all, return_inverse=True)
        first = np.full(NNptIDs.shape, -1, dtype=int)
        for k in range(NNptIDs_all.shape[0]):
            i = int(uniq_idx[k])
            if first[i] < 0:
                first[i] = k
        NNEDist_out = NNEDist_all[first]
        NNShlB_out = NNShlB_all[first]
    else:
        assert np.all(np.diff(np.sort(NNptIDs_all)) > 0), "duplicated NN in single-patch!"
        order = np.argsort(NNptIDs_all)
        NNptIDs = NNptIDs_all[order]
        NNEDist_out = NNEDist_all[order]
        NNShlB_out = NNShlB_all[order]
    return NNptIDs, NNEDist_out, NNShlB_out

def _get_mp_unique_neighbors_medial(mpCPT, mpCPTatrb, ptID, ptcIDsList):
    """Port of ``getMPUniqueNeighborsMedialLocus``."""
    c = mpCPT[ptID - 1]
    NNptIDs_all = np.zeros(0, dtype=int)
    for ptc in c.ptcIDs:
        ptc = int(ptc)
        number = tuple(int(x) for x in mpCPTatrb[ptc - 1].number)
        ptIdx3 = int(np.asarray(c.ptIdx3)[np.nonzero(c.ptcIDs == ptc)[0][0]])
        nn = _get_shl_nn_ids_medial(number, ptIdx3, mpCPT, ptc, ptcIDsList)
        NNptIDs_all = np.concatenate([NNptIDs_all, nn])
    if c.n_patches > 1:
        NNptIDs = np.unique(NNptIDs_all)
    else:
        assert np.all(np.diff(np.sort(NNptIDs_all)) > 0), "duplicated NN in single-patch!"
        NNptIDs = np.sort(NNptIDs_all)
    NNEDist = np.ones(NNptIDs.shape)
    NNShlB = np.zeros(NNptIDs.shape, dtype=bool)
    return NNptIDs, NNEDist, NNShlB

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def _get_relative_pt_label(dim, idx: int) -> int:
    """Port of ``getRelativePtLabel``.  dim = 1-based size, idx = 1-based."""
    u, v, w = np.unravel_index(idx - 1, dim, order="F")
    u += 1
    v += 1
    w += 1
    uBoundary = (u == 1) or (u == dim[0])
    vBoundary = (v == 1) or (v == dim[1])
    wBoundary = (w == 1) or (w == dim[2])
    BoundaryValence = int(uBoundary) + int(vBoundary) + int(wBoundary)
    if BoundaryValence == 3:
        return -1
    if BoundaryValence == 2:
        return 0
    return 1

def _get_relative_pts_labels(mpCPT, NNptIDs, ptID, verboseFlag):
    """Port of ``getRelativePtsLabels``."""
    NNLblsSP = np.zeros(NNptIDs.shape)
    for nn in NNptIDs:
        nn = int(nn)
        nn_lbls = []
        c = mpCPT[nn - 1]
        for ptc in c.ptcIDs:
            k = int(np.nonzero(c.ptcIDs == ptc)[0][0])
            NNpatchSize = c.dimShlR[k, :]
            NNptIdx3 = int(c.ptIdx3R[k])
            nn_lbls.append(_get_relative_pt_label(NNpatchSize, NNptIdx3))
        nn_lbls = np.unique(np.array(nn_lbls))
        if nn_lbls.size == 1:
            NNLblsSP[np.nonzero(NNptIDs == nn)[0][0]] = nn_lbls[0]
        else:
            if verboseFlag:
                print(" * [WRN] getRelativePtsLabels: non-unique NNptLbls found! - Please Check for BUGS!")

    # point itself
    c = mpCPT[ptID - 1]
    lbls = []
    for ptc in c.ptcIDs:
        k = int(np.nonzero(c.ptcIDs == ptc)[0][0])
        lbls.append(_get_relative_pt_label(c.dimShlR[k, :], int(c.ptIdx3R[k])))
    lbls = np.unique(np.array(lbls))
    if lbls.size == 1:
        ptLblSP = int(lbls[0])
    else:
        ptLblSP = -1
        if verboseFlag:
            print(" * [WRN] getRelativePtsLabels: non-unique ptLblSP found! - Please Check for BUGS!")
    return NNLblsSP, ptLblSP

def _get_mp_sp_pts_labels_generic(mpCPT, ptID, NNEDist, NNptIDs, verboseFlag):
    """Port of ``getMPSPptsLabelsGenericPt``.  Returns (NNShlLblsMP, NNLblsSP, ptShlLblMP, ptLblSP)."""
    NNShlLblsMP = NNEDist > 1
    # getShellMPptsLabels
    s_lab = int(np.sum(NNShlLblsMP))
    s_not = int(np.sum(~NNShlLblsMP))
    if s_lab > s_not:
        ptShlLblMP = 1
    elif s_lab == s_not and s_lab > 1:
        ptShlLblMP = 1
    else:
        if s_lab == 1:
            ptShlLblMP = -1
        else:
            ptShlLblMP = 0
    NNLblsSP, ptLblSP = _get_relative_pts_labels(mpCPT, NNptIDs, ptID, verboseFlag)
    return NNShlLblsMP, NNLblsSP, ptShlLblMP, ptLblSP

def _get_mp_sp_pts_labels_medial(NNptIDs, isShlB):
    """Port of ``getMPSPptsLabelsMedialLocus``."""
    NNShlLblsMP = np.zeros(NNptIDs.shape, dtype=bool)
    NNLblsSP = np.ones(NNptIDs.shape)
    if not isShlB:
        ptShlLblMP = 1
        ptLblSP = -1
    else:
        ptShlLblMP = 0
        ptLblSP = -1
    return NNShlLblsMP, NNLblsSP, ptShlLblMP, ptLblSP

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def append_smth_atrbs_new(mpCPT: List[CPT], mpCPTatrb, *pairs):
    """Port of ``appendSmthAtrbs_new``.  Mutates and returns ``mpCPT``.

    ``mpCPTatrb`` is the per-patch attribute list (supplies each patch's
    ``number`` used as the neighbour-enumeration grid size).  ``pairs`` are
    the MATLAB name/value args: ``'SHELLS', [...],
    'FREEZESHELLBOUNDARY', bool`` (plus optional ``'VERBOSE'``).
    """
    OPTs: dict = {}
    it = iter(pairs)
    for key in it:
        OPTs[str(key).upper()] = next(it)
    selShells = OPTs.get("SHELLS", [])
    freezeShellBoundary = bool(OPTs.get("FREEZESHELLBOUNDARY", False))
    verboseFlag = bool(OPTs.get("VERBOSE", False))

    from .aggregate import get_list_ptc_ids

    ptSmthIDs = _get_pt_ids_to_smth(mpCPT, selShells, freezeShellBoundary)
    ptcIDsList = get_list_ptc_ids(mpCPT, ptSmthIDs)

    for pt in ptSmthIDs:
        pt = int(pt)
        c = mpCPT[pt - 1]
        if c.ptShl > 0 and c.ptFrm > 0:
            NNptIDs, NNEDist, NNShlB = _get_mp_unique_neighbors_generic(mpCPT, mpCPTatrb, pt, ptcIDsList)
        elif c.ptShl == 0 or c.ptFrm < 0:
            NNptIDs, NNEDist, NNShlB = _get_mp_unique_neighbors_medial(mpCPT, mpCPTatrb, pt, ptcIDsList)
        else:
            NNptIDs, NNEDist, NNShlB = np.zeros(0, dtype=int), np.zeros(0), np.zeros(0, dtype=bool)

        if c.ptShl > 0 and c.ptFrm > 0:
            NNShlLblsMP, NNLblsSP, ptShlLblMP, ptLblSP = _get_mp_sp_pts_labels_generic(
                mpCPT, pt, NNEDist, NNptIDs, verboseFlag
            )
        elif c.ptShl == 0:
            NNShlLblsMP, NNLblsSP, ptShlLblMP, ptLblSP = _get_mp_sp_pts_labels_medial(
                NNptIDs, c.isShlB
            )
        else:
            NNShlLblsMP, NNLblsSP, ptShlLblMP, ptLblSP = (
                np.zeros(0, dtype=bool), np.zeros(0), 0, 0
            )

        if ptShlLblMP == 1:  # CORE
            if ptLblSP == 0:
                sel = ~NNShlLblsMP & (NNLblsSP == 1)
                nnIDs = NNptIDs[sel]
                nnLbls = NNShlLblsMP[sel]
                isSmth = False
            elif ptLblSP == -1:
                nnIDs = NNptIDs
                nnLbls = NNShlLblsMP
                isSmth = False
            else:
                nnIDs, nnLbls, isSmth = c.nnIDs, c.nnLbls, c.isSmth
        elif ptShlLblMP == 0:  # SKIRT
            if c.ptShl == 0:
                nnIDs, nnLbls, isSmth = np.zeros(0, dtype=int), np.zeros(0), True
            elif ptLblSP != 1:
                sel = NNShlB & (NNLblsSP != -1)
                nnIDs = NNptIDs[sel]
                nnLbls = NNShlLblsMP[sel]
                isSmth = False
            else:
                nnIDs, nnLbls, isSmth = c.nnIDs, c.nnLbls, c.isSmth
        else:  # CORNER -> identity
            nnIDs, nnLbls, isSmth = c.nnIDs, c.nnLbls, c.isSmth

        c.nnIDs = np.asarray(nnIDs, dtype=int)
        c.nnLbls = np.asarray(nnLbls, dtype=bool)
        c.isSmth = bool(isSmth)

    return mpCPT
