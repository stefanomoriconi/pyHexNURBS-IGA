"""
hexiga.smth.aggregate
=====================

  Aggregates the per-patch control points of a multi-patch
hexahedral into a single de-duplicated topology list ``mpCPT`` together with a
per-patch attribute list ``mpCPTatrb``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..nurbs.nrb import Nrb
from .cpt import CPT, CPTAttr, Cuboid, ind2sub, sub2ind  # noqa: F401  (sub2ind used below)
from .find_ctrl_pt import find_ctrl_pt

__all__ = ["aggregate_multi_patch_hexa", "get_list_ptc_ids"]

# ---------------------------------------------------------------------------
# Field-value computation for one control point of one patch
# ---------------------------------------------------------------------------
def _get_relative_grid_size_and_pt_idx(Grid: np.ndarray, idx_1b: int):
    """Port of ``getRelativeGridSizeAndPtIdxForShellPt`` (1-based ``idx``).

    MATLAB uses column-major linear indexing on the 3-D grid, so all linear
    reads/writes here go through ``order="F"`` flat views (reads) or
    ``np.unravel_index(..., order="F")`` (subscripts).
    """
    flat = Grid.ravel(order="F")              # column-major linear (read)
    shell_val = flat[idx_1b - 1]
    sel = np.flatnonzero(flat == shell_val)   # column-major linear indices
    u, v, w = np.unravel_index(sel, Grid.shape, order="F")
    umin, umax = int(u.min()), int(u.max())
    vmin, vmax = int(v.min()), int(v.max())
    wmin, wmax = int(w.min()), int(w.max())
    dimR = np.array([umax - umin + 1, vmax - vmin + 1, wmax - wmin + 1])
    # Position of *this* point within the shell bounding box.  MATLAB does:
    #   boolGrid(idx)=true; boolGrid = boolGrid(umin:umax, vmin:vmax, wmin:wmax);
    #   ptIdx3R = find(boolGrid);            % -> single 1-based linear index
    # i.e. the 1-based column-major index of the point inside the box (1..prod(dimR)).
    su, sv, sw = np.unravel_index(idx_1b - 1, Grid.shape, order="F")
    # Single 1-based local index -> store as a 1-element (column) vector to
    # mirror MATLAB (one entry per patch; concatenated in updateEntry).
    ptIdx3R = np.array([sub2ind(tuple(dimR), su - umin + 1, sv - vmin + 1, sw - wmin + 1)], dtype=int)
    return dimR, ptIdx3R

def _compute_field_vals(Hexa: Nrb, Cuboid: Cuboid, mp: int, ptIDX: int):
    """Port of ``computeFieldVals``.  All 1-based."""
    ptcID = mp
    ptIdx3 = np.array([ptIDX])
    U, V, W = ind2sub(np.asarray(Hexa.number), ptIDX)
    coefs = Hexa.coefs
    pt3D = np.asarray(coefs[0:3, U - 1, V - 1, W - 1], dtype=float)
    ptw = float(coefs[3, U - 1, V - 1, W - 1])
    ptShl = float(Cuboid.Grid[U - 1, V - 1, W - 1])
    isShlB = bool(Cuboid.ShellBoundary[U - 1, V - 1, W - 1])
    ptFrm = int(Cuboid.Frame[U - 1, V - 1, W - 1])
    dimShlR, ptIdx3R = _get_relative_grid_size_and_pt_idx(Cuboid.Grid, ptIDX)
    return ptcID, ptIdx3, pt3D, ptw, ptShl, isShlB, ptFrm, dimShlR, ptIdx3R

def _gen_void_cpt() -> CPT:
    """Port of ``genMultiPatchHexaCtrlPtsTopology`` (empty CPT)."""
    return CPT(
        pt3D=np.zeros(3),
        ptw=0.0,
        ptcIDs=np.zeros(0, dtype=int),
        ptIdx3=np.zeros(0, dtype=int),
        ptShl=0.0,
        isShlB=False,
        ptFrm=0,
        dimShlR=np.zeros((0, 3), dtype=int),
        ptIdx3R=np.zeros(0, dtype=int),
        nnIDs=np.zeros(0, dtype=int),
        nnLbls=np.zeros(0, dtype=bool),
        isSmth=True,
    )

# ---------------------------------------------------------------------------
# Entry make / update
# ---------------------------------------------------------------------------
def _make_entry(mpCPT: List[CPT], Hexa, Cuboid, mp, ptIDX) -> List[CPT]:
    """Port of ``makeEntry``."""
    (
        ptcID, ptIdx3, pt3D, ptw,
        ptShl, isShlB, ptFrm, dimShlR, ptIdx3R,
    ) = _compute_field_vals(Hexa, Cuboid, mp, ptIDX)
    entry = _gen_void_cpt()
    entry.ptcIDs = np.array([ptcID])
    entry.ptIdx3 = ptIdx3
    entry.pt3D = np.asarray(pt3D, dtype=float)
    entry.ptw = float(ptw)
    entry.ptShl = float(ptShl)
    entry.isShlB = bool(isShlB)
    entry.ptFrm = int(ptFrm)
    entry.dimShlR = np.atleast_2d(np.asarray(dimShlR, dtype=int))
    entry.ptIdx3R = np.asarray(ptIdx3R, dtype=int)
    entry.isSmth = True
    if len(mpCPT) == 0:
        return [entry]
    return mpCPT + [entry]

def _update_entry(
    mpCPT: List[CPT], mpCPTidx, Hexa, Cuboid, mp, ptIDX, verboseFlag
) -> Tuple[List[CPT], bool]:
    """Port of ``updateEntry``."""
    excFlag = False
    (
        ptcID, ptIdx3, pt3D, ptw,
        ptShl, isShlB, ptFrm, dimShlR, ptIdx3R,
    ) = _compute_field_vals(Hexa, Cuboid, mp, ptIDX)

    i = mpCPTidx - 1
    old = mpCPT[i]
    new = _gen_void_cpt()
    new.ptcIDs = np.concatenate([old.ptcIDs, np.array([ptcID])])
    new.ptIdx3 = np.concatenate([old.ptIdx3, ptIdx3])
    new.dimShlR = np.vstack([old.dimShlR, np.atleast_2d(np.asarray(dimShlR, dtype=int))])
    new.ptIdx3R = np.concatenate([old.ptIdx3R, np.asarray(ptIdx3R, dtype=int)])

    def _same(a, b):
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            return bool(a) == bool(b)
        return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float))

    # pt3D / ptw / ptShl / isShlB / ptFrm
    if not _same(old.pt3D, pt3D) or not (np.isclose(old.ptw, float(ptw))):
        if verboseFlag:
            print(" * [WRN] smthCPT: inconsistent pt3D/ptw found! - Please Check for BUGS!")
    if not np.isclose(old.ptShl, float(ptShl)):
        if verboseFlag:
            print(" * [WRN] smthCPT: inconsistent ptShl found! - Please Check for BUGS!")
    if bool(old.isShlB) != bool(isShlB):
        if verboseFlag:
            print(" * [WRN] smthCPT: inconsistent isShlB found! - Please Check for BUGS!")
    if int(old.ptFrm) != int(ptFrm):
        if verboseFlag:
            print(" * [WRN] smthCPT: inconsistent ptFrm found! - Please Check for BUGS!")
        new.ptFrm = max(int(old.ptFrm), int(ptFrm))
        excFlag = True
    else:
        new.ptFrm = int(ptFrm)

    new.pt3D = np.asarray(old.pt3D, dtype=float)
    new.ptw = float(old.ptw)
    new.ptShl = float(old.ptShl)
    new.isShlB = bool(old.isShlB)
    new.isSmth = True
    mpCPT[i] = new
    return mpCPT, excFlag

def _update_mpctrlpts(
    mpCPT: List[CPT], mpHexa, mpCuboid, mp, verboseFlag
) -> Tuple[List[CPT], bool]:
    """Port of ``updateMPctrlpts``."""
    excptFlag = False
    Hexa = mpHexa[mp - 1]
    Cuboid = mpCuboid[mp - 1]

    # cat on columns all the listed pt3D in mpCPT (3 x N)
    if len(mpCPT) == 0:
        mpCPTpts3D = np.zeros((3, 0))
    else:
        mpCPTpts3D = np.column_stack([c.pt3D for c in mpCPT])
    # MATLAB's ``reshape(coefs(1:3,:), 3, [])`` is column-major, so we must
    # use ``order="F"`` to keep ``ptIDX`` aligned with the column-major
    # ``ptIdx3`` produced by ``ind2sub`` in ``_compute_field_vals``.
    Hpts3D = np.asarray(Hexa.coefs[0:3, ...], dtype=float).reshape(3, -1, order="F")

    npts = Hpts3D.shape[1]
    for ptIDX in range(1, npts + 1):
        mpCPTidx = find_ctrl_pt(Hpts3D[:, ptIDX - 1], mpCPTpts3D)
        if mpCPTidx.size == 0:
            mpCPT = _make_entry(mpCPT, Hexa, Cuboid, mp, ptIDX)
        else:
            assert mpCPTidx.size == 1, "individual CtrlPt with multiplicity in mpCPT!"
            mpCPT, excFlag = _update_entry(
                mpCPT, int(mpCPTidx[0]), Hexa, Cuboid, mp, ptIDX, verboseFlag
            )
            excptFlag = excptFlag or excFlag
    return mpCPT, excptFlag

# ---------------------------------------------------------------------------
# Top-level aggregate
# ---------------------------------------------------------------------------
def aggregate_multi_patch_hexa(
    mpHexa: Sequence[Nrb], mpCuboid: Sequence[Cuboid], verboseFlag=False, waitbarFlag=False
) -> Tuple[List[CPT], List[CPTAttr]]:
    """Port of ``aggregateMultiPatchHexa``."""
    assert len(mpHexa) == len(mpCuboid), (
        "aggregateMultiPatchHexa: expected consistent input structures: mpHexa and mpCuboid!"
    )

    mpCPT: List[CPT] = []
    mpCPTatrb: List[CPTAttr] = []
    excptFlag = False

    for mp in range(1, len(mpHexa) + 1):
        h = mpHexa[mp - 1]
        mpCPTatrb.append(
            CPTAttr(
                form=h.form,
                dim=h.dim,
                number=tuple(h.number),
                knots=tuple(h.knots),
                order=tuple(h.order),
            )
        )
        mpCPT, eflag = _update_mpctrlpts(mpCPT, mpHexa, mpCuboid, mp, verboseFlag)
        excptFlag = excptFlag or eflag

    if excptFlag and verboseFlag:
        print(" *** Ctrl-Pts Topology *** ")
        print(" *** [WRN] Smoothing the multi-patch might locally yield inaccurate continuity within the solid medium!")

    return mpCPT, mpCPTatrb

# ---------------------------------------------------------------------------
# Patch-id -> point-id mapping
# ---------------------------------------------------------------------------
class PtcIDsListEntry:
    def __init__(self, ptcID: int, ptIDs: np.ndarray):
        self.ptcID = ptcID
        self.ptIDs = np.asarray(ptIDs, dtype=int)

def get_list_ptc_ids(mpCPT: List[CPT], ptSmthIDs: np.ndarray) -> List[PtcIDsListEntry]:
    """Port of ``getListPtcIDs`` (1-based ptSmthIDs)."""
    if np.size(ptSmthIDs) == 0:
        return []
    all_ptc = np.concatenate([mpCPT[i - 1].ptcIDs for i in ptSmthIDs])
    PtcIDs = np.unique(all_ptc)

    npt = len(mpCPT)
    LUT = np.zeros((PtcIDs.size, npt), dtype=bool)
    for pt in range(1, npt + 1):
        for ptc in mpCPT[pt - 1].ptcIDs:
            LUT[ptc - 1, pt - 1] = True

    out: List[PtcIDsListEntry] = []
    for ptc in range(PtcIDs.size):
        ptIDs = np.nonzero(LUT[ptc, :])[0] + 1
        out.append(PtcIDsListEntry(int(PtcIDs[ptc]), ptIDs))
    return out
