"""
hexiga.smth.cuboid
==================

All grid arrays use shape ``Hexa.number`` (3-D).  MATLAB subscripts are
1-based column-major; numpy arrays here are indexed with 0-based tuples that we
derive from the 1-based values via :func:`hexiga.smth.cpt.sub2ind` /
:func:`hexiga.smth.cpt.ind2sub` at the boundaries.  The *stored* ``ptIdx3``
fields are kept 1-based to match the MATLAB ground truth.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..nurbs.nrb import Nrb
from .cpt import Cuboid, ind2sub, sub2ind

__all__ = [
    "gen_cuboid_grid",
    "compute_cuboid_grid_shells",
    "compute_cuboid_grid_dist",
    "compute_frame_atrbs",
]

# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------
def gen_cuboid_grid(mpHexa: Sequence[Nrb]) -> List[Cuboid]:
    """Port of ``genCuboidGrid``."""
    mpCuboid: List[Cuboid] = []
    for hh in mpHexa:
        number = np.asarray(hh.number)
        mpCuboid.append(
            Cuboid(
                Grid=np.full(number, np.nan),
                Solved=False,
                Ready=False,
                ShellBoundary=np.zeros(number, dtype=bool),
                Frame=np.zeros(number, dtype=int),
            )
        )
    return mpCuboid

# ---------------------------------------------------------------------------
# Grid distance solve
# ---------------------------------------------------------------------------
def _get27neighbours(dim, idx: int):
    """Port of ``get27Neighbours``.

    Returns the 26 neighbours of cell ``idx`` (1-based column-major linear) as
    three 0-based subscript arrays ``(u, v, w)`` for direct numpy indexing,
    which is independent of the array's memory order.
    """
    dim = np.asarray(dim)
    sbr, sbc, sbp = ind2sub(dim, idx)  # 1-based
    rmin = max(sbr - 1, 1)
    rmax = min(sbr + 1, dim[0])
    cmin = max(sbc - 1, 1)
    cmax = min(sbc + 1, dim[1])
    pmin = max(sbp - 1, 1)
    pmax = min(sbp + 1, dim[2])

    rG, cG, pG = np.meshgrid(
        np.arange(rmin, rmax + 1),
        np.arange(cmin, cmax + 1),
        np.arange(pmin, pmax + 1),
        indexing="ij",
    )
    # drop the seed cell itself, convert to 0-based subscripts
    mask = ~((rG == sbr) & (cG == sbc) & (pG == sbp))
    return rG[mask] - 1, cG[mask] - 1, pG[mask] - 1

def compute_cuboid_grid_dist(GridDist: np.ndarray) -> np.ndarray:
    """Port of ``computeCuboidGridDist``.

    Propagates the (finite) seeded cell values outward using the 26-neighbour
    stencil, taking the element-wise min against ``seed+1`` until every cell is
    finite.
    """
    GridDist = GridDist.astype(float).copy()
    dim = GridDist.shape
    converged = int(np.isfinite(GridDist).sum()) == GridDist.size
    while not converged:
        gmax = np.nanmax(GridDist)
        # 0-based subscript triples of the (nan)max cells
        seed = np.unravel_index(np.flatnonzero(GridDist.reshape(-1, order="F") == gmax),
                                GridDist.shape, order="F")
        for s in zip(*seed):
            nu, nv, nw = _get27neighbours(dim, np.ravel_multi_index(s, dim, order="F") + 1)
            val = GridDist[s] + 1
            cand = GridDist[nu, nv, nw]
            # nanmin(GridDist(nn27), val): NaN neighbours take val, else min
            GridDist[nu, nv, nw] = np.where(np.isnan(cand), val, np.minimum(cand, val))
        converged = int(np.isfinite(GridDist).sum()) == GridDist.size
    return GridDist

# ---------------------------------------------------------------------------
# Frame computation
# ---------------------------------------------------------------------------
def _update_shl_frame_vals_grid(
    grid: np.ndarray,
    Umin: int, Umax: int,
    Vmin: int, Vmax: int,
    Wmin: int, Wmax: int,
) -> None:
    """Port of ``updateshlFrameValsGrid`` (verbatim 8-case).

    The 12 canonical frame edges are each added (+1) EXCEPT when the line
    references ``Xmax`` of a *collapsed* dimension (``Xmin == Xmax``); the
    commented-out lines in MATLAB are exactly those duplicates.
    """
    Umax_c = (Umin == Umax)
    Vmax_c = (Vmin == Vmax)
    Wmax_c = (Wmin == Wmax)

    # (u, v, w): 'm' = min, 'M' = max, 's' = span (that axis is the line)
    lines = [
        ("m", "m", "s"), ("m", "M", "s"), ("M", "m", "s"), ("M", "M", "s"),  # span W
        ("m", "s", "m"), ("m", "s", "M"), ("M", "s", "m"), ("M", "s", "M"),  # span V
        ("s", "m", "m"), ("s", "m", "M"), ("s", "M", "m"), ("s", "M", "M"),  # span U
    ]
    for (u, v, w) in lines:
        if (u == "M" and Umax_c) or (v == "M" and Vmax_c) or (w == "M" and Wmax_c):
            continue  # duplicate of a collapsed dim's min line -> commented in MATLAB
        ui = Umin if u == "m" else Umax
        vi = Vmin if v == "m" else Vmax
        wi = Wmin if w == "m" else Wmax
        if w == "s":
            grid[ui - 1, vi - 1, :] += 1
        elif v == "s":
            grid[ui - 1, :, wi - 1] += 1
        else:
            grid[:, vi - 1, wi - 1] += 1

def _get_shell_frame(ShlGrid: np.ndarray, ShlVal: float) -> np.ndarray:
    """Port of ``getShellFrame``.

    MATLAB:
        idx = find(ShlGrid == ShlVal);            % flat indices of shell cells
        [Usub, Vsub, Wsub] = ind2sub(size(ShlGrid), idx);
        Umin/Umax/Vmin/Vmax/Wmin/Wmax = min/max over the shell cells
        shlFrameValsGrid(idx) = 0;
        updateshlFrameValsGrid(...);
    We use the 3 coordinate arrays directly (equivalent to find+ind2sub), which
    is order-agnostic and avoids the flat-index/axis pitfall.
    """
    u0, v0, w0 = np.nonzero(ShlGrid == ShlVal)   # 0-based coordinate arrays
    shlFrameValsGrid = np.full(ShlGrid.shape, np.nan)
    if u0.size == 0:
        return np.zeros(ShlGrid.shape, dtype=int)

    Umin, Umax = int(u0.min()) + 1, int(u0.max()) + 1
    Vmin, Vmax = int(v0.min()) + 1, int(v0.max()) + 1
    Wmin, Wmax = int(w0.min()) + 1, int(w0.max()) + 1

    shlFrameValsGrid[u0, v0, w0] = 0
    _update_shl_frame_vals_grid(
        shlFrameValsGrid, Umin, Umax, Vmin, Vmax, Wmin, Wmax
    )

    shlFrame = np.zeros(ShlGrid.shape, dtype=int)
    finite = ~np.isnan(shlFrameValsGrid)
    shlFrame[finite & (shlFrameValsGrid == 1)] = 1   # Edges
    shlFrame[finite & (shlFrameValsGrid == 3)] = 2   # Corners
    return shlFrame

def _ind2sub_grid(Grid: np.ndarray, idx: np.ndarray) -> Tuple:
    u, v, w = np.unravel_index(idx, Grid.shape, order="F")
    return u + 1, v + 1, w + 1

def ind2sub_subgrid(Grid, idx):
    """Helper: 1-based subscripts for a numpy linear (0-based) index array."""
    return _ind2sub_grid(Grid, idx)

def _get_cuboid_patch_frame(ShlGrid: np.ndarray) -> np.ndarray:
    """Port of ``getCuboidPatchFrame``."""
    Frame = np.zeros(ShlGrid.shape, dtype=int)
    ShlVals = np.sort(np.unique(ShlGrid))
    for ShlVal in ShlVals:
        Frame = Frame + _get_shell_frame(ShlGrid, float(ShlVal))
    return Frame

def compute_frame_atrbs(mpCuboid: List[Cuboid]) -> List[Cuboid]:
    """Port of ``computeFrameAtrbs`` (waitbar omitted)."""
    for cub in mpCuboid:
        cub.Frame = _get_cuboid_patch_frame(cub.Grid)
    return mpCuboid

# ---------------------------------------------------------------------------
# Side get/set + interface/boundary propagation
# ---------------------------------------------------------------------------
def _patch_grid_idx(dim) -> np.ndarray:
    """Port of ``getPatchGridIdx``: 1-based linear index grid (sub2ind of ndgrid)."""
    dim = np.asarray(dim)
    rG, cG, pG = np.meshgrid(
        np.arange(1, dim[0] + 1),
        np.arange(1, dim[1] + 1),
        np.arange(1, dim[2] + 1),
        indexing="ij",
    )
    r = rG.reshape(-1)
    c = cG.reshape(-1)
    p = pG.reshape(-1)
    out = np.empty(r.size, dtype=int)
    for k in range(r.size):
        out[k] = sub2ind(dim, int(r[k]), int(c[k]), int(p[k]))
    return out

def _get_cuboid_grid_side_vals(mpCuboid, patchID: int, sideID: int):
    """Port of ``getCuboidGridSideVals`` (1-based patchID/sideID)."""
    Grid = mpCuboid[patchID - 1].Grid
    GridIdxs = _patch_grid_idx(Grid.shape)
    r0 = np.arange(Grid.shape[0])
    c0 = np.arange(Grid.shape[1])
    p0 = np.arange(Grid.shape[2])
    # column-major ordering: index = (u-1)*nv*nw + (v-1)*nw + (w-1)
    def vals_mask(u, v, w):
        return (u, v, w)
    slices = {
        1: (slice(0, 1), slice(None), slice(None)),
        2: (slice(-1, None), slice(None), slice(None)),
        3: (slice(None), slice(0, 1), slice(None)),
        4: (slice(None), slice(-1, None), slice(None)),
        5: (slice(None), slice(None), slice(0, 1)),
        6: (slice(None), slice(None), slice(-1, None)),
    }
    s = slices[sideID]
    # Vals linearised in column-major order
    Vals = np.asarray(Grid[s]).reshape(-1, order="F") if Grid[s].ndim > 1 else np.asarray(Grid[s]).reshape(-1)
    # Idxs: build full index grid and select the same slice
    idxgrid = _idx_grid(Grid.shape)
    Idxs = idxgrid[s].reshape(-1, order="F") if idxgrid[s].ndim > 1 else idxgrid[s].reshape(-1)
    return Vals, Idxs

def _idx_grid(dim) -> np.ndarray:
    dim = np.asarray(dim)
    out = np.empty(dim, dtype=int)
    for u in range(dim[0]):
        for v in range(dim[1]):
            for w in range(dim[2]):
                out[u, v, w] = sub2ind(dim, u + 1, v + 1, w + 1)
    return out

def _set_cuboid_grid_side_vals(Grid, sideID, Vals, Idxs):
    """Port of ``setCuboidGridSideVals``.  Returns the modified Grid in-place.

    MATLAB:
        if isempty(Idxs)
            Grid(side-slice) = Vals;
        else
            Grid(Idxs) = Vals;   % element-wise linear-index assignment
        end
    """
    if np.size(Idxs) == 0:
        slices = {
            1: (slice(0, 1), slice(None), slice(None)),
            2: (slice(-1, None), slice(None), slice(None)),
            3: (slice(None), slice(0, 1), slice(None)),
            4: (slice(None), slice(-1, None), slice(None)),
            5: (slice(None), slice(None), slice(0, 1)),
            6: (slice(None), slice(None), slice(-1, None)),
        }
        Grid[slices[sideID]] = Vals
    else:
        # MATLAB Grid(Idxs) = Vals uses COLUMN-MAJOR linear indexing.
        Idxs = np.asarray(Idxs, dtype=int).ravel()
        Vals = np.asarray(Vals).ravel(order="F")
        # Convert 1-based column-major linear indices to 0-based (u,v,w)
        # subscripts and assign via numpy indexing (memory-order independent).
        u, v, w = np.unravel_index(Idxs - 1, Grid.shape, order="F")
        Grid[u, v, w] = Vals
    return Grid

def _get_mapped_cuboid_grid_side_vals(
    mpCuboid, patchID, sideID, flag, ornt1, ornt2
):
    """Port of ``getMappedCuboidGridSideVals``."""
    Grid = mpCuboid[patchID - 1].Grid
    # select side (squeezed 2-D)
    slices = {
        1: (slice(0, 1), slice(None), slice(None)),
        2: (slice(-1, None), slice(None), slice(None)),
        3: (slice(None), slice(0, 1), slice(None)),
        4: (slice(None), slice(-1, None), slice(None)),
        5: (slice(None), slice(None), slice(0, 1)),
        6: (slice(None), slice(None), slice(-1, None)),
    }
    Vals = np.squeeze(np.asarray(Grid[slices[sideID]]))
    Idxs = np.squeeze(_idx_grid(Grid.shape)[slices[sideID]])

    T = np.transpose
    FU = lambda A: A[::-1, :]        # flipud
    FL = lambda A: A[:, ::-1]        # fliplr

    if (flag == 1) and (ornt1 == 1) and (ornt2 == 1):
        pass
    elif (flag == -1) and (ornt1 == 1) and (ornt2 == 1):
        Vals = T(Vals); Idxs = T(Idxs)
    elif (flag == -1) and (ornt1 == -1) and (ornt2 == 1):
        Vals = T(Vals); Idxs = T(Idxs); Vals = FU(Vals); Idxs = FU(Idxs)
    elif (flag == 1) and (ornt1 == -1) and (ornt2 == 1):
        Vals = FU(Vals); Idxs = FU(Idxs)
    elif (flag == 1) and (ornt1 == -1) and (ornt2 == -1):
        Vals = FU(Vals); Idxs = FU(Idxs); Vals = FL(Vals); Idxs = FL(Idxs)
    elif (flag == -1) and (ornt1 == -1) and (ornt2 == -1):
        Vals = FU(Vals); Idxs = FU(Idxs)
        Vals = T(Vals); Idxs = T(Idxs)
        Vals = FU(Vals); Idxs = FU(Idxs)
    elif (flag == -1) and (ornt1 == 1) and (ornt2 == -1):
        Vals = FU(Vals); Idxs = FU(Idxs); Vals = T(Vals); Idxs = T(Idxs)
    elif (flag == 1) and (ornt1 == 1) and (ornt2 == -1):
        Vals = FL(Vals); Idxs = FL(Idxs)

    Vals = np.asarray(Vals).reshape(-1, order="F")
    Idxs = np.asarray(Idxs).reshape(-1, order="F")
    return Vals, Idxs

# ---------------------------------------------------------------------------
# Propagation steps
# ---------------------------------------------------------------------------
def _update_interfaces(mpCuboid, interfaces) -> None:
    """Port of ``updateInterfaces``."""
    for patchID in range(1, len(mpCuboid) + 1):
        cub = mpCuboid[patchID - 1]
        if cub.Ready or cub.Solved:
            continue
        for ii in interfaces:
            if ii["patch1"] == patchID:
                trg_sideID = ii["side1"]
                src_patchID = ii["patch2"]
                src_sideID = ii["side2"]
                updateInterface = True
            elif ii["patch2"] == patchID:
                trg_sideID = ii["side2"]
                src_patchID = ii["patch1"]
                src_sideID = ii["side1"]
                updateInterface = True
            else:
                updateInterface = False

            if updateInterface:
                src_Vals, _ = _get_cuboid_grid_side_vals(
                    mpCuboid, src_patchID, src_sideID
                )
                if np.all(np.isfinite(src_Vals)) and len(np.unique(src_Vals)) == 1:
                    src_Vals = np.unique(src_Vals)
                    mpCuboid[patchID - 1].Grid = _set_cuboid_grid_side_vals(
                        mpCuboid[patchID - 1].Grid, trg_sideID, src_Vals, []
                    )
                    mpCuboid[patchID - 1].Ready = True
    return mpCuboid

def _is_member_pair(pair, pairs) -> bool:
    """MATLAB ``ismember([patchID,sideID], ExeptPatchSidePairs)``."""
    if pairs is None or len(pairs) == 0:
        return False
    arr = np.atleast_2d(pairs)
    for row in arr:
        if (row[0] == pair[0]) and (row[1] == pair[1]):
            return True
    return False

def _update_boundary(mpCuboid, boundaries, interfaces, OPTs) -> None:
    """Port of ``updateBoundary``."""
    BoundaryValue = 0
    ShellBoundaryValue = True
    BoundaryApplied = False
    ExeptPatchSidePairs = OPTs.get("ExeptPatchSidePairs", [])
    InOutletsSides = np.atleast_1d(OPTs.get("InOutletsSides", []))
    ExeptInOutletsPatches = np.atleast_1d(OPTs.get("ExeptInOutletsPatches", []))

    for patchID in range(1, len(mpCuboid) + 1):
        cub = mpCuboid[patchID - 1]
        if cub.Ready or cub.Solved:
            continue
        BoundarySides = []
        ShellBoundarySides = []
        for bb in boundaries:
            if bb["patches"] == patchID:
                sideID = bb["faces"]
                if not _is_member_pair([patchID, sideID], ExeptPatchSidePairs):
                    if not (
                        (sideID in list(InOutletsSides))
                        and (patchID not in list(ExeptInOutletsPatches))
                    ):
                        BoundarySides.append(sideID)
                    else:
                        ShellBoundarySides.append(sideID)

        if len(BoundarySides) > 0:
            for bs in BoundarySides:
                mpCuboid[patchID - 1].Grid = _set_cuboid_grid_side_vals(
                    mpCuboid[patchID - 1].Grid, bs, BoundaryValue, []
                )
                BoundaryApplied = True
            mpCuboid[patchID - 1].Ready = True
        if len(ShellBoundarySides) > 0:
            for sb in ShellBoundarySides:
                mpCuboid[patchID - 1].ShellBoundary = _set_cuboid_grid_side_vals(
                    mpCuboid[patchID - 1].ShellBoundary, sb, ShellBoundaryValue, []
                )

    if BoundaryApplied:
        _enforce_consistent_interfaces(mpCuboid, interfaces, OPTs.get("verboseFlag", False))
    return mpCuboid

def _enforce_consistent_interfaces(mpCuboid, interfaces, verboseFlag) -> None:
    """Port of ``enforceConsistentInterfaces`` (bug preserved: patch2 uses patch1)."""
    for patchID in range(1, len(mpCuboid) + 1):
        cub = mpCuboid[patchID - 1]
        if not (cub.Ready and not cub.Solved):
            continue
        for ii in interfaces:
            if ii["patch1"] == patchID:
                ptc1, side1 = patchID, ii["side1"]
                ptc2, side2 = ii["patch2"], ii["side2"]
                chk = True
            elif ii["patch2"] == patchID:
                ptc1, side1 = ii["patch1"], ii["side1"]
                ptc2, side2 = patchID, ii["side2"]
                chk = True
            else:
                chk = False
            if not chk:
                continue
            flag = ii["flag"]
            ornt1 = ii["ornt1"]
            ornt2 = ii["ornt2"]

            ptc1Vals, ptc1Idxs = _get_cuboid_grid_side_vals(mpCuboid, ptc1, side1)
            ptc2Vals, ptc2Idxs = _get_mapped_cuboid_grid_side_vals(
                mpCuboid, ptc2, side2, flag, ornt1, ornt2
            )

            ptc1Vals = np.asarray(ptc1Vals, dtype=float)
            ptc2Vals = np.asarray(ptc2Vals, dtype=float)

            if np.array_equal(np.isnan(ptc1Vals), np.isnan(ptc2Vals)):
                # numeric check (no change)
                n1 = ptc1Vals[~np.isnan(ptc1Vals)]
                n2 = ptc2Vals[~np.isnan(ptc2Vals)]
                if not np.array_equal(n1, n2) and verboseFlag:
                    print(" * enforceConsistentInterfaces: Unexpected Inconsistency! BUG?")
                setConsistentInterface = False
            else:
                for ss in range(len(ptc1Vals)):
                    if np.isnan(ptc1Vals[ss]) and not np.isnan(ptc2Vals[ss]):
                        ptc1Vals[ss] = ptc2Vals[ss]
                    if np.isnan(ptc2Vals[ss]) and not np.isnan(ptc1Vals[ss]):
                        ptc2Vals[ss] = ptc1Vals[ss]
                setConsistentInterface = True

            if setConsistentInterface:
                mpCuboid[ptc1 - 1].Grid = _set_cuboid_grid_side_vals(
                    mpCuboid[ptc1 - 1].Grid, side1, ptc1Vals, ptc1Idxs
                )
                mpCuboid[ptc2 - 1].Grid = _set_cuboid_grid_side_vals(
                    mpCuboid[ptc2 - 1].Grid, side2, ptc2Vals, ptc2Idxs
                )
                mpCuboid[ptc1 - 1].Ready = True
                # MATLAB bug: uses ptc1 twice (ground-truth quirk, preserved)
                mpCuboid[ptc2 - 1].Ready = mpCuboid[ptc1 - 1].Ready or True
    return mpCuboid

def _solve_cuboids(mpCuboid) -> None:
    """Port of ``solveCuboids``."""
    for cub in mpCuboid:
        if cub.Ready and not cub.Solved:
            cub.Grid = compute_cuboid_grid_dist(cub.Grid)
            cub.Solved = True
    return mpCuboid

def _sort_shell_values(mpCuboid) -> None:
    """Port of ``sortShellValues``."""
    maxShellVal = 0
    for cub in mpCuboid:
        maxShellVal = max(maxShellVal, np.max(cub.Grid))
    for cub in mpCuboid:
        cub.Grid = np.abs(cub.Grid - maxShellVal)
    return mpCuboid

# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def compute_cuboid_grid_shells(mpCuboid, boundaries, interfaces, OPTs):
    """Port of ``computeCuboidGridShells``."""
    while not all(c.Solved for c in mpCuboid):
        _update_interfaces(mpCuboid, interfaces)
        _update_boundary(mpCuboid, boundaries, interfaces, OPTs)
        _solve_cuboids(mpCuboid)
    _sort_shell_values(mpCuboid)
    compute_frame_atrbs(mpCuboid)
    return mpCuboid
