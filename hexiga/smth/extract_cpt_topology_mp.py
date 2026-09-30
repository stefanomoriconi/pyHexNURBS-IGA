"""
hexiga.smth.extract_cpt_topology_mp
===================================

"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.multipatch import nrbmultipatch
from .aggregate import aggregate_multi_patch_hexa
from .cpt import CPT, CPTAttr
from .cuboid import compute_cuboid_grid_shells, gen_cuboid_grid

__all__ = ["extract_ctrl_pts_topology_from_multipatch_hexa"]

def _make_non_rational(mpHexa: Sequence[Nrb]) -> List[Nrb]:
    """Port of ``makeNonRational``.

    Splits the homogeneous 4-row coefficient array into Cartesian 3 rows and a
    weight row (``coefs(4,:)``).  ``coefs(1:3) = coefs(1:3) ./ coefs(4)``,
    ``coefs(4) = 1``.  A fresh :class:`Nrb` is returned with ``dim=4`` and
    ``number``/``knots``/``order`` unchanged, but the stored coefficient
    layout already matches the homogeneous 4-row layout used downstream.
    """
    out: List[Nrb] = []
    for h in mpHexa:
        coefs = np.asarray(h.coefs, dtype=float)
        if coefs.shape[0] == 3:
            # already Cartesian -> make homogeneous (w = 1)
            ones = np.ones((1,) + coefs.shape[1:])
            new = np.concatenate([coefs, ones], axis=0)
        else:
            w = coefs[3, ...]
            xyz = coefs[0:3, ...] / w
            ones = np.ones((1,) + w.shape)
            new = np.concatenate([xyz, ones], axis=0)
        new_h = Nrb(
            number=tuple(h.number),
            order=tuple(h.order),
            knots=tuple(h.knots),
            coefs=new,
        )
        new_h.dim = 4
        out.append(new_h)
    return out

def extract_ctrl_pts_topology_from_multipatch_hexa(
    mpHexa: Sequence[Nrb], *pairs
) -> Tuple[List[CPT], List[CPTAttr], List]:
    """Port of ``extractCtrlPtsTopologyFromMultiPatchHexa`` (non-verbose path).

    ``pairs`` are the trailing MATLAB name/value arguments, e.g.
    ``'INOUTLETSSIDES', [...], 'EXEPTINOUTLETSPATCHES', [...], ...``.
    """
    OPTs: dict = {}
    it = iter(pairs)
    for key in it:
        OPTs[str(key).upper()] = next(it)
    # MATLAB's getInputs uses these camelCase keys; also accept the
    # uppercase name/value spellings used by the caller.
    if "InOutletsSides" not in OPTs:
        OPTs["InOutletsSides"] = OPTs.get("INOUTLETSSIDES", [])
    if "ExeptInOutletsPatches" not in OPTs:
        OPTs["ExeptInOutletsPatches"] = OPTs.get("EXEPTINOUTLETSPATCHES", [])
    if "ExeptPatchSidePairs" not in OPTs:
        OPTs["ExeptPatchSidePairs"] = OPTs.get("EXEPTPATCHSIDEPAIRS", [])
    OPTs.setdefault("verboseFlag", False)
    OPTs.setdefault("waitbarFlag", False)

    if len(mpHexa) == 0:
        return [], [], []

    mpHexa = _make_non_rational(mpHexa)
    interfaces, boundaries = nrbmultipatch(mpHexa)
    mpCuboid = gen_cuboid_grid(mpHexa)
    mpCuboid = compute_cuboid_grid_shells(mpCuboid, boundaries, interfaces, OPTs)
    mpCPT, mpCPTatrb = aggregate_multi_patch_hexa(
        mpHexa, mpCuboid, OPTs["verboseFlag"], OPTs["waitbarFlag"]
    )
    return mpCPT, mpCPTatrb, mpCuboid
