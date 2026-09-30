"""
hexiga.nurbs.glue
==================

Faithful Python port of MATLAB's ``nrbglue.m``: glue two NURBS patches
along a common side, producing a single C^0-continuous patch whose
parameter domain spans both inputs.

The resulting patch has the same overall orientation as ``nrb1``.

MATLAB algorithm (``nrbglue.m``, ported step by step):

1.  ``nrbmultipatch`` finds the shared side(s); error if none.
2.  Resolve gluing sides (explicit side numbers, or the unique interface).
3.  Gluing directions ``dir1 = ceil(side1/2)``, ``dir2 = ceil(side2/2)``.
4.  Degree-elevate the lower-order patch along its gluing direction so both
    share the same order (MATLAB warns when this happens).
5.  Decide which patch plays the role of ``nrbs(1)`` (the low-end of the
    concatenated first axis) using ``mod(side1, 2)``.
6.  If ``mod(side1, 2) == mod(side2, 2)`` (both sides at the same parametric
    end), reverse ``nrb2`` along its gluing direction so the two boundaries
    meet end-to-end.
7.  Rearrange both patches so their gluing direction is the first axis.
8.  Re-run ``nrbmultipatch`` on the rearranged pair to obtain fresh
    orientation metadata (``ornt`` / ``flag`` / ``ornt1`` / ``ornt2``).
9.  If ``ornt == -1`` (2D) or ``flag/ornt1/ornt2 == -1`` (3D), apply the
    corresponding reversal / transposition to ``nrb2``.
10. Concatenate knot vectors and control points along the first axis,
    dropping the duplicated boundary column of ``nrbs(id2)``.
11. ``nrbmak`` builds the glued patch.
12. Restore ``nrb1``'s original parametric-direction order with
    ``nrbpermute`` (MATLAB special-cases the 3D ``dir1 == 3`` case as
    ``[2 3 1]`` -- we reproduce that behaviour exactly).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .nrb import Nrb
from .make import nrbmak
from .degelev import nrbdegelev
from .reverse import nrbreverse
from .permute import nrbpermute
from .multipatch import nrbmultipatch

__all__ = ["nrbglue"]

def _raise_one_dir(nrb: Nrb, direction: int, count: int) -> Nrb:
    """Elevate the degree of ``nrb`` by ``count`` along a single 1-based
    direction (MATLAB's one-hot ``deg_raise`` vector).
    """
    ndim = nrb.ndim
    deg_raise = [0] * ndim
    deg_raise[direction - 1] = count
    return nrbdegelev(nrb, deg_raise)

def _restore_perm(ndim: int, dir1: int) -> Optional[List[int]]:
    """Compute the direction-permutation that undoes moving parametric
    direction ``dir1`` (1-based) to the front.

    MATLAB special-cases the 3-D case:
        if (dir_change(1) ~= 3)
            glued = nrbpermute (glued, dir_change);
        else
            glued = nrbpermute (glued, [2 3 1]);
        end
    where ``dir_change = [dir1, setdiff(1:ndim, dir1)]``.  We mirror that
    exactly.
    """
    if ndim <= 1:
        return None
    dir_change = [dir1] + [d for d in range(1, ndim + 1) if d != dir1]
    if ndim == 3 and dir1 == 3:
        return [2, 3, 1]
    return dir_change

def nrbglue(
    nrb1: Nrb,
    nrb2: Nrb,
    side1: Optional[int] = None,
    side2: Optional[int] = None,
) -> Nrb:
    """Glue ``nrb1`` and ``nrb2`` along a common side.

    Parameters
    ----------
    nrb1, nrb2
        Two NURBS patches of the same dimension (1, 2 or 3 parametric
        directions) that share a geometric side.
    side1, side2
        Optional 1-based side numbers selecting *which* shared side to glue
        (needed when the two patches can be glued in more than one way, e.g.
        a ring).  Side numbers run ``1 .. 2*ndim``; the gluing direction is
        ``ceil(side/2)``.  When omitted, the unique shared side is used.

    Returns
    -------
    Nrb
        The single NURBS patch obtained by gluing the two inputs.

    Raises
    ------
    ValueError
        If the patches cannot be glued (non-compatible sides, wrong side
        numbers, or more than one possible gluing without an explicit side).
    """
    if nrb1.ndim != nrb2.ndim:
        raise ValueError("The patches cannot be glued together: "
                         "non-compatible sides. Check degree and knot vectors")

    # 1. Detect the shared side.
    interfaces, _ = nrbmultipatch([nrb1, nrb2])
    if len(interfaces) == 0:
        raise ValueError("The patches cannot be glued together: "
                         "non-compatible sides. Check degree and knot vectors")

    ndim = nrb1.ndim

    # 2. Resolve the gluing sides.
    if side1 is not None and side2 is not None:
        side1 = int(side1)
        side2 = int(side2)
        if side1 > 2 * ndim or side2 > 2 * ndim or side1 < 1 or side2 < 1:
            raise ValueError("Wrong number of the sides")
        match = [
            it for it in interfaces
            if it["side1"] == side1 and it["side2"] == side2
        ]
        if not match:
            raise ValueError("The input sides are not correct")
        sel = match[0]
    else:
        if len(interfaces) > 1:
            raise ValueError("The two patches can be glued in more than one "
                             "way (a ring?). Specify the sides")
        sel = interfaces[0]
    side1 = int(sel["side1"])
    side2 = int(sel["side2"])

    dir1 = (side1 + 1) // 2  # ceil(side1/2)
    dir2 = (side2 + 1) // 2

    # 4. Degree elevation so both patches share the same order along the
    #    gluing direction (MATLAB warns when this happens).
    o1 = nrb1.order[dir1 - 1]
    o2 = nrb2.order[dir2 - 1]
    if o1 < o2:
        print(
            " * nrbglue: The order of nrb1 has been raised along the "
            "gluing direction"
        )
        nrb1 = _raise_one_dir(nrb1, dir1, o2 - o1)
    elif o1 > o2:
        print(
            " * nrbglue: The order of nrb2 has been raised along the "
            "gluing direction"
        )
        nrb2 = _raise_one_dir(nrb2, dir2, o1 - o2)

    # 5. Decide which patch plays the role of nrbs(1) (the low end of the
    #    concatenated first axis).
    if side1 % 2 == 0:
        id1, id2 = 1, 2
    else:
        id1, id2 = 2, 1

    # 6. If both gluing sides sit at the same parametric end, reverse nrb2
    #    along its gluing direction so the boundaries meet end-to-end.
    if side1 % 2 == side2 % 2:
        nrb2 = nrbreverse(nrb2, dir2)

    # 7. Make the gluing direction the first parametric axis for both.
    if ndim > 1:
        dir_change1 = [dir1] + [d for d in range(1, ndim + 1) if d != dir1]
        dir_change2 = [dir2] + [d for d in range(1, ndim + 1) if d != dir2]
        nrb1 = nrbpermute(nrb1, dir_change1)
        nrb2 = nrbpermute(nrb2, dir_change2)

    # 8. Re-run nrbmultipatch on the rearranged pair for fresh orientation
    #    metadata.
    nrbs = [nrb1, nrb2]
    interfaces2, _ = nrbmultipatch(nrbs)
    if len(interfaces2) == 0:
        raise ValueError(
            "The patches cannot be glued together: non-compatible sides. "
            "Check degree and knot vectors"
        )
    # Pick the interface whose side1 matches the low-end parity of side1.
    if len(interfaces2) > 1:
        side = (side1 - 1) % 2 + 1
        matches = [it for it in interfaces2 if it["side1"] == side]
        sel2 = matches[0] if matches else interfaces2[0]
    else:
        sel2 = interfaces2[0]

    # 9. Orientation fixes on the second patch.
    if ndim == 2:
        if sel2.get("ornt", 1) == -1:
            nrb2 = nrbreverse(nrb2, 2)
    elif ndim == 3:
        if sel2.get("flag", 1) == -1:
            nrb2 = nrbpermute(nrb2, [1, 3, 2])
        if sel2.get("ornt1", 1) == -1:
            nrb2 = nrbreverse(nrb2, 2)
        if sel2.get("ornt2", 1) == -1:
            nrb2 = nrbreverse(nrb2, 3)
    nrbs = [nrb1, nrb2]

    # 10. Concatenate knots and control points along the first axis,
    #     dropping the duplicated boundary column of nrbs(id2).
    a = nrbs[id1 - 1]
    b = nrbs[id2 - 1]
    order = a.order[0]
    knot0 = np.concatenate([a.knots[0][:-1], b.knots[0][order:] + a.knots[0][-1]])

    if ndim == 1:
        knots = tuple([knot0])
        coefs = np.concatenate([a.coefs, b.coefs[:, 1:]], axis=1)
    else:
        knots = list(a.knots)
        knots[0] = knot0
        knots = tuple(knots)
        coefs = np.concatenate([a.coefs, b.coefs[:, 1:, ...]], axis=1)

    glued = nrbmak(coefs, knots)

    # 12. Recover nrb1's original parametric-direction order.
    if ndim > 1:
        perm = _restore_perm(ndim, dir1)
        if perm is not None:
            glued = nrbpermute(glued, perm)

    return glued
