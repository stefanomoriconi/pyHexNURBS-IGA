"""
hexiga.nurbs.multipatch
=======================

Detects shared interfaces / exterior boundaries across a list of NURBS
patches (curves, surfaces or volumes): finds the shared interfaces (pairs
of coincident sides) and the free boundary sides.

Algorithm:

* every patch contributes ``2*ndim`` sides, obtained via
  :func:`hexiga.nurbs.extract.nrbextract`;
* sides are matched greedily: for each still-unmatched side of patch
  ``i1``, scan the still-unmatched sides of the *later* patches
  ``i2 = i1+1 .. npatch`` and compare;
* comparison:

  - 1-D: coincident endpoint coefs, ``max(abs(c1-c2)) < 1e-15``;
  - 2-D: :func:`_compare_sides_univariate` (coincident boundary curve up
    to reversal);
  - 3-D: :func:`_compare_sides_bivariate` (coincident boundary surface up
    to transposition and/or reversal);

* unmatched sides are recorded as boundary.

Output (1-based patch/side numbers, as in MATLAB):

* ``interfaces`` -- list of dicts
  ``{'patch1','side1','patch2','side2'}`` (1-D), plus ``'ornt'`` (2-D) or
  ``'flag','ornt1','ornt2'`` (3-D);
* ``boundary`` -- list of dicts ``{'nsides': 1, 'patches': i1, 'faces': j1}``.
"""

from __future__ import annotations

import warnings
from typing import List, Sequence

import numpy as np

from .nrb import Nrb
from .extract import nrbextract
from .reverse import nrbreverse
from .transp import nrbtransp

__all__ = ["nrbmultipatch"]

_TOL = 1e-15
_TOLKNT = 1e-6

def _scale_tol(ref: np.ndarray) -> float:
    """A robust relative tolerance.

    MATLAB computes ``tol = 1e-6 * max(abs(c(:,1) - c(:,end)))``.  When that
    reference is zero (e.g. a patch with a single control point, or a face
    with constant first/last columns) the tolerance collapses to zero and
    every comparison fails.  Fall back to ``1e-9 * max(abs(ref))`` so the
    tolerance stays relative to the magnitude of the coefficients.
    """
    ref = np.asarray(ref, dtype=float)
    base = 1e-6 * np.max(np.abs(ref))
    if base == 0:
        base = 1e-9 * np.max(np.abs(ref))
    if base == 0:
        base = _TOL
    return float(base)

def _display_warning(msgflag: int, i1: int, j1: int, i2: int, j2: int) -> None:
    """Port of MATLAB's ``display_warning`` subfunction."""
    if msgflag == -1:
        msg = (
            "the number of control points is different. No information is "
            "saved in this case"
        )
    elif msgflag == -2:
        msg = "the internal control points do not"
    elif msgflag == -3:
        msg = "the degree is different"
    elif msgflag == -4:
        msg = "the number of knots is different"
    elif msgflag == -5:
        msg = "the knot vectors are different"
    else:
        msg = "are different"
    warnings.warn(
        f"The corners of PATCH {i1} FACE {j1}, and PATCH {i2} FACE {j2} "
        f"coincide, but {msg}."
    )

def _compare_sides_univariate(nrb1: Nrb, nrb2: Nrb) -> tuple[int, int]:
    """Port of ``compare_sides_univariate``: compare two boundary *curves*
    up to reversal.  Returns ``(flag, MsgFlag)``; ``flag`` is ``+1`` /
    ``-1`` (reversed) / ``0`` (no match), ``MsgFlag`` is ``0`` or a
    negative code (1..-5) explaining a near-miss.

    MATLAB compares *corners only* (``reshape(x.coefs(:, [1 end]), 4, [])``);
    the full control-point and knot-vector comparison happens after any
    reversal/alignment.
    """
    c1 = np.asarray(nrb1.coefs, dtype=float)[:, [0, -1]]          # (4, 2)
    c2 = np.asarray(nrb2.coefs, dtype=float)[:, [0, -1]]          # (4, 2)
    full1 = np.asarray(nrb1.coefs, dtype=float)

    tol = _scale_tol(c1[0:3, 0] - c1[0:3, -1])
    tolknt = _TOLKNT

    MsgFlag = 0
    flag = 0
    if np.max(np.abs(c1 - c2)) < tol:
        flag = 1
    elif np.max(np.abs(c1 - c2[:, [1, 0]])) < tol:
        flag = -1

    if flag != 0:
        if flag == -1:
            nrb2 = nrbreverse(nrb2)
        if tuple(nrb1.order) != tuple(nrb2.order):
            flag = 0
            MsgFlag = -3
        elif tuple(nrb1.number) != tuple(nrb2.number):
            flag = 0
            MsgFlag = -1
        elif nrb1.knots[0].size != nrb2.knots[0].size:
            flag = 0
            MsgFlag = -4
        else:
            k1 = nrb1.knots[0]
            k2 = nrb2.knots[0]
            knt1 = (k1 - k1[0]) / (k1[-1] - k1[0])
            knt2 = (k2 - k2[0]) / (k2[-1] - k2[0])
            if np.max(np.abs(full1.ravel() - nrb2.coefs.ravel())) > tol:
                flag = 0
                MsgFlag = -2
            elif np.max(np.abs(knt1 - knt2)) > tolknt:
                flag = 0
                MsgFlag = -5

    return int(flag), int(MsgFlag)

def _compare_sides_bivariate(
    nrb1: Nrb, nrb2: Nrb
) -> tuple[int, int, int, int]:
    """Port of ``compare_sides_bivariate``: compare two boundary
    *surfaces* up to transposition and/or reversal.  Returns
    ``(flag, ornt1, ornt2, MsgFlag)``; ``flag`` is ``+1``/``-1``
    (transposed) / ``0``, ``ornt1``/``ornt2`` are ``+1``/``-1`` (0 on no
    match), ``MsgFlag`` is ``0`` or a negative code."""
    # Face corners: MATLAB reshape(x.coefs(:, [1 end], [1 end]), 4, [])
    # -> 4x4 matrix, one column per corner, column-major flattening.
    # MATLAB uses implicit meshgrid for `coefs(:, [1 end], [1 end])`,
    # yielding shape (4, 2, 2).  NumPy pairs two index arrays element-wise
    # by default, so we use np.ix_ to reproduce the MATLAB behaviour.
    _c1 = np.asarray(nrb1.coefs, dtype=float)
    # MATLAB `coefs(:, [1 end], [1 end])` = implicit meshgrid, shape (4, 2, 2).
    # NumPy pairs index arrays element-wise, so build the 2-D grid manually.
    _iu = np.array([[0], [-1]])   # (2,1) -> 2nd axis
    _iv = np.array([[0, -1]])     # (1,2) -> 3rd axis
    c1 = _c1[:, _iu, _iv].reshape(4, 4, order="F")
    c1full = _c1

    tol = _scale_tol(c1full[0:3, 0] - c1full[0:3, -1])
    tolknt = _TOLKNT

    flag = 0
    ornt1 = 0
    ornt2 = 0
    # The 8 corner permutations, exactly as in nrbmultipatch.m.
    for perm, f, o1, o2 in (
        ((0, 1, 2, 3), 1, 1, 1),
        ((0, 2, 1, 3), -1, 1, 1),
        ((2, 0, 3, 1), -1, -1, 1),
        ((1, 0, 3, 2), 1, -1, 1),
        ((3, 2, 1, 0), 1, -1, -1),
        ((3, 1, 2, 0), -1, -1, -1),
        ((1, 3, 0, 2), -1, 1, -1),
        ((2, 3, 0, 1), 1, 1, -1),
    ):
        c2 = np.asarray(nrb2.coefs, dtype=float)[
            :, _iu, _iv
        ].reshape(4, 4, order="F")
        if np.max(np.abs(c1 - c2[:, list(perm)])) < tol:
            flag = f
            ornt1 = o1
            ornt2 = o2
            break

    MsgFlag = 0
    if flag != 0:
        if flag == -1:
            nrb2 = nrbtransp(nrb2)
        if ornt1 == -1:
            nrb2 = nrbreverse(nrb2, 1)
        if ornt2 == -1:
            nrb2 = nrbreverse(nrb2, 2)
        if nrb1.order != nrb2.order:
            flag = 0
            MsgFlag = -3
        elif nrb1.number != nrb2.number:
            flag = 0
            MsgFlag = -1
        elif any(nrb1.knots[a].size != nrb2.knots[a].size for a in range(2)):
            flag = 0
            MsgFlag = -4
        else:
            knt1 = (
                (nrb1.knots[0] - nrb1.knots[0][0])
                / (nrb1.knots[0][-1] - nrb1.knots[0][0]),
                (nrb1.knots[1] - nrb1.knots[1][0])
                / (nrb1.knots[1][-1] - nrb1.knots[1][0]),
            )
            knt2 = (
                (nrb2.knots[0] - nrb2.knots[0][0])
                / (nrb2.knots[0][-1] - nrb2.knots[0][0]),
                (nrb2.knots[1] - nrb2.knots[1][0])
                / (nrb2.knots[1][-1] - nrb2.knots[1][0]),
            )
            if np.max(
                np.abs(
                    np.asarray(nrb1.coefs).ravel()
                    - np.asarray(nrb2.coefs).ravel()
                )
            ) > tol:
                flag = 0
                MsgFlag = -2
            elif (
                np.max(np.abs(knt1[0] - knt2[0])) > tolknt
                or np.max(np.abs(knt1[1] - knt2[1])) > tolknt
            ):
                flag = 0
                MsgFlag = -5

    return int(flag), int(ornt1), int(ornt2), int(MsgFlag)

def nrbmultipatch(
    nurbs: Sequence[Nrb],
) -> tuple[List[dict], List[dict]]:
    """Detect interfaces and boundary sides of a list of NURBS patches.

    Parameters
    ----------
    nurbs
        List of NURBS patches, all with the same dimension (1, 2 or 3).

    Returns
    -------
    (interfaces, boundary)
        ``interfaces``: list of dicts ``{'patch1', 'side1', 'patch2',
        'side2'}`` with 1-based patch/side numbers; for 2-D patches an
        extra key ``'ornt'`` (``+1``/``-1``), for 3-D patches the keys
        ``'flag'``, ``'ornt1'``, ``'ornt2'``.

        ``boundary``: list of dicts ``{'nsides': 1, 'patches': i1,
        'faces': j1}``.
    """
    if len(nurbs) == 0:
        return [], []

    ndim = nurbs[0].ndim
    npatch = len(nurbs)
    for p in nurbs:
        if p.ndim != ndim:
            raise ValueError(
                "All the patches must have the same dimension (at least "
                "for now)"
            )

    non_set_faces: List[List[int]] = [
        list(range(1, 2 * ndim + 1)) for _ in range(npatch)
    ]

    nrb_faces = [nrbextract(p) for p in nurbs]

    interfaces: List[dict] = []
    boundary: List[dict] = []

    for i1 in range(npatch):
        nrb_faces1 = nrb_faces[i1]
        for j1 in list(non_set_faces[i1]):
            # MATLAB: if isempty(intersect(non_set_faces{i1}, j1));
            # continue; end
            if j1 not in non_set_faces[i1]:
                continue
            nrb1 = nrb_faces1[j1 - 1]
            non_set_faces[i1].remove(j1)

            flag = 0
            ornt1 = 0
            ornt2 = 0
            ornt = 0
            MsgFlag = 0
            i2 = i1
            side2 = -1
            while flag == 0 and i2 < npatch - 1:
                i2 += 1
                nrb_faces2 = nrb_faces[i2]
                for j2pos in list(range(len(non_set_faces[i2]))):
                    side2 = non_set_faces[i2][j2pos]
                    nrb2 = nrb_faces2[side2 - 1]
                    if ndim == 1:
                        flag = int(
                            np.max(
                                np.abs(
                                    np.asarray(nrb1.coefs).ravel()
                                    - np.asarray(nrb2.coefs).ravel()
                                )
                            )
                            < _TOL
                        )
                    elif ndim == 2:
                        flag, MsgFlag = _compare_sides_univariate(
                            nrb1, nrb2
                        )
                        if flag != 0:
                            ornt = flag
                    else:
                        flag, ornt1, ornt2, MsgFlag = (
                            _compare_sides_bivariate(nrb1, nrb2)
                        )
                    if MsgFlag != 0:
                        _display_warning(
                            MsgFlag, i1 + 1, j1, i2 + 1, side2
                        )
                    if flag != 0:
                        break
                if flag != 0:
                    break

            if flag != 0:
                non_set_faces[i2].remove(side2)
                entry = {
                    "patch1": i1 + 1,
                    "side1": j1,
                    "patch2": i2 + 1,
                    "side2": side2,
                }
                if ndim == 2:
                    entry["ornt"] = ornt
                elif ndim == 3:
                    entry["flag"] = flag
                    entry["ornt1"] = ornt1
                    entry["ornt2"] = ornt2
                interfaces.append(entry)
            else:
                boundary.append(
                    {"nsides": 1, "patches": i1 + 1, "faces": j1}
                )

    return interfaces, boundary
