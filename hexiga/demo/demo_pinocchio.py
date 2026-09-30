"""
hexiga.demo.demo_pinocchio
==========================

An
Iga-compatible grafting-based scaffolding for a primary/secondary branch
(a straight main "aorta" with one side branch grafted onto it).

Signature: ``mpPinocchio = demoPinocchio(varargin)`` with options
``'VERSION'``, ``'VISUALISE'``, ``'SMOOTH'``, ``'TURBOSMOOTH'``,
``'BENCHMARK'``, ``'PUBLISH'`` (defaults
``version=1, visualise=False, smooth=True, turbosmooth=False,
benchmark=True, publish=False``).

Notes
-----

* ``VERSION`` selects ``genGraftLarge`` (``version == 1``) or
  ``genGraftSmall``.  Both build the same construction and differ only in
  the graft parameters (number of graft angles, the ``thRim`` index, the
  rim stride ``1:2:end`` vs ``1:4:end`` and the ``mapIDX`` reordering), all
  of which are preserved literally.
* Hexahedral cells are built with
  :func:`hexiga.scaff.get_hexa_coeffs_knots.getHexaCoeffsKnots` +
  :func:`hexiga.nurbs.make.nrbmak`, exactly as in MATLAB, and every patch
  is finally degree-elevated to ``[2 2 2]`` (cubic) by ``nrbdegelev``.
* The branch-interface rotation uses
  :func:`hexiga.scaff.rot_matrix_4_vect.RotMatrix4Vect`; MATLAB applies
  ``R' * col`` to each (3x1) column of the (3,N) point block, which is the
  numpy column-batch form ``R.T @ P`` (equivalent per-column action).
* The ``smooth`` path calls ``mpTurboSmooth(...,'InOutletsSides',[5,6])``
  (optionally nested for ``turbosmooth``) -- not yet ported (A4).  Preserved
  literally and raises :class:`NotImplementedError`.
* ``visualise``/``benchmark`` conveniences are not part of the ported core;
  accepted and ignored.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.make import nrbmak
from ..nurbs.degelev import nrbdegelev
from ..scaff.uvect import uvect
from ..scaff.rot_matrix_4_vect import RotMatrix4Vect
from ..scaff.get_hexa_coeffs_knots import getHexaCoeffsKnots
from ..smth.mp_turbo_smooth import mp_turbo_smooth

__all__ = ["demoPinocchio", "genGraftLarge", "genGraftSmall"]

def _get_inputs(inputs: object):
    class OPTs:
        version: int = 1
        visualise: bool = False
        smooth: bool = True
        turbosmooth: bool = False
        benchmark: bool = True
        publish: bool = False

    opts = OPTs()
    n = len(inputs) if inputs is not None else 0
    jj = 0
    while jj < n:
        key = str(inputs[jj]).upper()
        if jj + 1 < n:
            val = inputs[jj + 1]
            s = str(val)
            flag = (s[0].lower() in ("t", "true")) if s else False
        else:
            flag = False
        if key == "VERSION":
            opts.version = int(float(val))
        elif key == "VISUALISE":
            opts.visualise = flag
        elif key == "SMOOTH":
            opts.smooth = flag
        elif key == "TURBOSMOOTH":
            opts.turbosmooth = flag
        elif key == "BENCHMARK":
            opts.benchmark = flag
        elif key == "PUBLISH":
            opts.publish = flag
        else:
            value = inputs[jj + 1] if jj + 1 < n else ""
            print(
                f" * demoPinocchio1: Unrecognised Parsed Parameter: {value} - Default Applied."
            )
        jj += 2
    if opts.publish:
        opts.visualise = True
        opts.smooth = True
        opts.turbosmooth = True
    return opts

def _append(arr: List[np.ndarray], pt: np.ndarray) -> List[np.ndarray]:
    arr.append(np.asarray(pt, dtype=float))
    return arr

def _gen_graft(
    thGraft: List[int],
    thetaRim: np.ndarray,
    thRim: int,
    mapIDX: List[int],
    rimStride: int,
) -> List[Nrb]:
    """Port of the shared ``genGraftLarge``/``genGraftSmall`` body."""
    mpGraft: List[Nrb] = []

    # Cylindrical parameters
    theta = np.pi / 8
    Rad = 1
    zetas = np.array([0.0, 1.5, 3.0])
    # centers = [zeros(2,3); zetas]  ->  3x3 (columns are the centre points)
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.5, 3.0],
        ],
        dtype=float,
    )

    GraftRad = 0.4
    RimRad = 0.2

    n = int(round(2 * np.pi / theta))  # 16
    thGraft_set = set(thGraft)

    ptsLow: List[np.ndarray] = []
    ptsHig: List[np.ndarray] = []
    ptsMid1: List[np.ndarray] = []
    ptsMid2: List[np.ndarray] = []
    iRimLow: List[np.ndarray] = []
    iRimHig: List[np.ndarray] = []
    grftC = grftE = AorGDir = None

    for th in range(1, n + 1):
        ca = np.cos(theta * (th - 1)) * Rad
        sa = np.sin(theta * (th - 1)) * Rad

        ptLow = np.array([ca, sa, zetas[0]])

        if th in thGraft_set:
            cr = np.cos(thetaRim[th - 1]) * GraftRad
            ptGrftHig = np.array([ca, sa, zetas[1] + cr])
            ptGrftLow = np.array([ca, sa, zetas[1] - cr])
            _append(iRimHig, ptGrftHig)
            _append(iRimLow, ptGrftLow)

            if th == thRim:
                grftC = np.array([ca, sa, zetas[1]])
                grftE = np.array([ca * 2, sa * 2, zetas[1]])
                aortDir = np.array([0.0, 0.0, 1.0])
                grftDir = uvect(np.array([ca, sa, zetas[0]]))
                AorGDir = uvect(np.cross(aortDir, grftDir))
        else:
            ptGrftHig = np.array([ca, sa, zetas[1]])
            ptGrftLow = np.array([ca, sa, zetas[1]])

        ptHig = np.array([ca, sa, zetas[2]])

        _append(ptsLow, ptLow)
        _append(ptsHig, ptHig)
        _append(ptsMid1, ptGrftLow)
        _append(ptsMid2, ptGrftHig)

    ptsLow = np.column_stack(ptsLow)
    ptsHig = np.column_stack(ptsHig)
    ptsMid1 = np.column_stack(ptsMid1)
    ptsMid2 = np.column_stack(ptsMid2)
    iRimLow = np.column_stack(iRimLow)
    iRimHig = np.column_stack(iRimHig)

    # ptsRim / ptsTub  (1-indexed stride ``rimStride``)
    ptsRim = ptsLow[:, 0::rimStride] * RimRad
    ptsTub = ptsLow[:, 0::rimStride] * RimRad

    R = RotMatrix4Vect(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), AorGDir, aortDir)

    # MATLAB: ptsRim(:,pp) = R' * ptsRim(:,pp) + grftC;  (column-batch: R.T @ P)
    ptsRim = R.T @ ptsRim + grftC[:, None]
    ptsTub = R.T @ ptsTub + grftE[:, None]

    # Padding (append the first two columns)
    def _pad(A: np.ndarray) -> np.ndarray:
        return np.column_stack([A, A[:, 0:2]])

    ptsLow = _pad(ptsLow)
    ptsMid1 = _pad(ptsMid1)
    ptsMid2 = _pad(ptsMid2)
    ptsHig = _pad(ptsHig)

    def _add_hex(iA, iB, iC, iD, eA, eB, eC, eD):
        Hcfs, Hknt = getHexaCoeffsKnots(iA, iB, iC, iD, eA, eB, eC, eD)
        H = nrbmak(Hcfs, Hknt)
        mpGraft.append(H)

    # %% Lower Part of the Aorta
    for pp in range(0, ptsLow.shape[1] - 2, 2):
        _add_hex(
            ptsLow[:, pp], ptsLow[:, pp + 1], centers[:, 0], ptsLow[:, pp + 2],
            ptsMid1[:, pp], ptsMid1[:, pp + 1], centers[:, 1], ptsMid1[:, pp + 2],
        )

    # %% Upper part of the Aorta
    for pp in range(0, ptsHig.shape[1] - 2, 2):
        _add_hex(
            ptsMid2[:, pp], ptsMid2[:, pp + 1], centers[:, 1], ptsMid2[:, pp + 2],
            ptsHig[:, pp], ptsHig[:, pp + 1], centers[:, 2], ptsHig[:, pp + 2],
        )

    # %% Branching Interface
    K = iRimLow.shape[1]
    ptsBrc = np.column_stack([iRimHig, iRimLow[:, 1:K - 1][:, ::-1]])
    mapIDX0 = np.array(mapIDX, dtype=int) - 1  # 1-indexed -> 0-indexed
    ptsBrc = ptsBrc[:, mapIDX0]
    ptsBrc = _pad(ptsBrc)
    ptsRim = _pad(ptsRim)

    for pp in range(0, ptsBrc.shape[1] - 2, 2):
        _add_hex(
            ptsBrc[:, pp], ptsBrc[:, pp + 1], centers[:, 1], ptsBrc[:, pp + 2],
            ptsRim[:, pp], ptsRim[:, pp + 1], grftC, ptsRim[:, pp + 2],
        )

    # %% Branching Tube
    ptsTub = _pad(ptsTub)
    for pp in range(0, ptsRim.shape[1] - 2, 2):
        _add_hex(
            ptsRim[:, pp], ptsRim[:, pp + 1], grftC, ptsRim[:, pp + 2],
            ptsTub[:, pp], ptsTub[:, pp + 1], grftE, ptsTub[:, pp + 2],
        )

    for jj in range(len(mpGraft)):
        mpGraft[jj] = nrbdegelev(mpGraft[jj], [2, 2, 2])

    return mpGraft

def genGraftLarge() -> List[Nrb]:
    """Port of ``genGraftLarge`` (default ``version == 1``)."""
    thetaRim = np.linspace(-np.pi / 2, np.pi / 2, 5)
    return _gen_graft(
        thGraft=list(range(1, 6)),
        thetaRim=thetaRim,
        thRim=3,
        mapIDX=[5, 4, 3, 2, 1, 8, 7, 6],
        rimStride=2,
    )

def genGraftSmall() -> List[Nrb]:
    """Port of ``genGraftSmall`` (``version ~= 1``)."""
    thetaRim = np.linspace(-np.pi / 2, np.pi / 2, 3)
    return _gen_graft(
        thGraft=list(range(1, 4)),
        thetaRim=thetaRim,
        thRim=2,
        mapIDX=[3, 2, 1, 4],
        rimStride=4,
    )

def _smooth_or_raise(mpPinocchio: List[Nrb], turbosmooth: bool) -> List[Nrb]:
    """Port of the ``OPTs.smooth`` block in ``demoPinocchio.m``.

    Kept as a helper for signature compatibility; the body now delegates
    to :func:`mp_turbo_smooth` exactly per MATLAB ground truth.
    """
    if turbosmooth:
        mpPinocchio = mp_turbo_smooth(
            mp_turbo_smooth(mpPinocchio, "INOUTLETSSIDES", [5, 6])
        )
        mpPinocchio = mp_turbo_smooth(mpPinocchio, "TURBO", True, "SKIP", True)
    else:
        mpPinocchio = mp_turbo_smooth(
            mp_turbo_smooth(mpPinocchio, "INOUTLETSSIDES", [5, 6])
        )
    return mpPinocchio

def demoPinocchio(*inputs: object) -> List[Nrb]:
    """Build the synthetic grafting scaffolding (see module docstring)."""
    opts = _get_inputs(inputs)

    if opts.version == 1:
        mpPinocchio = genGraftLarge()
    else:
        mpPinocchio = genGraftSmall()

    if opts.smooth:
        mpPinocchio = _smooth_or_raise(mpPinocchio, opts.turbosmooth)

    return mpPinocchio
