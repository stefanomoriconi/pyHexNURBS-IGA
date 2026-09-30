"""
hexiga.demo.demo_egg_lw
=======================

An
Iga-compatible "egg" geometry composed of 4 lumen (Q) hexahedra and 16
wall (W) hexahedra (one lumen + four walls per quadrant, four quadrants
rotated about z by ``-pi/2`` each).

Signature: ``[EggL, EggW] = demoEggLW(varargin)`` with options
``'VISUALISE'``, ``'SMOOTH'``, ``'TURBOSMOOTH'``, ``'BENCHMARK'``,
``'PUBLISH'`` (defaults ``smooth=True, turbosmooth=False,
benchmark=True, visualise=False``).

Notes (literal port, quirks preserved)
--------------------------------------

* The per-patch ``nrbdegelev(..., [2 2 2])`` (cubic -> quintic) is applied
  to *every* of the 20 patches -- this is the immediately portable core and
  is what the tests exercise.
* In ``genEgg`` the MATLAB source assigns ``l1D = loD`` and ``u1D = uoD``
  WITHOUT the per-quadrant rotation ``R`` (a literal quirk of the ground
  truth).  The port mirrors that exactly: ``loD`` / ``uoD`` are used as-is
  rather than ``R * loD`` / ``R * uoD``.
* The explosion transforms (``getExplosionMatrixTransform_new``) are
  visualization-only aids; they are not part of the solver-facing geometry
  and are therefore omitted from the ported core (documented, not stubbed).
* The ``smooth`` path requires ``mpTurboSmooth`` (milestone A4) and raises
  :class:`NotImplementedError` until then, instead of silently skipping.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.degelev import nrbdegelev
from .make_cuboid import makeCuboid
from ..smth.mp_turbo_smooth import mp_turbo_smooth

__all__ = ["demoEggLW", "genEgg"]

def _get_inputs(inputs: Sequence):
    class OPTs:
        visualise: bool = False
        smooth: bool = True
        turbosmooth: bool = False
        benchmark: bool = True
        publish: bool = False

    opts = OPTs()
    n = len(inputs)
    jj = 0
    while jj < n:
        key = str(inputs[jj]).upper()
        if jj + 1 < n:
            s = str(inputs[jj + 1])
            flag = (s[0].lower() in ("t", "true")) if s else False
        else:
            flag = False
        if key == "VISUALISE":
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
            print(f" * demoEggLW: Unrecognised Parsed Parameter: {value} - Default Applied.")
        jj += 2
    if opts.publish:
        opts.visualise = True
        opts.smooth = True
        opts.turbosmooth = True
    return opts

def genEgg() -> Tuple[List[Nrb], List[Nrb]]:
    """Port of ``genEgg``: return ``(HexaQ, HexaW)`` (4 + 16 linear patches)."""
    Ll = 1.5
    Lu = 0.75
    Lz = np.array([0.0, 0.0, 2.0])
    Lzl = 0.5
    Lzu = 0.5
    alpha = -np.pi / 2

    # Lower-Inner A B C D
    liA = np.array([0.0, 1.0, 0.0])
    liB = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0])
    liC = np.array([1.0, 0.0, 0.0])
    liD = np.array([0.0, 0.0, -0.5])

    # Lower-Outer A B C
    loA = Ll * liA
    loB = Ll * liB
    loC = Ll * liC
    loD = liD - np.array([0.0, 0.0, Lzl])

    # Upper-Inner A B C D
    uiA = (Lu * liA) + Lz
    uiB = (Lu * liB) + Lz
    uiC = (Lu * liC) + Lz
    uiD = Lz - liD

    # Upper-Outer A B C D
    uoA = (Lu * loA) + Lz
    uoB = (Lu * loB) + Lz
    uoC = (Lu * loC) + Lz
    uoD = uiD + np.array([0.0, 0.0, Lzu])

    HexaQ: List[Nrb] = []
    HexaW: List[Nrb] = []
    for jj in range(1, 5):
        tht = (jj - 1) * alpha
        R = np.array(
            [[np.cos(tht), -np.sin(tht), 0.0],
             [np.sin(tht), np.cos(tht), 0.0],
             [0.0, 0.0, 1.0]]
        )

        l0A = R @ liA
        l0B = R @ liB
        l0C = R @ liC
        l0D = R @ liD

        l1A = R @ loA
        l1B = R @ loB
        l1C = R @ loC
        l1D = loD  # literal MATLAB quirk: NOT rotated

        u0A = R @ uiA
        u0B = R @ uiB
        u0C = R @ uiC
        u0D = R @ uiD

        u1A = R @ uoA
        u1B = R @ uoB
        u1C = R @ uoC
        u1D = uoD  # literal MATLAB quirk: NOT rotated

        # Quadrant (Lumen)
        Q = makeCuboid(l0A, l0B, l0C, l0D, u0A, u0B, u0C, u0D)
        # Walls (W1 and W2)
        W1 = makeCuboid(l1A, l1B, l0B, l0A, u1A, u1B, u0B, u0A)
        W2 = makeCuboid(l1B, l1C, l0C, l0B, u1B, u1C, u0C, u0B)
        W3 = makeCuboid(l1A, l1B, l1C, l1D, l0A, l0B, l0C, l0D)
        W4 = makeCuboid(u0A, u0B, u0C, u0D, u1A, u1B, u1C, u1D)

        if jj == 1:
            HexaQ = [Q]
            HexaW = [W1, W2, W3, W4]
        else:
            HexaQ = HexaQ + [Q]
            HexaW = HexaW + [W1, W2, W3, W4]

    return HexaQ, HexaW

def _smooth_or_raise(mpHexa: List[Nrb], turbosmooth: bool) -> List[Nrb]:
    """Port of the ``OPTs.smooth`` block in ``demoEggLW.m``."""
    if turbosmooth:
        mpHexa = mp_turbo_smooth(mpHexa, "TURBO", True)
    else:
        mpHexa = mp_turbo_smooth(mpHexa)
    return mpHexa

def demoEggLW(*inputs: object) -> Tuple[List[Nrb], List[Nrb]]:
    """Build the egg lumen/wall geometry, returning ``(EggL, EggW)``.

    ``EggL`` holds the 4 lumen hexahedra and ``EggW`` the 16 wall
    hexahedra.  Mirroring the MATLAB ground truth, the non-smooth path
    returns the original *linear* patches: the per-patch
    ``nrbdegelev(..., [2 2 2])`` (cubic) loop is computed but the result is
    only consumed by the (A4) smoothing branch.  Thus ``demoEggLW('SMOOTH',
    False)`` yields linear (degree-1) geometry.
    """
    opts = _get_inputs(inputs)

    HexaL, HexaW = genEgg()
    mpHexa = HexaL + HexaW
    for hh in range(len(mpHexa)):
        mpHexa[hh] = nrbdegelev(mpHexa[hh], [2, 2, 2])

    if opts.smooth:
        mpHexaSmth = _smooth_or_raise(mpHexa, opts.turbosmooth)
        nL = len(HexaL)
        EggL = mpHexaSmth[0:nL]
        EggW = mpHexaSmth[nL:]
    else:
        EggL = HexaL
        EggW = HexaW

    return EggL, EggW
