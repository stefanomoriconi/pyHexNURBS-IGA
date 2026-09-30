"""
hexiga.demo.demo_socket
=======================

An Iga-compatible
socket geometry built from 32 linear hexahedral patches (degree-elevated to
cubic via ``nrbdegelev(..., [2,2,2])``).

Signature: ``mpSocket = demoSocketCAD(varargin)`` with options
``'RIN'``, ``'ROUT'``, ``'SMOOTH'``, ``'TURBOSMOOTH'``, ``'VISUALISE'``,
``'BENCHMARK'``, ``'PUBLISH'`` (defaults ``rin=3, rout=4, smooth=True,
turbosmooth=False, benchmark=True, visualise=False, publish=False``).

Notes
-----

* ``genSocket`` is ported verbatim: 4 base corner-matrix pairs
  ``H1_0..H4_0`` (bottom, ``z=0``) and ``H*_1`` (top, ``z=hgt=rout``); each
  pair becomes one hexahedron via the idiom
  ``nrbdegelev( nrbmak( cat(4, c0, c1), {k1, k2, [0 0 1 1]}) , [2 2 2] )``.
* Patches ``H5..H8`` are ``Tform_mirrorX`` (-y) mirrors of ``H4..H1``.
* Patches ``H9..H16`` are ``Tform_ShiftZ`` (z - rout) shifts of ``H1..H8``.
* Patches ``H17..H24`` are ``Tform_rot90X`` composed after ``Tform_mirrorY``
  (-x) on ``H1..H8``; patches ``H25..H32`` are ``Tform_ShiftY`` (y - rout)
  shifts of ``H17..H24``.  Total: 32 patches.
* **MATLAB BUG (reproduced literally):** the ``'RIN'`` option writes to
  ``OPTs.replicates`` and the ``'ROUT'`` option writes to ``OPTs.zLevels``
  (both unused fields), so overriding ``rin``/``rout`` via these options has
  no effect.  The typo message name is ``"demoSocketCAD"``.
* The ``smooth`` path calls ``mpTurboSmooth`` -- not yet ported (A4).  It is
  preserved literally and raises :class:`NotImplementedError`.
* ``visualise``/``benchmark`` are MATLAB-side conveniences with no solver
  effect; they are accepted and ignored (consistent with the A2 synthetic
  demo ports, which likewise do not invoke ``mpStats``/timing).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.degelev import nrbdegelev
from ..nurbs.make import nrbmak
from ..nurbs.tform import nrbtform
from ..nurbs.surf4 import nrb4surf
from ..smth.mp_turbo_smooth import mp_turbo_smooth

__all__ = ["demoSocketCAD", "genSocket"]

def _get_inputs(inputs: Sequence):
    class OPTs:
        rin: float = 3
        rout: float = 4
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
            val = inputs[jj + 1]
            s = str(val)
            flag = (s[0].lower() in ("t", "true")) if s else False
        else:
            flag = False
        if key == "RIN":
            # MATLAB BUG (literal): writes to the unused `replicates` field.
            opts.replicates = val[0] if hasattr(val, "__len__") else val
        elif key == "ROUT":
            # MATLAB BUG (literal): writes to the unused `zLevels` field.
            opts.zLevels = val[0] if hasattr(val, "__len__") else val
        elif key == "SMOOTH":
            opts.smooth = flag
        elif key == "TURBOSMOOTH":
            opts.turbosmooth = flag
        elif key == "VISUALISE":
            opts.visualise = flag
        elif key == "BENCHMARK":
            opts.benchmark = flag
        elif key == "PUBLISH":
            opts.publish = flag
        else:
            value = inputs[jj + 1] if jj + 1 < n else ""
            print(f" * demoSocketCAD: Unrecognised Parsed Parameter: {value} - Default Applied.")
        jj += 2
    if opts.publish:
        opts.visualise = True
        opts.smooth = True
        opts.turbosmooth = True
    return opts

def _mk_hexa(H0: np.ndarray, H1: np.ndarray) -> Nrb:
    """Port of the repeated ``nrb4surf`` x2 -> ``nrbmak`` -> ``nrbdegelev`` idiom.

    ``H0`` / ``H1`` are ``(3, 4)`` matrices: each column is one Cartesian
    corner (x, y, z).  Matches
    ``nrbdegelev( nrbmak( cat(4, c0, c1), {k1, k2, [0 0 1 1]}) , [2 2 2] )``.
    """
    H0srf = nrb4surf(H0[:, 0], H0[:, 1], H0[:, 2], H0[:, 3])
    H1srf = nrb4surf(H1[:, 0], H1[:, 1], H1[:, 2], H1[:, 3])
    coefs = np.stack([H0srf.coefs, H1srf.coefs], axis=-1)
    knots = [H0srf.knots[0], H0srf.knots[1], np.array([0.0, 0.0, 1.0, 1.0])]
    return nrbdegelev(nrbmak(coefs, knots), [2, 2, 2])

def genSocket(rin: float, rout: float) -> List[Nrb]:
    """Port of ``genSocket(rin, rout)``: 32 cubic socket patches."""
    hgt = rout

    H1_0 = np.array(
        [
            [rin - rout, 0, rin * np.sqrt(2) / 2 - rout, 0],
            [0, 0, rin * np.sqrt(2) / 2, rout],
            [0, 0, 0, 0],
        ]
    )
    H1_1 = H1_0.copy()
    H1_1[2, :] = hgt

    H2_0 = np.array(
        [
            [rin * np.sqrt(2) / 2 - rout, 0, -rout, -rout],
            [rin * np.sqrt(2) / 2, rout, rin, rout],
            [0, 0, 0, 0],
        ]
    )
    H2_1 = H2_0.copy()
    H2_1[2, :] = hgt

    H3_0 = np.array(
        [
            [-rout, -rout, -rin * np.sqrt(2) / 2 - rout, -rout * np.sqrt(2) / 2 - rout],
            [rin, rout, rin * np.sqrt(2) / 2, rout * np.sqrt(2) / 2],
            [0, 0, 0, 0],
        ]
    )
    H3_1 = H3_0.copy()
    H3_1[2, :] = hgt

    H4_0 = np.array(
        [
            [-rin * np.sqrt(2) / 2 - rout, -rout * np.sqrt(2) / 2 - rout, -rin - rout, -2 * rout],
            [rin * np.sqrt(2) / 2, rout * np.sqrt(2) / 2, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    H4_1 = H4_0.copy()
    H4_1[2, :] = hgt

    H1_nrb = _mk_hexa(H1_0, H1_1)
    H2_nrb = _mk_hexa(H2_0, H2_1)
    H3_nrb = _mk_hexa(H3_0, H3_1)
    H4_nrb = _mk_hexa(H4_0, H4_1)

    Tform_mirrorX = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    H5_nrb = nrbtform(H4_nrb, Tform_mirrorX)
    H6_nrb = nrbtform(H3_nrb, Tform_mirrorX)
    H7_nrb = nrbtform(H2_nrb, Tform_mirrorX)
    H8_nrb = nrbtform(H1_nrb, Tform_mirrorX)

    Tform_ShiftZ = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -rout], [0, 0, 0, 1]]
    )
    H9_nrb = nrbtform(H1_nrb, Tform_ShiftZ)
    H10_nrb = nrbtform(H2_nrb, Tform_ShiftZ)
    H11_nrb = nrbtform(H3_nrb, Tform_ShiftZ)
    H12_nrb = nrbtform(H4_nrb, Tform_ShiftZ)
    H13_nrb = nrbtform(H5_nrb, Tform_ShiftZ)
    H14_nrb = nrbtform(H6_nrb, Tform_ShiftZ)
    H15_nrb = nrbtform(H7_nrb, Tform_ShiftZ)
    H16_nrb = nrbtform(H8_nrb, Tform_ShiftZ)

    Tform_mirrorY = np.array(
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    Tform_rot90X = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
    Tform_ShiftY = np.array(
        [[1, 0, 0, 0], [0, 1, 0, -rout], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    H17_nrb = nrbtform(nrbtform(H1_nrb, Tform_mirrorY), Tform_rot90X)
    H18_nrb = nrbtform(nrbtform(H2_nrb, Tform_mirrorY), Tform_rot90X)
    H19_nrb = nrbtform(nrbtform(H3_nrb, Tform_mirrorY), Tform_rot90X)
    H20_nrb = nrbtform(nrbtform(H4_nrb, Tform_mirrorY), Tform_rot90X)
    H21_nrb = nrbtform(nrbtform(H5_nrb, Tform_mirrorY), Tform_rot90X)
    H22_nrb = nrbtform(nrbtform(H6_nrb, Tform_mirrorY), Tform_rot90X)
    H23_nrb = nrbtform(nrbtform(H7_nrb, Tform_mirrorY), Tform_rot90X)
    H24_nrb = nrbtform(nrbtform(H8_nrb, Tform_mirrorY), Tform_rot90X)

    H25_nrb = nrbtform(H17_nrb, Tform_ShiftY)
    H26_nrb = nrbtform(H18_nrb, Tform_ShiftY)
    H27_nrb = nrbtform(H19_nrb, Tform_ShiftY)
    H28_nrb = nrbtform(H20_nrb, Tform_ShiftY)
    H29_nrb = nrbtform(H21_nrb, Tform_ShiftY)
    H30_nrb = nrbtform(H22_nrb, Tform_ShiftY)
    H31_nrb = nrbtform(H23_nrb, Tform_ShiftY)
    H32_nrb = nrbtform(H24_nrb, Tform_ShiftY)

    return [
        H1_nrb, H2_nrb, H3_nrb, H4_nrb, H5_nrb, H6_nrb, H7_nrb, H8_nrb,
        H9_nrb, H10_nrb, H11_nrb, H12_nrb, H13_nrb, H14_nrb, H15_nrb, H16_nrb,
        H17_nrb, H18_nrb, H19_nrb, H20_nrb, H21_nrb, H22_nrb, H23_nrb, H24_nrb,
        H25_nrb, H26_nrb, H27_nrb, H28_nrb, H29_nrb, H30_nrb, H31_nrb, H32_nrb,
    ]

def demoSocketCAD(*inputs) -> List[Nrb]:
    """Port of ``demoSocketCAD(varargin)`` -> 32 cubic socket patches.

    The ``smooth`` path is preserved literally and raises
    :class:`NotImplementedError` until ``mpTurboSmooth`` (A4) is ported.
    """
    OPTs = _get_inputs(inputs)

    mpSocket = genSocket(OPTs.rin, OPTs.rout)

    if OPTs.smooth:
        # MATLAB demoSocketCAD.m lines 12-20, verbatim:
        #   if OPTs.turbosmooth
        #       mpSocket = mpTurboSmooth(mpSocket,'SHELLS',[3,4]);
        #       mpSocket = mpTurboSmooth(mpSocket,'TURBO',true,...
        #                                 'TURBOCYCLES',4,...
        #                                 'SKIP',true );
        #   else
        #       mpSocket = mpTurboSmooth(mpSocket,'SHELLS',[3,4]);
        #   end
        if OPTs.turbosmooth:
            mpSocket = mp_turbo_smooth(mpSocket, "SHELLS", [3, 4])
            mpSocket = mp_turbo_smooth(
                mpSocket,
                "TURBO", True,
                "TURBOCYCLES", 4,
                "SKIP", True,
            )
        else:
            mpSocket = mp_turbo_smooth(mpSocket, "SHELLS", [3, 4])

    return mpSocket
