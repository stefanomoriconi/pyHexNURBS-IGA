"""
hexiga.demo.demo_stent
======================

A stent geometry
built as a ring of 8 hexahedral patches repeated over ``replicates`` angular
sectors and ``zLevels`` axial levels.

Signature: ``mpStent = demoStentCAD(varargin)`` with options
``'RIN'``, ``'ROUT'``, ``'ZMOD'``, ``'REPLICATES'``, ``'ZLEVELS'``,
``'SMOOTH'``, ``'VISUALISE'``, ``'BENCHMARK'``, ``'PUBLISH'`` (defaults
``Rin=1, Rout=1.1, zMod=3, replicates=5, alphaSections=10 (FIXED), zLevels=3,
smooth=True, visualise=False, benchmark=True, publish=False``).  The numerics
``RIN``/``ROUT``/``ZMOD`` are passed through ``abs(...)``, and
``REPLICATES``/``ZLEVELS`` through ``round(abs(...))`` -- all preserved
literally.

Notes
-----

* ``genStent(Rin,Rout,Zmod,Zlvls,replicates)`` is ported verbatim:
  8 corner-matrix pairs (``H0_0/H0_1`` ... ``H7_0/H7_1``) are transformed
  from cylindrical (rho, angle, z) to Cartesian via
  ``[rho*cos(angle + (rr-1)*alpha); rho*sin(angle + (rr-1)*alpha);
    z + zz*Zmod]`` where ``alpha = 2*pi/replicates`` and
  ``alphaSections = 10`` (FIXED), so ``dalpha = alpha/alphaSections``.
* Each pair becomes one hex via the idiom
  ``nrbmak(cat(4, c0, c1), {k1, k2, [0 0 1 1]})``; then per patch
  ``nrbdegelev(..., [2,2,2])`` and ``nrbkntins(..., {[], [], [0.1 0.9]})``.
* Total: ``8 * replicates * zLevels = 8 * 5 * 3 = 120`` patches.
* **MATLAB BUG (reproduced literally):** ``H6nrb`` is built from
  ``cat(4, H60srf.coefs, H61srf.coefs)`` but the knot vectors are taken from
  ``H70srf`` (a cross-index bug in the original; preserved verbatim).
* The ``smooth`` path calls ``mpTurboSmooth`` -- not yet ported (A4) -- and
  is preserved literally, raising :class:`NotImplementedError`.  The
  non-smooth path returns ``mpHexa`` unchanged.
* The typo message name is ``"gearCAD"`` (a copy/paste artifact of the
  original, preserved verbatim).  ``visualise``/``benchmark`` are MATLAB-side
  conveniences with no solver effect; accepted and ignored.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.degelev import nrbdegelev
from ..nurbs.kntins import nrbkntins
from ..nurbs.make import nrbmak
from ..nurbs.surf4 import nrb4surf
from ..smth.mp_turbo_smooth import mp_turbo_smooth

__all__ = ["demoStentCAD", "genStent"]

def _get_inputs(inputs: Sequence):
    class OPTs:
        Rin: float = 1
        Rout: float = 1.1
        zMod: float = 3
        replicates: int = 5
        alphaSections: int = 10  # FIXED!
        zLevels: int = 3
        smooth: bool = True
        visualise: bool = False
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
            # Scalar (MATLAB: Inputs{jj+1}(1)) -- take first element if array-like.
            scalar = val[0] if hasattr(val, "__len__") and not isinstance(val, str) else val
        else:
            flag = False
            scalar = 0
        if key == "RIN":
            opts.Rin = abs(scalar)
        elif key == "ROUT":
            opts.Rout = abs(scalar)
        elif key == "ZMOD":
            opts.zMod = abs(scalar)
        elif key == "REPLICATES":
            opts.replicates = int(round(abs(scalar)))
        elif key == "ZLEVELS":
            opts.zLevels = int(round(abs(scalar)))
        elif key == "SMOOTH":
            opts.smooth = flag
        elif key == "VISUALISE":
            opts.visualise = flag
        elif key == "BENCHMARK":
            opts.benchmark = flag
        elif key == "PUBLISH":
            opts.publish = flag
        else:
            value = inputs[jj + 1] if jj + 1 < n else ""
            print(f" * gearCAD: Unrecognised Parsed Parameter: {value} - Default Applied.")
        jj += 2
    if opts.publish:
        opts.visualise = True
        opts.smooth = True
    return opts

def _cyl_to_cart(H: np.ndarray, rr: int, alpha: float, dalpha: float, zz: int, Zmod: float) -> np.ndarray:
    """Literal port of the per-column cylindrical -> Cartesian transform.

    ``H`` is a ``(3, 4)`` matrix whose rows are ``(rho, angle, z)`` (the
    pre-transform MATLAB layout).  Returns the transformed ``(3, 4)``
    Cartesian matrix:
    ``x = rho*cos(angle + (rr-1)*alpha)``
    ``y = rho*sin(angle + (rr-1)*alpha)``
    ``z = z + zz*Zmod``
    """
    out = np.empty_like(H)
    for jj in range(H.shape[1]):
        rho = H[0, jj]
        ang = H[1, jj] * dalpha + (rr - 1) * alpha
        z = H[2, jj] + zz * Zmod
        out[0, jj] = rho * math.cos(ang)
        out[1, jj] = rho * math.sin(ang)
        out[2, jj] = z
    return out

def genStent(Rin: float, Rout: float, Zmod: float, Zlvls: int, replicates: int) -> List[Nrb]:
    """Port of ``genStent(Rin,Rout,Zmod,Zlvls,replicates)``.

    Returns ``8 * replicates * Zlvls`` cubic patches (120 by default).
    """
    alpha = 2 * math.pi / replicates
    alphaSections = 10  # FIXED!
    dalpha = alpha / alphaSections

    # Corner-matrix templates in cylindrical (rho, angle, z) form.
    H0_0 = np.array([[Rin, Rout, Rin, Rout], [0, 0, 1, 1], [0, 0, 0, 0]])
    H0_1 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [5, 5, 5, 5],
            [Zmod / 2, Zmod / 2, Zmod / 2 - (Rout - Rin), Zmod / 2 - (Rout - Rin)],
        ]
    )
    H1_0 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [0, 0, 0, 0],
            [0 + (Rout - Rin), 0 + (Rout - Rin), 0, 0],
        ]
    )
    H1_1 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [4, 4, 5, 5],
            [Zmod / 2, Zmod / 2, Zmod / 2, Zmod / 2],
        ]
    )
    H2_0 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [0, 0, 0, 0],
            [Zmod - (Rout - Rin), Zmod - (Rout - Rin), Zmod, Zmod],
        ]
    )
    H2_1 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [4, 4, 5, 5],
            [Zmod / 2, Zmod / 2, Zmod / 2, Zmod / 2],
        ]
    )
    H3_0 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [0, 0, 1, 1],
            [Zmod, Zmod, Zmod, Zmod],
        ]
    )
    H3_1 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [5, 5, 5, 5],
            [Zmod / 2, Zmod / 2, Zmod / 2 + (Rout - Rin), Zmod / 2 + (Rout - Rin)],
        ]
    )
    H4_0 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [5, 5, 6, 6],
            [Zmod / 2, Zmod / 2, Zmod / 2, Zmod / 2],
        ]
    )
    H4_1 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [10, 10, 10, 10],
            [Zmod, Zmod, Zmod - (Rout - Rin), Zmod - (Rout - Rin)],
        ]
    )
    H5_0 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [5, 5, 5, 5],
            [Zmod / 2 + (Rout - Rin), Zmod / 2 + (Rout - Rin), Zmod / 2, Zmod / 2],
        ]
    )
    H5_1 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [9, 9, 10, 10],
            [Zmod, Zmod, Zmod, Zmod],
        ]
    )
    H6_0 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [5, 5, 5, 5],
            [Zmod / 2 - (Rout - Rin), Zmod / 2 - (Rout - Rin), Zmod / 2, Zmod / 2],
        ]
    )
    H6_1 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [9, 9, 10, 10],
            [0, 0, 0, 0],
        ]
    )
    H7_0 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [5, 5, 6, 6],
            [Zmod / 2, Zmod / 2, Zmod / 2, Zmod / 2],
        ]
    )
    H7_1 = np.array(
        [
            [Rin, Rout, Rin, Rout],
            [10, 10, 10, 10],
            [0, 0, 0 + (Rout - Rin), 0 + (Rout - Rin)],
        ]
    )

    mpHexa: List[Nrb] = []
    for zz in range(Zlvls):
        for rr in range(1, replicates + 1):
            H0_0p = _cyl_to_cart(H0_0, rr, alpha, dalpha, zz, Zmod)
            H0_1p = _cyl_to_cart(H0_1, rr, alpha, dalpha, zz, Zmod)
            H1_0p = _cyl_to_cart(H1_0, rr, alpha, dalpha, zz, Zmod)
            H1_1p = _cyl_to_cart(H1_1, rr, alpha, dalpha, zz, Zmod)
            H2_0p = _cyl_to_cart(H2_0, rr, alpha, dalpha, zz, Zmod)
            H2_1p = _cyl_to_cart(H2_1, rr, alpha, dalpha, zz, Zmod)
            H3_0p = _cyl_to_cart(H3_0, rr, alpha, dalpha, zz, Zmod)
            H3_1p = _cyl_to_cart(H3_1, rr, alpha, dalpha, zz, Zmod)
            H4_0p = _cyl_to_cart(H4_0, rr, alpha, dalpha, zz, Zmod)
            H4_1p = _cyl_to_cart(H4_1, rr, alpha, dalpha, zz, Zmod)
            H5_0p = _cyl_to_cart(H5_0, rr, alpha, dalpha, zz, Zmod)
            H5_1p = _cyl_to_cart(H5_1, rr, alpha, dalpha, zz, Zmod)
            H6_0p = _cyl_to_cart(H6_0, rr, alpha, dalpha, zz, Zmod)
            H6_1p = _cyl_to_cart(H6_1, rr, alpha, dalpha, zz, Zmod)
            H7_0p = _cyl_to_cart(H7_0, rr, alpha, dalpha, zz, Zmod)
            H7_1p = _cyl_to_cart(H7_1, rr, alpha, dalpha, zz, Zmod)

            H00srf = nrb4surf(H0_0p[:, 0], H0_0p[:, 1], H0_0p[:, 2], H0_0p[:, 3])
            H01srf = nrb4surf(H0_1p[:, 0], H0_1p[:, 1], H0_1p[:, 2], H0_1p[:, 3])
            H10srf = nrb4surf(H1_0p[:, 0], H1_0p[:, 1], H1_0p[:, 2], H1_0p[:, 3])
            H11srf = nrb4surf(H1_1p[:, 0], H1_1p[:, 1], H1_1p[:, 2], H1_1p[:, 3])
            H20srf = nrb4surf(H2_0p[:, 0], H2_0p[:, 1], H2_0p[:, 2], H2_0p[:, 3])
            H21srf = nrb4surf(H2_1p[:, 0], H2_1p[:, 1], H2_1p[:, 2], H2_1p[:, 3])
            H30srf = nrb4surf(H3_0p[:, 0], H3_0p[:, 1], H3_0p[:, 2], H3_0p[:, 3])
            H31srf = nrb4surf(H3_1p[:, 0], H3_1p[:, 1], H3_1p[:, 2], H3_1p[:, 3])
            H40srf = nrb4surf(H4_0p[:, 0], H4_0p[:, 1], H4_0p[:, 2], H4_0p[:, 3])
            H41srf = nrb4surf(H4_1p[:, 0], H4_1p[:, 1], H4_1p[:, 2], H4_1p[:, 3])
            H50srf = nrb4surf(H5_0p[:, 0], H5_0p[:, 1], H5_0p[:, 2], H5_0p[:, 3])
            H51srf = nrb4surf(H5_1p[:, 0], H5_1p[:, 1], H5_1p[:, 2], H5_1p[:, 3])
            H60srf = nrb4surf(H6_0p[:, 0], H6_0p[:, 1], H6_0p[:, 2], H6_0p[:, 3])
            H61srf = nrb4surf(H6_1p[:, 0], H6_1p[:, 1], H6_1p[:, 2], H6_1p[:, 3])
            H70srf = nrb4surf(H7_0p[:, 0], H7_0p[:, 1], H7_0p[:, 2], H7_0p[:, 3])
            H71srf = nrb4surf(H7_1p[:, 0], H7_1p[:, 1], H7_1p[:, 2], H7_1p[:, 3])

            def _mk(srf0, srf1, knots_from):
                coefs = np.stack([srf0.coefs, srf1.coefs], axis=-1)
                knots = [knots_from.knots[0], knots_from.knots[1], np.array([0.0, 0.0, 1.0, 1.0])]
                return nrbmak(coefs, knots)

            H0nrb = _mk(H00srf, H01srf, H00srf)
            H1nrb = _mk(H10srf, H11srf, H10srf)
            H2nrb = _mk(H20srf, H21srf, H20srf)
            H3nrb = _mk(H30srf, H31srf, H30srf)
            H4nrb = _mk(H40srf, H41srf, H40srf)
            H5nrb = _mk(H50srf, H51srf, H50srf)
            # MATLAB BUG (literal): coefs from H60srf/H61srf, but knots from H70srf.
            H6nrb = _mk(H60srf, H61srf, H70srf)
            H7nrb = _mk(H70srf, H71srf, H70srf)

            mpHexa.extend([H0nrb, H1nrb, H2nrb, H3nrb, H4nrb, H5nrb, H6nrb, H7nrb])

    for jj in range(len(mpHexa)):
        mpHexa[jj] = nrbdegelev(mpHexa[jj], [2, 2, 2])
        mpHexa[jj] = nrbkntins(mpHexa[jj], [np.array([]), np.array([]), np.array([0.1, 0.9])])

    return mpHexa

def demoStentCAD(*inputs) -> List[Nrb]:
    """Port of ``demoStentCAD(varargin)``: 120 cubic stent patches by default.

    The ``smooth`` path is preserved literally and raises
    :class:`NotImplementedError` until ``mpTurboSmooth`` (A4) is ported.
    """
    OPTs = _get_inputs(inputs)

    mpHexa = genStent(OPTs.Rin, OPTs.Rout, OPTs.zMod, OPTs.zLevels, OPTs.replicates)

    if OPTs.smooth:
        mpHexa = mp_turbo_smooth(mpHexa)
    return mpHexa
