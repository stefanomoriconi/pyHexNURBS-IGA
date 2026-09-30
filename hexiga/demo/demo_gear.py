"""
hexiga.demo.demo_gear
=====================

A gear geometry
built from ``replicates`` angular sectors, each containing 6 hexahedral
patches (S0–S5) stacked over ``zLevels`` axial levels.

Signature: ``mpGear = demoGearCAD(varargin)`` with options
``REPLICATES``, ``ZLEVELS``, ``ZRNG``, ``TWISTFACTOR``, ``TAPERINGRNG``,
``SMOOTH``, ``TURBOSMOOTH``, ``VISUALISE``, ``BENCHMARK``, ``PUBLISH``.

Defaults (MATLAB):
``replicates=9, alphaSections=4 (FIXED), zLevels=2, zRng=[0,2],
twistFactor=0, tapRng=[1,1], visualise=False, smooth=True,
turbosmooth=False, benchmark=True, publish=False``.

Notes
-----

* 6 patches per replicate × ``replicates`` = 54 by default.
* Taper: the outer rho (2.5) in S0p corners 2&4 and S1p corners 2&4 is
  replaced by ``2.5*(tapRng(2) - tt*dtap)`` (1-indexed tt).
* Twist: angular offset ``dtheta/nltwist * (tt-1)^2`` added to each column's
  cylindrical angle; ``nltwist = sum(0:1:zLevels-1)``.
* ``smooth`` path requires ``mpTurboSmooth`` (A4) → :class:`NotImplementedError`.
* Typo message name: ``"gearCAD"`` (preserved literally).
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.degelev import nrbdegelev
from ..nurbs.make import nrbmak
from ..nurbs.surf4 import nrb4surf
from ..smth.mp_turbo_smooth import mp_turbo_smooth

__all__ = ["demoGearCAD", "genGear"]

def _get_inputs(inputs):
    class OPTs:
        replicates: int = 9
        alpha: float = 0.0  # derived
        alphaSections: int = 4  # FIXED!
        zLevels: int = 2
        zRng: list = [0.0, 2.0]
        twistFactor: float = 0.0
        theta: float = 0.0  # derived
        tapRng: list = [1.0, 1.0]
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
            # MATLAB: Inputs{jj+1}(1) -- first element if array-like
            scalar = val[0] if hasattr(val, "__len__") and not isinstance(val, str) else val
        else:
            flag = False
            scalar = 0
        if key == "REPLICATES":
            opts.replicates = int(scalar)
        elif key == "ZLEVELS":
            opts.zLevels = int(scalar)
        elif key == "ZRNG":
            opts.zRng = [float(val[0]), float(val[1])]
        elif key == "TWISTFACTOR":
            opts.twistFactor = float(scalar)
        elif key == "TAPERINGRNG":
            opts.tapRng = [float(val[0]), float(val[1])]
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
            print(f" * gearCAD: Unrecognised Parsed Parameter: {value} - Default Applied.")
        jj += 2
    # Derived values (MATLAB computes after option loop)
    opts.alpha = (2 * math.pi) / opts.replicates
    opts.theta = opts.twistFactor * opts.alpha
    if opts.publish:
        opts.visualise = True
        opts.smooth = True
        opts.turbosmooth = True
    return opts

def _cyl_to_cart_cyl(H: np.ndarray, aa: int, alpha: float, dalpha: float,
                     twist_term: float) -> np.ndarray:
    """Cylindrical (rho, angle, z) → Cartesian per column.

    ``H`` is a ``(3, 4)`` matrix: row 0 = rho, row 1 = angle (integer index),
    row 2 = z (uniform across columns).  Returns ``(3, 4)`` Cartesian.
    """
    out = np.empty_like(H)
    for jj in range(H.shape[1]):
        rho = H[0, jj]
        ang = H[1, jj] * dalpha + (aa - 1) * alpha + twist_term
        out[0, jj] = rho * math.cos(ang)
        out[1, jj] = rho * math.sin(ang)
        out[2, jj] = H[2, jj]
    return out

def genGear(opts) -> List[Nrb]:
    """Port of ``genGear(OPTs)``: returns ``6 * replicates`` cubic patches.

    Each replicate stacks ``zLevels`` z-slices, builds 6 nrb4surfs per slice,
    concatenates coefficients across z, then applies nrbmak + nrbdelev([2,2,2]).
    """
    alpha = opts.alpha
    alphaSections = opts.alphaSections  # FIXED = 4
    dalpha = alpha / alphaSections

    zLevels = opts.zLevels
    zRng = opts.zRng
    dZ = (zRng[1] - zRng[0]) / zLevels

    theta = opts.theta
    # MATLAB: nltwist = sum(0:1:zLevels-1) = 0 + 1 + ... + (zLevels-1)
    nltwist = sum(range(0, zLevels))
    # Avoid division-by-zero when zLevels == 1 (nltwist would be 0)
    if nltwist == 0:
        nltwist_safe = 1
    else:
        nltwist_safe = nltwist
    dtheta = theta / zLevels

    tapRng = opts.tapRng
    dtap = (tapRng[1] - tapRng[0]) / zLevels

    mpGear: List[Nrb] = []

    for aa in range(1, opts.replicates + 1):
        # Accumulate coefs across z-levels
        S_coefs: list = [[] for _ in range(6)]  # S0..S5

        last_knots = None  # keep reference for nrbmak
        last_nrb: list = []  # last slice's nrb4surf results (for knot ref)

        for tt_1idx in range(1, zLevels + 1):  # tt = 1-indexed in MATLAB
            # Tapered outer rho: MATLAB 2.5*(tapRng(2) - tt*dtap)
            taper_outer = 2.5 * (tapRng[1] - tt_1idx * dtap)
            # z for this level
            z_val = zRng[0] + dZ * (tt_1idx - 1)
            # Twist term
            twist_term = (dtheta / nltwist_safe) * (tt_1idx - 1) ** 2

            # 6 corner tables (3,4) in cylindrical (rho, angle, z)
            S0p = np.array([
                [1.55, taper_outer, 1.5, taper_outer],
                [0, 0, 1, 1],
                [z_val, z_val, z_val, z_val],
            ])
            S1p = np.array([
                [1.5, taper_outer, 1.55, taper_outer],
                [1, 1, 2, 2],
                [z_val, z_val, z_val, z_val],
            ])
            S2p = np.array([
                [1.5, 1.55, 1.5, 1.55],
                [1, 2, 5, 4],
                [z_val, z_val, z_val, z_val],
            ])
            S3p = np.array([
                [1.25, 1.5, 1.25, 1.5],
                [2, 1, 4, 5],
                [z_val, z_val, z_val, z_val],
            ])
            S4p = np.array([
                [0.85, 1.5, 1.0, 1.25],
                [1, 1, 2, 2],
                [z_val, z_val, z_val, z_val],
            ])
            S5p = np.array([
                [1.0, 1.25, 0.85, 1.5],
                [0, 0, 1, 1],
                [z_val, z_val, z_val, z_val],
            ])

            # Convert to Cartesian
            tables = [S0p, S1p, S2p, S3p, S4p, S5p]
            cart = [_cyl_to_cart_cyl(t, aa, alpha, dalpha, twist_term) for t in tables]

            # Build nrb4surfs
            for ii, C in enumerate(cart):
                srf = nrb4surf(C[:, 0], C[:, 1], C[:, 2], C[:, 3])
                S_coefs[ii].append(srf.coefs)
            last_nrb = [nrb4surf(c[:, 0], c[:, 1], c[:, 2], c[:, 3]) for c in cart]

        # Now build 6 hex patches from stacked coefs
        # Use the LAST slice's knots for u,v (MATLAB: S0nrb.knots{1}, S0nrb.knots{2})
        # S0nrb = nrb4surf of the last slice's S0
        S0nrb_ref = last_nrb[0]

        for ii in range(6):
            coefs = np.stack(S_coefs[ii], axis=-1)
            knots = [S0nrb_ref.knots[0], S0nrb_ref.knots[1], np.array([0.0, 0.0, 1.0, 1.0])]
            nrb = nrbmak(coefs, knots)
            nrb = nrbdegelev(nrb, [2, 2, 2])
            mpGear.append(nrb)

    return mpGear

def demoGearCAD(*inputs) -> List[Nrb]:
    """Port of ``demoGearCAD(varargin)``: 54 cubic gear patches by default."""
    OPTs = _get_inputs(inputs)

    mpGear = genGear(OPTs)

    if OPTs.smooth:
        if OPTs.turbosmooth:
            mpGear = mp_turbo_smooth(mpGear, "INOUTLETSSIDES", [5, 6])
            mpGear = mp_turbo_smooth(
                mpGear, "INOUTLETSSIDES", [5, 6],
                "FREEZEIOSIDES", True, "TURBO", True, "SKIP", True,
            )
        else:
            mpGear = mp_turbo_smooth(mpGear, "INOUTLETSSIDES", [5, 6])

    return mpGear
