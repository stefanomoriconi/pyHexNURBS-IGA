"""
hexiga.demo.demo_turbine
========================

A turbine
geometry built from ``replicates`` angular sectors, each containing
4 solid blade patches (S0–S3) and 3 fluid patches (F0–F2) stacked over
``zLevels`` axial levels.

Signature: ``[mpBlades, mpAir] = demoTurbineCAD(varargin)`` with
options ``REPLICATES``, ``ZLEVELS``, ``ZRNG``, ``TWISTFACTOR``,
``TAPERINGRNG``, ``SMOOTH``, ``TURBOSMOOTH``, ``VISUALISE``, ``BENCHMARK``,
``PUBLISH``.

Defaults (MATLAB):
``replicates=12, alphaSections=6 (FIXED), zLevels=4, zRng=[0,5],
twistFactor=2.5, tapRng=[1, 0.65], visualise=False, smooth=True,
turbosmooth=False, benchmark=True, publish=False``.

Notes
-----

* Solid: 4 patches × 12 replicates = 48; Fluid: 3 × 12 = 36.
* z-knots: ``[0 0 0 0 1 1 1 1]`` (4 ctrl pts in z).
* ``nrbdelev(..., [2, 2, 0])`` -- u,v 1→3, w stays linear.
* ``smooth`` path requires ``mpTurboSmooth`` (A4) → :class:`NotImplementedError`.
* Typo message name: ``"turbineCAD"`` (preserved literally).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.degelev import nrbdegelev
from ..nurbs.make import nrbmak
from ..nurbs.surf4 import nrb4surf
from ..smth.mp_turbo_smooth import mp_turbo_smooth

__all__ = ["demoTurbineCAD", "genTurbineDomains"]

def _get_inputs(inputs):
    class OPTs:
        replicates: int = 12
        alpha: float = 0.0  # derived
        alphaSections: int = 6  # FIXED!
        zLevels: int = 4
        zRng: list = [0.0, 5.0]
        twistFactor: float = 2.5
        theta: float = 0.0  # derived
        tapRng: list = [1.0, 0.65]
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
            print(f" * turbineCAD: Unrecognised Parsed Parameter: {value} - Default Applied.")
        jj += 2
    opts.alpha = (2 * math.pi) / opts.replicates
    opts.theta = opts.twistFactor * opts.alpha
    if opts.publish:
        opts.visualise = True
        opts.smooth = True
        opts.turbosmooth = True
    return opts

def _cyl_to_cart(H: np.ndarray, aa: int, alpha: float, dalpha: float,
                 twist_term: float) -> np.ndarray:
    """Cylindrical (rho, angle, z) → Cartesian per column (3×4 matrix)."""
    out = np.empty_like(H)
    for jj in range(H.shape[1]):
        rho = H[0, jj]
        ang = H[1, jj] * dalpha + (aa - 1) * alpha + twist_term
        out[0, jj] = rho * math.cos(ang)
        out[1, jj] = rho * math.sin(ang)
        out[2, jj] = H[2, jj]
    return out

def genTurbineDomains(opts) -> Tuple[List[Nrb], List[Nrb]]:
    """Port of ``genTurbineDomains(OPTs)``.

    Returns ``(mpHexaT, mpHexaF)``:
    - ``mpHexaT``: 4 × replicates solid blade patches
    - ``mpHexaF``: 3 × replicates fluid patches
    """
    alpha = opts.alpha
    alphaSections = opts.alphaSections  # FIXED = 6
    dalpha = alpha / alphaSections

    zLevels = opts.zLevels
    zRng = opts.zRng
    dZ = (zRng[1] - zRng[0]) / zLevels

    theta = opts.theta
    nltwist = sum(range(0, zLevels))
    nltwist_safe = max(nltwist, 1)
    dtheta = theta / zLevels

    tapRng = opts.tapRng
    dtap = (tapRng[1] - tapRng[0]) / zLevels

    mpHexaT: List[Nrb] = []
    mpHexaF: List[Nrb] = []

    for aa in range(1, opts.replicates + 1):
        S_coefs: list = [[] for _ in range(4)]  # S0..S3
        F_coefs: list = [[] for _ in range(3)]  # F0..F2
        last_nrb_S: list = []
        last_nrb_F: list = []

        for tt_1idx in range(1, zLevels + 1):
            # Tapered outer radii
            taper6 = 6 * (tapRng[1] - tt_1idx * dtap)
            taper5 = 5 * (tapRng[1] - tt_1idx * dtap)
            z_val = zRng[0] + dZ * (tt_1idx - 1)
            twist_term = (dtheta / nltwist_safe) * (tt_1idx - 1) ** 2

            # 4 solid tables (3,4)
            S0p = np.array([
                [1.75, taper6, 1.5, taper6],
                [0, 0, 1, 0.5],
                [z_val, z_val, z_val, z_val],
            ])
            S1p = np.array([
                [1.5, taper6, 1.75, taper5],
                [1, 0.5, 2, 1],
                [z_val, z_val, z_val, z_val],
            ])
            S2p = np.array([
                [1.5, 1.75, 1.5, 1.75],
                [1, 2, 7, 6],
                [z_val, z_val, z_val, z_val],
            ])
            S3p = np.array([
                [1.25, 1.5, 1.25, 1.5],
                [1, 1, 7, 7],
                [z_val, z_val, z_val, z_val],
            ])

            # 3 fluid tables (3,4)
            F0p = np.array([
                [1.75, taper5, 1.75, taper6],
                [2, 1, 6, 6],
                [z_val, z_val, z_val, z_val],
            ])
            F1p = np.array([
                [taper5, taper6, taper6, 7.0],
                [1, 0.5, 6, 6],
                [z_val, z_val, z_val, z_val],
            ])
            F2p = np.array([
                [taper6, 7.0, taper6, 7.0],
                [0, 0, 0.5, 6],
                [z_val, z_val, z_val, z_val],
            ])

            # Convert to Cartesian and build nrb4surfs
            S_tables = [S0p, S1p, S2p, S3p]
            F_tables = [F0p, F1p, F2p]

            last_nrb_S = []
            for ii, t in enumerate(S_tables):
                C = _cyl_to_cart(t, aa, alpha, dalpha, twist_term)
                srf = nrb4surf(C[:, 0], C[:, 1], C[:, 2], C[:, 3])
                S_coefs[ii].append(srf.coefs)
                last_nrb_S.append(srf)

            last_nrb_F = []
            for ii, t in enumerate(F_tables):
                C = _cyl_to_cart(t, aa, alpha, dalpha, twist_term)
                srf = nrb4surf(C[:, 0], C[:, 1], C[:, 2], C[:, 3])
                F_coefs[ii].append(srf.coefs)
                last_nrb_F.append(srf)

        # Build solid patches (4 per replicate)
        S0nrb_ref = last_nrb_S[0]
        z_knots = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

        for ii in range(4):
            coefs = np.stack(S_coefs[ii], axis=-1)
            knots = [S0nrb_ref.knots[0], S0nrb_ref.knots[1], z_knots]
            nrb = nrbmak(coefs, knots)
            nrb = nrbdegelev(nrb, [2, 2, 0])
            mpHexaT.append(nrb)

        # Build fluid patches (3 per replicate)
        F0nrb_ref = last_nrb_F[0]

        for ii in range(3):
            coefs = np.stack(F_coefs[ii], axis=-1)
            knots = [F0nrb_ref.knots[0], F0nrb_ref.knots[1], z_knots]
            nrb = nrbmak(coefs, knots)
            nrb = nrbdegelev(nrb, [2, 2, 0])
            mpHexaF.append(nrb)

    return mpHexaT, mpHexaF

def demoTurbineCAD(*inputs) -> Tuple[List[Nrb], List[Nrb]]:
    """Port of ``demoTurbineCAD(varargin)``.

    Returns ``(mpBlades, mpAir)``: 48 blade patches and 36 fluid patches by
    default.  The ``smooth`` path raises :class:`NotImplementedError` until
    ``mpTurboSmooth`` (A4) is ported.
    """
    OPTs = _get_inputs(inputs)

    mpHexaT, mpHexaF = genTurbineDomains(OPTs)

    if OPTs.smooth:
        if OPTs.turbosmooth:
            mpBlades = mp_turbo_smooth(mpHexaT, "INOUTLETSSIDES", [5, 6])
            mpAir = mp_turbo_smooth(mpHexaF, "INOUTLETSSIDES", [5, 6])
            mpBlades = mp_turbo_smooth(
                mpBlades, "INOUTLETSSIDES", [5, 6],
                "FREEZEIOSIDES", True, "TURBO", True, "SKIP", True,
            )
            mpAir = mp_turbo_smooth(
                mpAir, "INOUTLETSSIDES", [5, 6],
                "FREEZEIOSIDES", True, "TURBO", True, "SKIP", True,
            )
        else:
            mpBlades = mp_turbo_smooth(mpHexaT, "INOUTLETSSIDES", [5, 6])
            mpAir = mp_turbo_smooth(mpHexaF, "INOUTLETSSIDES", [5, 6])
    else:
        mpBlades, mpAir = mpHexaT, mpHexaF
    return mpBlades, mpAir
