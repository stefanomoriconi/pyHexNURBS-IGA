"""Port of ``demoPlateCAD.m``.

Faithful, literal MATLAB→Python port. The MATLAB file is the ground truth,
including all preserved quirks:

* The ``otherwise`` branch's disp message references ``Inputs{jj+1}`` (the
  VALUE) rather than ``Inputs{jj}`` (the KEY) — the typo is preserved
  literally.
* ``OPTs.smooth`` defaults to ``True`` in the MATLAB file (unlike the other
  CAD demos). The Python port uses ``None`` as the flag (equivalent to
  MATLAB's logical default) and mirrors the publish-time flip to True.

Smooth path (``OPTs.smooth=True``) delegates to
:func:`hexiga.smth.mp_turbo_smooth.mp_turbo_smooth` on the three
sub-assemblies (``mpPlt``, ``mpRng1+mpTub1``, ``mpRng2+mpTub2``) exactly as
in the MATLAB ground truth.

Public API:
    demoPlateCAD(*opts)
    genPlate(C1, R1in, R1out, C2, R2in, R2out, L, Zp, ZR1, ZR2)
    _get_inputs(varargs)
"""

from __future__ import annotations

import numpy as np
from typing import Any

from ..nurbs import nrb4surf, nrbmak, nrbdegelev, nrbtform, nrbpermute
from ..smth.mp_turbo_smooth import mp_turbo_smooth

def _get_inputs(varargs: list[Any]) -> dict[str, Any]:
    """Port of ``getInputs`` inside ``demoPlateCAD.m``."""

    opts: dict[str, Any] = {
        "C1": np.array([0.0, 0.0]),
        "R1in": 1.5,
        "R1out": 2.25,
        "C2": np.array([0.0, 0.0]) + np.array([-4.0, -4.0]),
        "R2in": 0.5,
        "R2out": 0.85,
        "L": 6.0,
        "Zp": np.array([0.0, 0.35]),
        "ZR1": 1.5,
        "ZR2": 0.75,
        "visualise": False,
        "smooth": True,       # MATLAB default is TRUE (unlike other demos)
        "benchmark": True,
        "publish": False,
    }

    if len(varargs) != 0:
        for jj in range(0, len(varargs), 2):
            key = str(varargs[jj]).upper()
            val = varargs[jj + 1]
            if key == "SMOOTH":
                opts["smooth"] = bool(val)
            elif key == "VISUALISE":
                opts["visualise"] = bool(val)
            elif key == "BENCHMARK":
                opts["benchmark"] = bool(val)
            elif key == "PUBLISH":
                opts["publish"] = bool(val)
            else:
                # BUG preserved verbatim: MATLAB prints the VALUE, not the key.
                print(f" * demoPlateCAD: Unrecognised Parsed Parameter: {val} - Default Applied.")

    if opts["publish"]:
        opts["visualise"] = True
        opts["smooth"] = True

    return opts

def _mk_hexa(srf0, srf1):
    """Literal MATLAB ``cat(4, srf0.coefs, srf1.coefs)`` + ``nrbmak`` with
    z-knots ``[0 0 1 1]``, then ``nrbdegelev(..., [2 2 2])``."""

    coefs = np.stack([srf0.coefs, srf1.coefs], axis=-1)
    k1 = srf0.knots[0]
    k2 = srf0.knots[1]
    kw = np.array([0.0, 0.0, 1.0, 1.0])
    nrb = nrbmak(coefs, [k1, k2, kw])
    nrb = nrbdegelev(nrb, [2, 2, 2])
    return nrb

def genPlate(C1, R1in, R1out, C2, R2in, R2out, L, Zp, ZR1, ZR2):
    """Port of ``genPlate`` inside ``demoPlateCAD.m``.

    Returns:
        (mpPlate, mpRing1, mpRing2, mpTube1, mpTube2) — each a list of
        homogeneous Nrb patches.

        mpPlate : 32 patches (8 × 4 rotations)
        mpRing1 : 8  patches (2 × 4 rotations)
        mpRing2 : 32 patches (8 × 4 rotations)
        mpTube1 : 4  patches (H0nrb × 4 transforms)
        mpTube2 : 16 patches (H10-H13 × 4 transforms each)
    """

    C1 = np.asarray(C1, dtype=float)
    C2 = np.asarray(C2, dtype=float)
    Zp = np.asarray(Zp, dtype=float)
    R1in = float(R1in); R1out = float(R1out)
    R2in = float(R2in); R2out = float(R2out)
    L = float(L)
    ZR1 = float(ZR1); ZR2 = float(ZR2)

    SQ2 = np.sqrt(2.0) / 2.0
    Z1 = Zp[0]; Z2 = Zp[1]

    # Build a 4x4 table from 4 corner points (each is a 3-vector [x,y,z]).
    # This matches MATLAB's H*_0/H*_1 = [x1,x2,x3,x4; y1,y2,y3,y4; z1,z2,z3,z4]'
    # where each point is [x;y;z] and the 4th row (the z-repeat) is the same
    # as row 3.
    def _H(pts):
        """pts: list/tuple of 4 3-vectors (corners). Returns (3,4) array."""
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        return np.array([xs, ys, zs], dtype=float)

    SQ2 = np.sqrt(2.0) / 2.0

    # %% Central Tube
    H0_0 = _H([
        [C1[0],              C1[1]-R1in,         Z1],
        [C1[0],              C1[1],              Z1],
        [C1[0]-R1in*SQ2,     C1[1]-R1in*SQ2,     Z1],
        [C1[0]-R1in,         C1[1],              Z1],
    ])
    H0_1 = _H([
        [C1[0],              C1[1]-R1in,         Z1+ZR1],
        [C1[0],              C1[1],              Z1+ZR1],
        [C1[0]-R1in*SQ2,     C1[1]-R1in*SQ2,     Z1+ZR1],
        [C1[0]-R1in,         C1[1],              Z1+ZR1],
    ])

    # %% Plate Bottom
    H1_0 = _H([
        [C1[0]-R1in*SQ2,     C1[1]-R1in*SQ2,     Z1],
        [C1[0]-R1in,         C1[1],              Z1],
        [C2[0]+R2in*SQ2,     C2[1]+R2in*SQ2,     Z1],
        [C2[0],              C2[1]+R2in,         Z1],
    ])
    H2_0 = _H([
        [C1[0]-R1in,         C1[1],              Z1],
        [C1[0]-L,            C1[1],              Z1],
        [C2[0],              C2[1]+R2in,         Z1],
        [C2[0]-R2in*SQ2,     C2[1]+R2in*SQ2,     Z1],
    ])
    H3_0 = _H([
        [C1[0]-L,            C1[1],              Z1],
        [C1[0]-L,            C2[1],              Z1],
        [C2[0]-R2in*SQ2,     C2[1]+R2in*SQ2,     Z1],
        [C2[0]-R2in,         C2[1],              Z1],
    ])
    H4_0 = _H([
        [C1[0]-L,            C2[1],              Z1],
        [C1[0]-L,            C1[1]-L,            Z1],
        [C2[0]-R2in,         C2[1],              Z1],
        [C2[0]-R2in*SQ2,     C2[1]-R2in*SQ2,     Z1],
    ])

    # %% Plate Top
    H1_1 = _H([
        [C1[0]-R1out*SQ2,    C1[1]-R1out*SQ2,    Z2],
        [C1[0]-R1out,        C1[1],              Z2],
        [C2[0]+R2out*SQ2,    C2[1]+R2out*SQ2,    Z2],
        [C2[0],              C2[1]+R2out,        Z2],
    ])
    H2_1 = _H([
        [C1[0]-R1out,        C1[1],              Z2],
        [C1[0]-L,            C1[1],              Z2],
        [C2[0],              C2[1]+R2out,        Z2],
        [C2[0]-R2out*SQ2,    C2[1]+R2out*SQ2,    Z2],
    ])
    H3_1 = _H([
        [C1[0]-L,            C1[1],              Z2],
        [C1[0]-L,            C2[1],              Z2],
        [C2[0]-R2out*SQ2,    C2[1]+R2out*SQ2,    Z2],
        [C2[0]-R2out,        C2[1],              Z2],
    ])
    H4_1 = _H([
        [C1[0]-L,            C2[1],              Z2],
        [C1[0]-L,            C1[1]-L,            Z2],
        [C2[0]-R2out,        C2[1],              Z2],
        [C2[0]-R2out*SQ2,    C2[1]-R2out*SQ2,    Z2],
    ])

    # %% Central Ring
    Zring1_top = Z2 + (ZR1 - Z2)
    Zring1_bot = Z1 + ZR1
    H5_0 = _H([
        [C1[0]-R1out*SQ2,    C1[1]-R1out*SQ2,    Z2],
        [C1[0]-R1out,        C1[1],              Z2],
        [C1[0]-R1out*SQ2,    C1[1]-R1out*SQ2,    Zring1_top],
        [C1[0]-R1out,        C1[1],              Zring1_top],
    ])
    H5_1 = _H([
        [C1[0]-R1in*SQ2,     C1[1]-R1in*SQ2,     Z1],
        [C1[0]-R1in,         C1[1],              Z1],
        [C1[0]-R1in*SQ2,     C1[1]-R1in*SQ2,     Zring1_bot],
        [C1[0]-R1in,         C1[1],              Zring1_bot],
    ])

    # %% Peripheral Ring(s)
    Zring2_top = Z2 + (ZR2 - Z2)
    Zring2_bot = Z1 + ZR2
    H6_0 = _H([
        [C2[0]+R2out*SQ2,    C2[1]+R2out*SQ2,    Z2],
        [C2[0],              C2[1]+R2out,        Z2],
        [C2[0]+R2out*SQ2,    C2[1]+R2out*SQ2,    Zring2_top],
        [C2[0],              C2[1]+R2out,        Zring2_top],
    ])
    H6_1 = _H([
        [C2[0]+R2in*SQ2,     C2[1]+R2in*SQ2,     Z1],
        [C2[0],              C2[1]+R2in,         Z1],
        [C2[0]+R2in*SQ2,     C2[1]+R2in*SQ2,     Zring2_bot],
        [C2[0],              C2[1]+R2in,         Zring2_bot],
    ])
    H7_0 = _H([
        [C2[0],              C2[1]+R2out,        Z2],
        [C2[0]-R2out*SQ2,    C2[1]+R2out*SQ2,    Z2],
        [C2[0],              C2[1]+R2out,        Zring2_top],
        [C2[0]-R2out*SQ2,    C2[1]+R2out*SQ2,    Zring2_top],
    ])
    H7_1 = _H([
        [C2[0],              C2[1]+R2in,         Z1],
        [C2[0]-R2in*SQ2,     C2[1]+R2in*SQ2,     Z1],
        [C2[0],              C2[1]+R2in,         Zring2_bot],
        [C2[0]-R2in*SQ2,     C2[1]+R2in*SQ2,     Zring2_bot],
    ])
    H8_0 = _H([
        [C2[0]-R2out*SQ2,    C2[1]+R2out*SQ2,    Z2],
        [C2[0]-R2out,        C2[1],              Z2],
        [C2[0]-R2out*SQ2,    C2[1]+R2out*SQ2,    Zring2_top],
        [C2[0]-R2out,        C2[1],              Zring2_top],
    ])
    H8_1 = _H([
        [C2[0]-R2in*SQ2,     C2[1]+R2in*SQ2,     Z1],
        [C2[0]-R2in,         C2[1],              Z1],
        [C2[0]-R2in*SQ2,     C2[1]+R2in*SQ2,     Zring2_bot],
        [C2[0]-R2in,         C2[1],              Zring2_bot],
    ])
    H9_0 = _H([
        [C2[0]-R2out,        C2[1],              Z2],
        [C2[0]-R2out*SQ2,    C2[1]-R2out*SQ2,    Z2],
        [C2[0]-R2out,        C2[1],              Zring2_top],
        [C2[0]-R2out*SQ2,    C2[1]-R2out*SQ2,    Zring2_top],
    ])
    H9_1 = _H([
        [C2[0]-R2in,         C2[1],              Z1],
        [C2[0]-R2in*SQ2,     C2[1]-R2in*SQ2,     Z1],
        [C2[0]-R2in,         C2[1],              Zring2_bot],
        [C2[0]-R2in*SQ2,     C2[1]-R2in*SQ2,     Zring2_bot],
    ])

    # %% Rings Tubes
    H10_0 = _H([
        [C2[0],              C2[1],              Z1],
        [C2[0],              C2[1]+R2in,         Z1],
        [C2[0]-R2in,         C2[1],              Z1],
        [C2[0]-R2in*SQ2,     C2[1]+R2in*SQ2,     Z1],
    ])
    H10_1 = _H([
        [C2[0],              C2[1],              Z1+ZR2],
        [C2[0],              C2[1]+R2in,         Z1+ZR2],
        [C2[0]-R2in,         C2[1],              Z1+ZR2],
        [C2[0]-R2in*SQ2,     C2[1]+R2in*SQ2,     Z1+ZR2],
    ])
    H11_0 = _H([
        [C2[0],              C2[1],              Z1],
        [C2[0]-R2in,         C2[1],              Z1],
        [C2[0],              C2[1]-R2in,         Z1],
        [C2[0]-R2in*SQ2,     C2[1]-R2in*SQ2,     Z1],
    ])
    H11_1 = _H([
        [C2[0],              C2[1],              Z1+ZR2],
        [C2[0]-R2in,         C2[1],              Z1+ZR2],
        [C2[0],              C2[1]-R2in,         Z1+ZR2],
        [C2[0]-R2in*SQ2,     C2[1]-R2in*SQ2,     Z1+ZR2],
    ])
    H12_0 = _H([
        [C2[0],              C2[1],              Z1],
        [C2[0],              C2[1]-R2in,         Z1],
        [C2[0]+R2in,         C2[1],              Z1],
        [C2[0]+R2in*SQ2,     C2[1]-R2in*SQ2,     Z1],
    ])
    H12_1 = _H([
        [C2[0],              C2[1],              Z1+ZR2],
        [C2[0],              C2[1]-R2in,         Z1+ZR2],
        [C2[0]+R2in,         C2[1],              Z1+ZR2],
        [C2[0]+R2in*SQ2,     C2[1]-R2in*SQ2,     Z1+ZR2],
    ])
    H13_0 = _H([
        [C2[0],              C2[1],              Z1],
        [C2[0]+R2in,         C2[1],              Z1],
        [C2[0],              C2[1]+R2in,         Z1],
        [C2[0]+R2in*SQ2,     C2[1]+R2in*SQ2,     Z1],
    ])
    H13_1 = _H([
        [C2[0],              C2[1],              Z1+ZR2],
        [C2[0]+R2in,         C2[1],              Z1+ZR2],
        [C2[0],              C2[1]+R2in,         Z1+ZR2],
        [C2[0]+R2in*SQ2,     C2[1]+R2in*SQ2,     Z1+ZR2],
    ])

    # %% Surfaces & multi-patch Solids Plate
    H0_0srf = nrb4surf(H0_0[:, 0], H0_0[:, 1], H0_0[:, 2], H0_0[:, 3])
    H0_1srf = nrb4surf(H0_1[:, 0], H0_1[:, 1], H0_1[:, 2], H0_1[:, 3])
    H0nrb = _mk_hexa(H0_0srf, H0_1srf)

    H1_0srf = nrb4surf(H1_0[:, 0], H1_0[:, 1], H1_0[:, 2], H1_0[:, 3])
    H1_1srf = nrb4surf(H1_1[:, 0], H1_1[:, 1], H1_1[:, 2], H1_1[:, 3])
    H1nrb = _mk_hexa(H1_0srf, H1_1srf)

    H2_0srf = nrb4surf(H2_0[:, 0], H2_0[:, 1], H2_0[:, 2], H2_0[:, 3])
    H2_1srf = nrb4surf(H2_1[:, 0], H2_1[:, 1], H2_1[:, 2], H2_1[:, 3])
    H2nrb = _mk_hexa(H2_0srf, H2_1srf)

    H3_0srf = nrb4surf(H3_0[:, 0], H3_0[:, 1], H3_0[:, 2], H3_0[:, 3])
    H3_1srf = nrb4surf(H3_1[:, 0], H3_1[:, 1], H3_1[:, 2], H3_1[:, 3])
    H3nrb = _mk_hexa(H3_0srf, H3_1srf)

    H4_0srf = nrb4surf(H4_0[:, 0], H4_0[:, 1], H4_0[:, 2], H4_0[:, 3])
    H4_1srf = nrb4surf(H4_1[:, 0], H4_1[:, 1], H4_1[:, 2], H4_1[:, 3])
    H4nrb = _mk_hexa(H4_0srf, H4_1srf)

    # %% Surfaces & multi-patch Solids Central Ring
    H5_0srf = nrb4surf(H5_0[:, 0], H5_0[:, 1], H5_0[:, 2], H5_0[:, 3])
    H5_1srf = nrb4surf(H5_1[:, 0], H5_1[:, 1], H5_1[:, 2], H5_1[:, 3])
    H5nrb = _mk_hexa(H5_0srf, H5_1srf)
    H5nrb = nrbpermute(H5nrb, [1, 3, 2])

    # %% Surfaces & multi-patch Solids Peripheral Ring(s)
    H6_0srf = nrb4surf(H6_0[:, 0], H6_0[:, 1], H6_0[:, 2], H6_0[:, 3])
    H6_1srf = nrb4surf(H6_1[:, 0], H6_1[:, 1], H6_1[:, 2], H6_1[:, 3])
    H6nrb = _mk_hexa(H6_0srf, H6_1srf)
    H6nrb = nrbpermute(H6nrb, [1, 3, 2])

    H7_0srf = nrb4surf(H7_0[:, 0], H7_0[:, 1], H7_0[:, 2], H7_0[:, 3])
    H7_1srf = nrb4surf(H7_1[:, 0], H7_1[:, 1], H7_1[:, 2], H7_1[:, 3])
    H7nrb = _mk_hexa(H7_0srf, H7_1srf)
    H7nrb = nrbpermute(H7nrb, [1, 3, 2])

    H8_0srf = nrb4surf(H8_0[:, 0], H8_0[:, 1], H8_0[:, 2], H8_0[:, 3])
    H8_1srf = nrb4surf(H8_1[:, 0], H8_1[:, 1], H8_1[:, 2], H8_1[:, 3])
    H8nrb = _mk_hexa(H8_0srf, H8_1srf)
    H8nrb = nrbpermute(H8nrb, [1, 3, 2])

    H9_0srf = nrb4surf(H9_0[:, 0], H9_0[:, 1], H9_0[:, 2], H9_0[:, 3])
    H9_1srf = nrb4surf(H9_1[:, 0], H9_1[:, 1], H9_1[:, 2], H9_1[:, 3])
    H9nrb = _mk_hexa(H9_0srf, H9_1srf)
    H9nrb = nrbpermute(H9nrb, [1, 3, 2])

    # %% Rings Tubes
    H10_0srf = nrb4surf(H10_0[:, 0], H10_0[:, 1], H10_0[:, 2], H10_0[:, 3])
    H10_1srf = nrb4surf(H10_1[:, 0], H10_1[:, 1], H10_1[:, 2], H10_1[:, 3])
    H10nrb = _mk_hexa(H10_0srf, H10_1srf)

    H11_0srf = nrb4surf(H11_0[:, 0], H11_0[:, 1], H11_0[:, 2], H11_0[:, 3])
    H11_1srf = nrb4surf(H11_1[:, 0], H11_1[:, 1], H11_1[:, 2], H11_1[:, 3])
    H11nrb = _mk_hexa(H11_0srf, H11_1srf)

    H12_0srf = nrb4surf(H12_0[:, 0], H12_0[:, 1], H12_0[:, 2], H12_0[:, 3])
    H12_1srf = nrb4surf(H12_1[:, 0], H12_1[:, 1], H12_1[:, 2], H12_1[:, 3])
    H12nrb = _mk_hexa(H12_0srf, H12_1srf)

    H13_0srf = nrb4surf(H13_0[:, 0], H13_0[:, 1], H13_0[:, 2], H13_0[:, 3])
    H13_1srf = nrb4surf(H13_1[:, 0], H13_1[:, 1], H13_1[:, 2], H13_1[:, 3])
    H13nrb = _mk_hexa(H13_0srf, H13_1srf)

    # %% Rigid Transforms
    Tform_rot90Z = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    Tform_flipY = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    Tform_flipX = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    # %% Replicating Module for Full-Plate
    mpModule1 = [H1nrb, H2nrb, H3nrb, H4nrb,
                 H5nrb,
                 H6nrb, H7nrb, H8nrb, H9nrb]  # 9 patches

    mpModule2 = [nrbtform(p, Tform_flipY) for p in mpModule1]
    mpModule000 = mpModule1 + mpModule2  # 18 patches

    pltIDX  = [0, 1, 2, 3, 9, 10, 11, 12]     # MATLAB 1-based [1,2,3,4,10,11,12,13]
    rng1IDX = [4, 13]                          # [5,14]
    rng2IDX = [5, 6, 7, 8, 14, 15, 16, 17]     # [6,7,8,9,15,16,17,18]

    mpModule090 = [nrbtform(p, Tform_rot90Z) for p in mpModule000]
    mpModule180 = [nrbtform(p, Tform_flipX) for p in mpModule000]
    mpModule270 = [nrbtform(p, Tform_flipY) for p in mpModule090]

    mpPlate = ([mpModule000[i] for i in pltIDX] +
               [mpModule090[i] for i in pltIDX] +
               [mpModule180[i] for i in pltIDX] +
               [mpModule270[i] for i in pltIDX])

    mpRing1 = ([mpModule000[i] for i in rng1IDX] +
               [mpModule090[i] for i in rng1IDX] +
               [mpModule180[i] for i in rng1IDX] +
               [mpModule270[i] for i in rng1IDX])

    mpRing2 = ([mpModule000[i] for i in rng2IDX] +
               [mpModule090[i] for i in rng2IDX] +
               [mpModule180[i] for i in rng2IDX] +
               [mpModule270[i] for i in rng2IDX])

    mpTube1 = [H0nrb,
               nrbtform(H0nrb, Tform_flipY),
               nrbtform(H0nrb, Tform_flipX),
               nrbtform(nrbtform(H0nrb, Tform_flipX), Tform_flipY)]

    mpModtube2 = [H10nrb, H11nrb, H12nrb, H13nrb]
    mpTube2 = list(mpModtube2)
    for p in mpModtube2:
        mpTube2.append(nrbtform(p, Tform_flipX))
        mpTube2.append(nrbtform(p, Tform_flipY))
        mpTube2.append(nrbtform(nrbtform(p, Tform_flipX), Tform_flipY))

    return mpPlate, mpRing1, mpRing2, mpTube1, mpTube2

def demoPlateCAD(*opts):
    """Top-level driver mirroring ``demoPlateCAD.m``.

    The ``smooth`` branch delegates to
    :func:`hexiga.smth.mp_turbo_smooth.mp_turbo_smooth` on the three
    sub-assemblies (``mpPlt``, ``mpRng1+mpTub1``, ``mpRng2+mpTub2``) and
    concatenates the smoothed patches, exactly as in the MATLAB ground
    truth.
    """

    OPTs = _get_inputs(list(opts))

    (mpPlt, mpRng1, mpRng2, mpTub1, mpTub2) = genPlate(
        OPTs["C1"], OPTs["R1in"], OPTs["R1out"],
        OPTs["C2"], OPTs["R2in"], OPTs["R2out"],
        OPTs["L"], OPTs["Zp"], OPTs["ZR1"], OPTs["ZR2"],
    )

    if OPTs["smooth"]:
        mpPltSmth  = mp_turbo_smooth(mpPlt,  "INOUTLETSSIDES", [5, 6])
        mpRng1Smth = mp_turbo_smooth(mpRng1 + mpTub1, "INOUTLETSSIDES", [5, 6])
        mpRng2Smth = mp_turbo_smooth(mpRng2 + mpTub2, "INOUTLETSSIDES", [5, 6])
        # MATLAB: cat(2,mpPltSmth,mpRng1Smth(1:end-length(mpTub1)),mpRng2Smth(1:end-length(mpTub2)))
        mpPlate = (mpPltSmth
                   + mpRng1Smth[:len(mpRng1)]
                   + mpRng2Smth[:len(mpRng2)])
    else:
        mpPlate = mpPlt + mpRng1 + mpRng2

    # (visualise branch is a no-op in the Python port; the user can call
    # hexiga.plot.mpVisualise themselves if desired.)

    return mpPlate
