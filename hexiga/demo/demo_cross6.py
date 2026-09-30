"""
hexiga.demo.demo_cross6
=======================

An
Iga-compatible 7-hexahedral "cross" scaffolding adapted for
6-face-connectivity (voxel-wise) patterns -- one central hexahedron plus six
arms translated along the coordinate axes.

Signature: ``mpCross6 = demoCross6(varargin)`` with options
``'VISUALISE'``, ``'SUBDVITER'``, ``'SMOOTH'``, ``'TURBOSMOOTH'``,
``'BENCHMARK'``, ``'PUBLISH'`` (defaults
``visualise=False, sbdvItr=1, smooth=True, turbosmooth=False,
benchmark=True, publish=False``).

Notes
-----

* ``genCross6`` is ported verbatim: a cubic ``makeIsoHexa([4,4,4], 3)``
  hexahedron ``H`` is subdivided ``sbdvItr`` times *once* (shared), then the
  six arms are its ``nrbtform`` translations by the ``Dxyz`` offsets.  The
  centre patch is prepended via ``cat(2, H, mpCross6)``, so the final order
  is ``[centre, +x, -y, -x, +z, -z]`` (7 patches).
* The ``smooth`` path calls ``mpTurboSmooth`` -- not yet ported (A4).  It
  is preserved literally and raises :class:`NotImplementedError`.
* ``visualise``/``benchmark`` conveniences are not part of the ported core;
  accepted and ignored.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.tform import nrbtform
from ..scaff.make_iso_hexa import makeIsoHexa
from ..smth.mp_turbo_smooth import mp_turbo_smooth
from ..scaff.sbdv_hexa import sbdvHexa

__all__ = ["demoCross6", "genCross6"]

def _get_inputs(inputs: object):
    class OPTs:
        visualise: bool = False
        sbdvItr: int = 1
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
        if key == "VISUALISE":
            opts.visualise = flag
        elif key == "SUBDVITER":
            opts.sbdvItr = int(round(abs(float(val))))
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
                f" * demoCross6: Unrecognised Parsed Parameter: {value} - Default Applied."
            )
        jj += 2
    if opts.publish:
        opts.visualise = True
        opts.smooth = True
        opts.turbosmooth = True
    return opts

def genCross6(opts: object) -> List[Nrb]:
    """Port of ``genCross6``: centre + 6 arms, once-subdivided (shared)."""
    H = makeIsoHexa([4, 4, 4], 3)
    if opts.sbdvItr > 0:
        for _ in range(opts.sbdvItr):
            H = sbdvHexa(H, True)

    Dxyz = np.array(
        [
            [0, 2, 0],
            [2, 0, 0],
            [0, -2, 0],
            [-2, 0, 0],
            [0, 0, 2],
            [0, 0, -2],
        ],
        dtype=float,
    )

    mpCross6: List[Nrb] = []
    for hh in range(Dxyz.shape[0]):
        selM = np.array(
            [
                [1, 0, 0, Dxyz[hh, 0]],
                [0, 1, 0, Dxyz[hh, 1]],
                [0, 0, 1, Dxyz[hh, 2]],
                [0, 0, 0, 1.0],
            ],
            dtype=float,
        )
        mpCross6.append(nrbtform(H, selM))

    # MATLAB: mpCross6 = cat(2, H, mpCross6);  -- centre patch first.
    return [H] + mpCross6

def _smooth_or_raise(mpCross6: List[Nrb], turbosmooth: bool) -> List[Nrb]:
    """Port of the ``OPTs.smooth`` block in ``demoCross6.m``."""
    if turbosmooth:
        mpCross6 = mp_turbo_smooth(mpCross6, "TURBO", True)
    else:
        mpCross6 = mp_turbo_smooth(mpCross6)
    return mpCross6

def demoCross6(*inputs: object) -> List[Nrb]:
    """Build the synthetic cross scaffolding (see module docstring)."""
    opts = _get_inputs(inputs)

    mpCross6 = genCross6(opts)

    if opts.smooth:
        mpCross6 = _smooth_or_raise(mpCross6, opts.turbosmooth)

    return mpCross6
