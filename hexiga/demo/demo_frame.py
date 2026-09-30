"""
hexiga.demo.demo_frame
======================

An
Iga-compatible 8-hexahedral "frame" scaffolding adapted for
6-face-connectivity (voxel-wise) patterns.  Each hexahedron is built from a
cubic isotropic hexahedron (``makeIsoHexa([4,4,4], 3)``), translated to its
position, then subdivided once (``sbdvHexa``) per patch.

Signature: ``mpFrame = demoFrame(varargin)`` with options
``'VISUALISE'``, ``'SUBDVITER'``, ``'SMOOTH'``, ``'TURBOSMOOTH'``,
``'BENCHMARK'``, ``'PUBLISH'`` (defaults
``visualise=False, sbdvItr=1, smooth=True, turbosmooth=False,
benchmark=True, publish=False``).

Notes
-----

* ``genFrame`` is ported verbatim: a cubic ``makeIsoHexa([4,4,4], 3)``
  hexahedron is translated by each of the 8 ``Dxy`` offsets via
  :func:`hexiga.nurbs.tform.nrbtform`, then subdivided
  ``sbdvItr`` times with :func:`hexiga.scaff.sbdv_hexa.sbdvHexa(., True)`.
* The MATLAB unrecognised-parameter message prints ``demoDoughnut`` (a
  copy-paste typo in the ground-truth); it is preserved literally.
* The ``smooth`` path calls ``mpTurboSmooth`` -- not yet ported (A4).  It
  is preserved literally and raises :class:`NotImplementedError`.
* ``visualise``/``benchmark`` conveniences (MATLAB figures / ``tic-toc``)
  are not part of the ported core (no solver effect); accepted and ignored.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.tform import nrbtform
from ..scaff.make_iso_hexa import makeIsoHexa
from ..smth.mp_turbo_smooth import mp_turbo_smooth
from ..scaff.sbdv_hexa import sbdvHexa

__all__ = ["demoFrame", "genFrame"]

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
            # NOTE: "demoDoughnut" is a literal typo in the MATLAB ground-truth.
            print(
                f" * demoDoughnut: Unrecognised Parsed Parameter: {value} - Default Applied."
            )
        jj += 2
    if opts.publish:
        opts.visualise = True
        opts.smooth = True
        opts.turbosmooth = True
    return opts

def genFrame(opts: object) -> List[Nrb]:
    """Port of ``genFrame``: 8 translated, once-subdivided hexahedra."""
    H = makeIsoHexa([4, 4, 4], 3)
    Dxy = np.array(
        [
            [0, 2],
            [2, 2],
            [2, 0],
            [2, -2],
            [0, -2],
            [-2, -2],
            [-2, 0],
            [-2, 2],
        ],
        dtype=float,
    )

    mpFrame: List[Nrb] = []
    for hh in range(Dxy.shape[0]):
        selM = np.array(
            [
                [1, 0, 0, Dxy[hh, 0]],
                [0, 1, 0, Dxy[hh, 1]],
                [0, 0, 1, 0.0],
                [0, 0, 0, 1.0],
            ],
            dtype=float,
        )
        hpatch = nrbtform(H, selM)
        if opts.sbdvItr > 0:
            for _ in range(opts.sbdvItr):
                hpatch = sbdvHexa(hpatch, True)
        mpFrame.append(hpatch)
    return mpFrame

def _smooth_or_raise(mpFrame: List[Nrb], turbosmooth: bool) -> List[Nrb]:
    """Port of the ``OPTs.smooth`` block in ``demoFrame.m``."""
    if turbosmooth:
        mpFrame = mp_turbo_smooth(mpFrame, "TURBO", True)
    else:
        mpFrame = mp_turbo_smooth(mpFrame)
    return mpFrame

def demoFrame(*inputs: object) -> List[Nrb]:
    """Build the synthetic frame scaffolding (see module docstring)."""
    opts = _get_inputs(inputs)

    mpFrame = genFrame(opts)

    if opts.smooth:
        mpFrame = _smooth_or_raise(mpFrame, opts.turbosmooth)

    return mpFrame
