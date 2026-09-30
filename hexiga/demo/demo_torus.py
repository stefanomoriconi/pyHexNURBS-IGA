"""
hexiga.demo.demo_torus
======================

An
Iga-compatible 8-hexahedral torus (linear cuboids degree-elevated to
cubic via ``nrbdegelev(..., [2,2,2])``), with an optional merge to 4
patches along w and an optional
smoothing pass (milestone A4 -- ``mpTurboSmooth``).

Signature: ``mpTorus = demoTorus(varargin)`` with options
``'MERGE'``, ``'VISUALISE'``, ``'SMOOTH'``, ``'TURBOSMOOTH'``,
``'BENCHMARK'``, ``'PUBLISH'`` (defaults ``smooth=True, merge=True,
turbosmooth=False, benchmark=True, visualise=False``).

Notes
-----

* ``genTorus`` is ported verbatim: 8 cuboids built from 9 rotated slices
  (``tht = (hh-1)*(-pi/4)``), each degree-elevated by ``[2,2,2]``.
* ``merge`` pairs patches ``1:2, 3:4, 5:6, 7:8`` along w (dim 3) via
  :func:`hexiga.scaff.mp_merge.mpMerge`.
* The ``smooth`` path calls ``mpTurboSmooth`` -- not yet ported (A4).  It
  is preserved literally and raises :class:`NotImplementedError` with a
  clear message instead of silently skipping.
* ``visualise``/``benchmark`` are matplotlib/``tic-toc`` conveniences and
  are not part of the ported core (they have no solver effect); they are
  accepted and ignored, matching "port the necessary codebase" scope.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.degelev import nrbdegelev
from ..scaff.mp_merge import mpMerge
from .make_cuboid import makeCuboid
from ..smth.mp_turbo_smooth import mp_turbo_smooth

__all__ = ["demoTorus", "genTorus"]

def _get_inputs(inputs: Sequence):
    class OPTs:
        visualise: bool = False
        smooth: bool = True
        merge: bool = True
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
        if key == "MERGE":
            opts.merge = flag
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
            print(f" * demoTorus: Unrecognised Parsed Parameter: {value} - Default Applied.")
        jj += 2
    if opts.publish:
        opts.visualise = True
        opts.smooth = True
        opts.turbosmooth = True
    return opts

def genTorus() -> List[Nrb]:
    """Port of ``genTorus``: 8 quintic torus-hexahedra."""
    A = np.array([1.0, 0.0, 0.5])
    B = np.array([2.0, 0.0, 0.5])
    C = np.array([1.0, 0.0, -0.5])
    D = np.array([2.0, 0.0, -0.5])
    alpha = -np.pi / 4

    # pts: (3, 2, 2, 9) -- homogeneous-free Cartesian corners per slice.
    pts = np.empty((3, 2, 2, 9), dtype=float)
    for hh in range(1, 10):
        tht = (hh - 1) * alpha
        M = np.array(
            [[np.cos(tht), -np.sin(tht), 0.0],
             [np.sin(tht), np.cos(tht), 0.0],
             [0.0, 0.0, 1.0]]
        )
        pts[:, 0, 0, hh - 1] = M @ A
        pts[:, 1, 0, hh - 1] = M @ B
        pts[:, 0, 1, hh - 1] = M @ C
        pts[:, 1, 1, hh - 1] = M @ D

    mpTorus: List[Nrb] = []
    for hh in range(1, 9):
        h0 = hh - 1
        h1 = hh
        hexa = makeCuboid(
            pts[:, 0, 0, h0], pts[:, 1, 0, h0], pts[:, 1, 1, h0], pts[:, 0, 1, h0],
            pts[:, 0, 0, h1], pts[:, 1, 0, h1], pts[:, 1, 1, h1], pts[:, 0, 1, h1],
        )
        mpTorus.append(nrbdegelev(hexa, [2, 2, 2]))
    return mpTorus

def _smooth_or_raise(mpTorus: List[Nrb], turbosmooth: bool) -> List[Nrb]:
    """Port of the ``OPTs.smooth`` block in ``demoTorus.m``."""
    if turbosmooth:
        mpTorus = mp_turbo_smooth(mpTorus, "TURBO", True)
    else:
        mpTorus = mp_turbo_smooth(mpTorus)
    return mpTorus

def demoTorus(*inputs: object) -> List[Nrb]:
    """Build the synthetic torus geometry (see module docstring)."""
    opts = _get_inputs(inputs)

    mpTorus = genTorus()

    if opts.smooth:
        mpTorus = _smooth_or_raise(mpTorus, opts.turbosmooth)

    if opts.merge:
        merged: List[Nrb] = []
        for a, b in ((0, 1), (2, 3), (4, 5), (6, 7)):
            m = mpMerge(mpTorus[a : b + 1], 3)
            if m is None:
                # mpMerge already printed the error line; surface it.
                raise RuntimeError("mpMerge failed for a torus patch pair")
            merged.append(m)
        mpTorus = merged

    return mpTorus
