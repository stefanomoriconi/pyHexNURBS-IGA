"""
hexiga.demo.demo_twist
======================

A single
twisted hexahedral tube swept along a helical axis.  Each cross-section is a
bilinear square (``nrb4surf``), stacked over ``Z`` rings and joined into one
volume NURBS, then degree-elevated to cubic in the two cross-section
directions and knot-inserted at the mid-span.

Signature: ``mpTwist = demoTwist(varargin)`` with options
``'Z'``, ``'THETA'``, ``'SMOOTH'``, ``'TURBOSMOOTH'``, ``'VISUALISE'``,
``'BENCHMARK'``, ``'PUBLISH'`` (defaults ``Z=25, theta=2*pi, smooth=True,
turbosmooth=False, visualise=False, benchmark=True, publish=False``).

Notes
-----

* ``genTwist(Z, theta)`` is ported verbatim: the base ring square is
  ``[[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]]'`` (a 3x4 block), replicated
  ``Z`` times and progressively rotated/translated by the incremental
  rotation ``R``; the ``z`` coordinate of ring ``jj`` is ``(jj-1)/3``.
* Each ring square is built as a bilinear surface via
  :func:`hexiga.nurbs.surf4.nrb4surf` with the corner order
  ``(H(:,1,jj), H(:,2,jj), H(:,4,jj), H(:,3,jj))`` -- note the ``4,3``
  swap -- and the surfaces are stacked along a new 4th axis
  (``cat(4, ...)``).
* ``nrbmak`` with knot vectors
  ``[getUniformKnotVect(2,1), getUniformKnotVect(2,1), getUniformKnotVect(Z,3)]``,
  then ``nrbdegelev(mpHexa, [2,2,0])`` and ``nrbkntins(mpHexa, {0.5,0.5,[]})``.
* The result is a **single** NURBS patch (the ``mpTwist`` list holds one
  element).
* The ``smooth`` path calls ``mpTurboSmooth`` with
  ``'InOutLetsSides',[5,6]`` options -- not yet ported (A4).  Preserved
  literally and raises :class:`NotImplementedError`.
* ``visualise``/``benchmark`` conveniences are not part of the ported core;
  accepted and ignored.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.make import nrbmak
from ..nurbs.surf4 import nrb4surf
from ..nurbs.degelev import nrbdegelev
from ..nurbs.kntins import nrbkntins
from ..scaff.get_uniform_knot_vect import getUniformKnotVect
from ..smth.mp_turbo_smooth import mp_turbo_smooth

__all__ = ["demoTwist", "genTwist"]

def _get_inputs(inputs: object):
    class OPTs:
        Z: int = 25
        theta: float = 2 * np.pi
        smooth: bool = True
        turbosmooth: bool = False
        visualise: bool = False
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
        if key == "Z":
            opts.Z = int(round(abs(float(val))))
        elif key == "THETA":
            opts.theta = float(val)
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
            # NOTE: MATLAB prints the *key* (Inputs{jj}) here, not the value.
            print(
                f" * demoTwist: Unrecognised Parsed Parameter: {inputs[jj]} - Default Applied."
            )
        jj += 2
    if opts.publish:
        opts.visualise = True
        opts.smooth = True
        opts.turbosmooth = True
    return opts

def genTwist(Z: int, theta: float) -> List[Nrb]:
    """Port of ``genTwist``: one twisted hexahedral tube (single patch)."""
    base = np.array(
        [
            [1.0, -1.0, -1.0, 1.0],   # x
            [1.0, 1.0, -1.0, -1.0],   # y
            [0.0, 0.0, 0.0, 0.0],     # z
        ],
        dtype=float,
    )
    H = np.stack([base] * Z, axis=2)  # (3, 4, Z)

    dth = theta / (Z - 1)
    dzh = (theta / 4) / (Z - 1)
    deh = (theta / 3) / (Z - 1)
    R = (
        np.array(
            [[np.cos(dth), -np.sin(dth), 0.0],
             [np.sin(dth), np.cos(dth), 0.0],
             [0.0, 0.0, 1.0]]
        )
        @ np.array(
            [[np.cos(deh), 0.0, np.sin(deh)],
             [0.0, 1.0, 0.0],
             [-np.sin(deh), 0.0, np.cos(deh)]]
        )
        @ np.array(
            [[1.0, 0.0, 0.0],
             [0.0, np.cos(dzh), -np.sin(dzh)],
             [0.0, np.sin(dzh), np.cos(dzh)]]
        )
    )

    for j in range(1, Z):
        H[:, :, j] = R @ H[:, :, j - 1]
        H[2, :, j] = j / 3.0

    Hexa_coefs = np.empty((4, 2, 2, Z), dtype=float)
    for j in range(Z):
        Hsrf = nrb4surf(H[:, 0, j], H[:, 1, j], H[:, 3, j], H[:, 2, j])
        Hexa_coefs[:, :, :, j] = Hsrf.coefs

    mpHexa = nrbmak(
        Hexa_coefs,
        [
            getUniformKnotVect(2, 1),
            getUniformKnotVect(2, 1),
            getUniformKnotVect(Z, 3),
        ],
    )

    mpHexa = nrbdegelev(mpHexa, [2, 2, 0])
    mpHexa = nrbkntins(mpHexa, [0.5, 0.5, np.array([])])

    return [mpHexa]

def _smooth_or_raise(mpTwist: List[Nrb], turbosmooth: bool) -> List[Nrb]:
    """Port of the ``OPTs.smooth`` block in ``demoTwist.m``.

    .. deprecated::
        Kept for backwards-compatible signatures; the implementation now
        delegates to :func:`mp_turbo_smooth` per MATLAB ground truth.
    """
    if turbosmooth:
        mpTwist = mp_turbo_smooth(mpTwist, "INOUTLETSSIDES", [5, 6])
        mpTwist = mp_turbo_smooth(
            mpTwist, "INOUTLETSSIDES", [5, 6],
            "FREEZEIOSIDES", [5, 6], "TURBO", True, "SKIP", True,
        )
    else:
        mpTwist = mp_turbo_smooth(mpTwist, "INOUTLETSSIDES", [5, 6])
    return mpTwist

def demoTwist(*inputs: object) -> List[Nrb]:
    """Build the synthetic twisted-tube geometry (see module docstring)."""
    opts = _get_inputs(inputs)

    mpTwist = genTwist(opts.Z, opts.theta)

    if opts.smooth:
        mpTwist = _smooth_or_raise(mpTwist, opts.turbosmooth)

    return mpTwist
