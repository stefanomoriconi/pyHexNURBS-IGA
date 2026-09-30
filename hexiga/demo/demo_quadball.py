"""
hexiga.demo.demo_quadball
=========================

A single-hexahedron "quadball" (an isoparametric cube whose corners are
degree-elevated so the boundary is a rounded ball).

Signature: ``mpQuadball = demoQuadball(varargin)`` with options
``'VISUALISE'``, ``'SMOOTH'``, ``'TURBOSMOOTH'``, ``'BENCHMARK'``,
``'PUBLISH'`` (defaults ``smooth=True, turbosmooth=False,
benchmark=True, visualise=False``).

Non-smooth path (the immediately portable core)::

    H = makeIsoHexa([3,3,3],[2,2,2]);
    mpQuadball = nrbdegelev(H, [1 1 1]);

Smooth path (milestone A4 -- ``mpTurboSmooth``)::

    mpQuadball = mpTurboSmooth(
                    mpTurboSmooth(
                        nrbdegelev(mpTurboSmooth(H), [1 1 1]) ), ...
                        'TURBO', true);
    S = sqrt(3)*eye(4); S(4,4)=1;
    mpQuadball = nrbtform(mpQuadball, S);

The smooth path is preserved literally and raises :class:`NotImplementedError`
until A4 lands.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.degelev import nrbdegelev
from ..nurbs.tform import nrbtform
from ..scaff.make_iso_hexa import makeIsoHexa
from ..smth.mp_turbo_smooth import mp_turbo_smooth

__all__ = ["demoQuadball"]

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
            # Literal MATLAB quirk: prints Inputs{jj+1} (the value), not the
            # flag name, and mislabels the prefix as 'demoSpheroid'.
            print(f" * demoSpheroid: Unrecognised Parsed Parameter: {value} - Default Applied.")
        jj += 2
    if opts.publish:
        opts.visualise = True
        opts.smooth = True
        opts.turbosmooth = True
    return opts

def demoQuadball(*inputs: object) -> List[Nrb]:
    """Build the synthetic quadball geometry (see module docstring).

    Returns a list of length 1 (the single hexahedral patch), matching the
    MATLAB behaviour that a single-patch multi-patch geometry is returned
    as a 1-element array.
    """
    opts = _get_inputs(inputs)

    H = makeIsoHexa([3, 3, 3], [2, 2, 2])

    # MATLAB treats ``H`` as a 1-element multi-patch array; ``mp_turbo_smooth``
    # operates on a list, so we wrap the single patch for the smoothing calls
    # and unwrap it for the per-patch ``nrbdegelev``/``nrbtform`` steps.
    if opts.smooth:
        if opts.turbosmooth:
            mpQuad = mp_turbo_smooth(
                [nrbdegelev(mp_turbo_smooth([H])[0], [1, 1, 1])],
                "TURBO", True,
            )
        else:
            mpQuad = mp_turbo_smooth(
                [nrbdegelev(mp_turbo_smooth([H])[0], [1, 1, 1])]
            )
        S = np.sqrt(3.0) * np.eye(4)
        S[3, 3] = 1.0
        mpQuadball = nrbtform(mpQuad[0], S)
    else:
        mpQuadball = nrbdegelev(H, [1, 1, 1])

    return [mpQuadball]
