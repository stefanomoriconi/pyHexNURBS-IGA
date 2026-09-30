"""
hexiga.demo.demo_tpipe
======================

An IGA-compatible
T-shaped pipe built from linear hexahedral patches (degree-elevated to cubic
via ``nrbdelev(..., [2,2,2])`` with a w-direction knot insertion at
``{0.25, 0.5, 0.75}``).

Signature: ``mpTPipe = demoTPipeCAD(varargin)`` with options
``'RIN'``, ``'ROUT'``, ``'LENGTH'``, ``'MERGE'``, ``'SMOOTH'``,
``'TURBOSMOOTH'``, ``'VISUALISE'``, ``'BENCHMARK'``, ``'PUBLISH'``
(defaults ``Rin=1, Rout=1.15, Length=3, smooth=True, merge=True,
turbosmooth=False, benchmark=True, visualise=False, publish=False``).

Notes
-----

* ``genTPipe`` is ported verbatim: two :func:`getExtrudeBevelQFS` calls
  (wall + outer) feed :func:`makeQFSs2nrbHexa` (core) and
  :func:`makeQFSs2nrbWallHexa` (wall).  All 3 tubes are inlets/outlets
  (``ioFlags = true``), so only wall hexahedra (no caps) are produced.
* **MATLAB BUG (reproduced literally):** the ``'RIN'`` option writes to
  ``OPTs.replicates`` and the ``'ROUT'`` option writes to ``OPTs.zLevels``
  (both unused fields), so overriding ``Rin``/``Rout`` via these options has
  no effect.  The unrecognised-parameter message prints the *value* and uses
  the name ``"demoAsymSocketCAD"`` (typo preserved).
* The ``smooth`` path calls ``mpTurboSmooth`` -- not yet ported (A4).  It is
  preserved literally and raises :class:`NotImplementedError`.
* The non-smooth path selects the last ``8*3 == 24`` patches
  (``mpHexa(end-8*3+1 : end)`` == ``mpHexa[-24:]``).
* ``visualise``/``benchmark`` are MATLAB-side conveniences with no solver
  effect; they are accepted and ignored.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.degelev import nrbdegelev
from ..nurbs.kntins import nrbkntins
from ..scaff.mp_merge import mpMerge
from ..junc.get_extrude_bevel_qfs import getExtrudeBevelQFS
from ..junc.make_qfs2nrb_hexa import makeQFSs2nrbHexa
from ..junc.make_qfs2nrb_wall_hexa import makeQFSs2nrbWallHexa
from ..smth.mp_turbo_smooth import mp_turbo_smooth
from ..smth.mp_turbo_smooth import mp_turbo_smooth

__all__ = ["demoTPipeCAD", "genTPipe"]

def _get_inputs(inputs: Sequence):
    class OPTs:
        Rin: float = 1
        Rout: float = 1.15
        Length: float = 3
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
        val = inputs[jj + 1] if jj + 1 < n else None
        if val is not None:
            s = str(val)
            flag = (s[0].lower() in ("t", "true")) if s else False
        else:
            flag = False
        if key == "RIN":
            # MATLAB BUG (literal): writes to the unused `replicates` field.
            opts.replicates = val[0] if (val is not None and hasattr(val, "__len__")) else val
        elif key == "ROUT":
            # MATLAB BUG (literal): writes to the unused `zLevels` field.
            opts.zLevels = val[0] if (val is not None and hasattr(val, "__len__")) else val
        elif key == "LENGTH":
            opts.Length = val[0] if hasattr(val, "__len__") else val
        elif key == "MERGE":
            opts.merge = flag
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
            # MATLAB typo (literal): name "demoAsymSocketCAD", prints the VALUE.
            print(f" * demoAsymSocketCAD: Unrecognised Parsed Parameter: {value} - Default Applied.")
        jj += 2
    if opts.publish:
        opts.visualise = True
        opts.smooth = True
        opts.turbosmooth = True
    return opts

def genTPipe(Rin: float, Rout: float, PipeLen: float) -> List[Nrb]:
    """Port of ``genTPipe(Rin, Rout, PipeLen)``: the T-pipe patch list."""
    # MATLAB: dirTubes = [-1,0,0; 0,-1,0; 1,0,0]'
    # Each COLUMN is a tube direction:
    #   col0 = [-1,0,0] (tube 1, -x)
    #   col1 = [0,-1,0] (tube 2, -y)
    #   col2 = [1,0,0]  (tube 3, +x)
    # Python: build the same 3x3 (columns are tubes).
    dirTubes = np.array([
        [-1.0,  0.0,  1.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  0.0],
    ], dtype=float)

    QFSpts = np.array([
        [0.0, -Rin, 0.0, Rin, 0.0],
        [0.0, -Rin, Rin, -Rin, 0.0],
        [-Rin, 0.0, 0.0, 0.0, Rin],
    ])

    QFSfcs = np.array([
        [1, 1, 1],
        [4, 2, 3],
        [5, 5, 5],
        [2, 3, 4],
    ], dtype=int)

    idx = np.array([2, 1, 3], dtype=int)

    ioLets = np.ones(idx.size, dtype=bool)
    extrLen = PipeLen * np.ones(idx.size)
    bevelF = np.ones(idx.size)
    dirMag = np.ones(idx.size)

    ebQFSfcs, ebQFSpts, _KAe, idT, edT = getExtrudeBevelQFS(
        dirTubes, QFSfcs, QFSpts, idx, extrLen, bevelF,
        isflat=ioLets, dirMag=dirMag,
    )
    ebQFSfcs2, ebQFSpts2, _KAe2, _idT2, _edT2 = getExtrudeBevelQFS(
        dirTubes, QFSfcs, QFSpts, idx, extrLen, bevelF + (Rout - Rin),
        isflat=ioLets, dirMag=dirMag,
    )

    H = makeQFSs2nrbHexa(QFSfcs, QFSpts, idT, ebQFSfcs, ebQFSpts, edT, idx)
    Hw, _CapFlag = makeQFSs2nrbWallHexa(
        ebQFSfcs, ebQFSpts, QFSfcs, QFSpts, idT,
        ebQFSfcs2, ebQFSpts2, QFSfcs, QFSpts * (1.0 + (Rout - Rin)),
        edT, np.zeros((3, idx.size), dtype=float), idx, np.ones(idx.size, dtype=bool),
    )

    mpHexa = H + Hw

    for hh in range(len(mpHexa)):
        mpHexa[hh] = nrbdegelev(mpHexa[hh], [2, 2, 2])  # CUBIC
        mpHexa[hh] = nrbkntins(mpHexa[hh], [np.array([]), np.array([]), np.array([0.25, 0.5, 0.75])])

    return mpHexa

def demoTPipeCAD(*varargs) -> List[Nrb]:
    """Port of ``demoTPipeCAD`` (see module docstring)."""
    opts = _get_inputs(varargs)

    # (benchmark timing / mpStats are MATLAB-side conveniences; ignored.)

    mpHexa = genTPipe(opts.Rin, opts.Rout, opts.Length)

    if opts.smooth:
        if opts.turbosmooth:
            mpHexaSmth = mp_turbo_smooth(mpHexa, "INOUTLETSSIDES", [5, 6])
            mpTPipe = mpHexaSmth[-8 * 3:]
            mpTPipe = mp_turbo_smooth(
                mpTPipe, "INOUTLETSSIDES", [5, 6],
                "FREEZEIOSIDES", True, "TURBO", True,
                "TURBOCYCLES", 3, "SKIP", True,
            )
        else:
            mpHexaSmth = mp_turbo_smooth(mpHexa, "INOUTLETSSIDES", [5, 6])
            mpTPipe = mpHexaSmth[-8 * 3:]
    else:
        # Non-smooth path: last 8*3 patches.
        mpTPipe = mpHexa[-8 * 3:]

    if opts.merge:
        mpTPipeTemp = list(mpTPipe)
        merged: List[Nrb] = []
        # MATLAB: for hh = 1:2:length(mpTPipeTemp) → hh = 1, 3, 5, ...
        # Python 0-based: 0, 2, 4, ...
        for hh in range(0, len(mpTPipeTemp), 2):
            pair = [mpTPipeTemp[hh], mpTPipeTemp[hh + 1]]
            m = mpMerge(pair, 1)
            merged.append(m)
        mpTPipe = merged

    # visualise is a MATLAB-side convenience; ignored.
    return mpTPipe
