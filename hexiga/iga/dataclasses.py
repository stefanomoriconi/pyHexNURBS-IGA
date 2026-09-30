"""
hexiga.iga.dataclasses
======================

Python data containers that mirror the MATLAB struct hierarchy used by the
HexIGA IGA layer:

* ``SideAttribs``    -- per-boundary-side attributes (``GeomDATA.bndrsAttrbs``
  entry): ``Srf``, ``pt3D``, ``n3D``, ``mag``, ``isDirichlet``.
* ``SideHandle``     -- per-boundary-side handle (``GeomDATA.bndrsHandles``
  entry): ``isPersistent``.
* ``SideAttribsRow`` -- ``{IDs, m, mx, my, mz}`` block used by
  ``BoundaryConditions.dirichlet / .leading / .floating``.
* ``BoundaryConditions`` -- the three-row container above.
* ``VolumetricSource`` -- ``{f, fx, fy, fz}`` volumetric force spec.
* ``FluidData``      -- input/output container for Stokes (matches the MATLAB
  ``FluidDATA`` struct field set).
* ``ElasticData``    -- input/output container for linear elasticity
  (matches the MATLAB ``ElasticDATA`` struct field set).

Field names are preserved exactly as in the MATLAB code so the converters
read as literal ports.  All arrays are ``np.ndarray``; the ``IDs`` arrays
carry **1-based** side IDs (matching MATLAB) so that the converters can
index ``bndrsAttrbs[bb-1]`` / ``bndrsHandles[bb-1]`` exactly as the
MATLAB loop does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "SideAttribs",
    "SideHandle",
    "SideAttribsRow",
    "BoundaryConditions",
    "VolumetricSource",
    "FluidData",
    "ElasticData",
    "MaxwellData",
]

# ---------------------------------------------------------------------------
# Per-side data (GeomDATA.bndrsAttrbs / bndrsHandles entries)
# ---------------------------------------------------------------------------

@dataclass
class SideAttribs:
    """One entry of ``GeomDATA.bndrsAttrbs`` (MATLAB struct).

    Field semantics (from ``MAIN_gui/HexIGASim.m``):

    * ``Srf``         -- NURBS surface patch for the side (optional).
    * ``pt3D``        -- 3-vector reference point on the side (optional).
    * ``n3D``         -- 3-vector outward normal on the side.
    * ``mag``         -- scalar magnitude of the side condition.
    * ``isDirichlet`` -- ``True`` for a zero-Dirichlet side.
    """

    Srf: object = None
    pt3D: Optional[np.ndarray] = None
    n3D: Optional[np.ndarray] = None
    mag: float = 0.0
    isDirichlet: bool = False

@dataclass
class SideHandle:
    """One entry of ``GeomDATA.bndrsHandles`` (MATLAB struct).

    MATLAB (``getVoidBoundaryHandles``)::

        struct('ArrowHandles',[],'isSelected',[],'isPersistent',[])

    * ``ArrowHandles`` -- GUI arrow-handle id (always ``None`` in the
      headless Python port).
    * ``isSelected``   -- ``True`` when the side is currently selected
      by the user (GUI state).
    * ``isPersistent`` -- ``True`` for a *leading* (persistent) side;
      ``False`` for a *floating* side.
    """

    ArrowHandles: object = None
    isSelected: bool = False
    isPersistent: bool = False

# ---------------------------------------------------------------------------
# BoundaryConditions (FluidDATA / ElasticDATA)
# ---------------------------------------------------------------------------

@dataclass
class SideAttribsRow:
    """One of the ``{IDs, m, mx, my, mz}`` rows of ``BoundaryConditions``.

    Arrays are 1-D.  ``IDs`` are 1-based side indices (MATLAB-compatible).
    """

    IDs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    m: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    mx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    my: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    mz: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

def _new_side_row() -> SideAttribsRow:
    return SideAttribsRow()

@dataclass
class BoundaryConditions:
    """``BoundaryConditions = struct('dirichlet', ..., 'leading', ...,
    'floating', ...)`` from the MATLAB ``getVoid*BoundaryConditions``."""

    dirichlet: SideAttribsRow = field(default_factory=_new_side_row)
    leading: SideAttribsRow = field(default_factory=_new_side_row)
    floating: SideAttribsRow = field(default_factory=_new_side_row)

@dataclass
class VolumetricSource:
    """``VolumetricSource = struct('f', .., 'fx', .., 'fy', .., 'mz', ..)``."""

    f: float = 0.0
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0

# ---------------------------------------------------------------------------
# Top-level DATA containers (MATLAB FluidDATA / ElasticDATA)
# ---------------------------------------------------------------------------

@dataclass
class FluidData:
    """``FluidDATA`` struct (see ``convertGeomDATA2FluidDATA.m``).

    Output fields ``geometry`` / ``space_v`` / ``vel`` / ``space_p`` /
    ``press`` / ``elpsTime`` / ``memFtprt`` are populated by the solver.
    """

    GeomFileName: object = None
    BoundaryConditions: Optional[BoundaryConditions] = None
    VolumetricSource: Optional[VolumetricSource] = None

    geometry: object = None
    space_v: object = None
    vel: Optional[np.ndarray] = None
    space_p: object = None
    press: Optional[np.ndarray] = None
    elpsTime: float = 0.0
    memFtprt: float = 0.0

@dataclass
class ElasticData:
    """``ElasticDATA`` struct (see ``convertGeomDATA2ElasticDATA.m``).

    Output fields ``geometry`` / ``space`` / ``u`` / ``lambda_lame`` /
    ``mu_lame`` / ``elpsTime`` / ``memFtprt`` are populated by the solver.
    """

    GeomFileName: object = None
    BoundaryConditions: Optional[BoundaryConditions] = None
    VolumetricSource: Optional[VolumetricSource] = None

    geometry: object = None
    space: object = None
    u: Optional[np.ndarray] = None
    lambda_lame: object = None
    mu_lame: object = None
    elpsTime: float = 0.0
    memFtprt: float = 0.0

@dataclass
class MaxwellData:
    """``MaxwellDATA`` struct (see ``convertGeomDATA2MaxwellDATA.m``).

    Output fields ``geometry`` / ``space`` / ``eigv`` / ``eigf`` /
    ``elpsTime`` / ``memFtprt`` are populated by the solver wrapper.  (The
    solver also returns ``msh``; the MATLAB wrapper discards it.)
    """

    GeomFileName: object = None
    BoundaryConditions: Optional[BoundaryConditions] = None
    VolumetricSource: Optional[VolumetricSource] = None

    geometry: object = None
    space: object = None
    eigv: Optional[np.ndarray] = None
    eigf: Optional[np.ndarray] = None
    elpsTime: float = 0.0
    memFtprt: float = 0.0
