"""
hexiga.iga.converters
=====================

Faithful Python ports of:

* ``iga_utils/convertGeomDATA2FluidDATA.m``   -> :func:`convert_geom_data_2_fluid_data`
* ``iga_utils/convertGeomDATA2ElasticDATA.m`` -> :func:`convert_geom_data_2_elastic_data`
* ``iga_utils/convertGeomDATA2MaxwellDATA.m`` -> :func:`convert_geom_data_2_maxwell_data`

All three MATLAB functions have **identical boundary-condition logic**; only
the top-level DATA struct field names differ (``FluidDATA`` / ``ElasticDATA``
/ ``MaxwellDATA``).  The ports preserve that structure: a single shared helper
:func:`_build_boundary_conditions` implements the loop, and the three public
functions wrap it into the appropriate dataclass.

Input contract (``GeomDATA``):

* ``mpHexa``       -- multipatch NURBS geometry (list of :class:`Nrb` or a
  pre-loaded geometry object).
* ``bndrsAttrbs``  -- list of :class:`SideAttribs`, one per boundary side
  (1-based index ``bb`` corresponds to ``bndrsAttrbs[bb-1]``).
* ``bndrsHandles`` -- list of :class:`SideHandle`, same ordering.
* ``volSource``    -- a :class:`VolumetricSource`.

Output: a :class:`FluidData` / :class:`ElasticData` ready to be consumed by
the solver (or, in the MATLAB world, by the GeoPDEs API wrappers
``runFluidSim_GeoPDEsAPI`` / ``runElastSim_GeoPDEsAPI``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .dataclasses import (
    BoundaryConditions,
    ElasticData,
    FluidData,
    MaxwellData,
    SideAttribs,
    SideAttribsRow,
    SideHandle,
    VolumetricSource,
)

__all__ = [
    "convert_geom_data_2_fluid_data",
    "convert_geom_data_2_elastic_data",
    "convert_geom_data_2_maxwell_data",
]

# ---------------------------------------------------------------------------
# Shared boundary-condition builder (identical logic in both MATLAB files)
# ---------------------------------------------------------------------------

def _new_side_row() -> SideAttribsRow:
    return SideAttribsRow()

def _append(row: SideAttribsRow, id: int, m: float,
            mx: float, my: float, mz: float) -> None:
    """Append one (ID, m, mx, my, mz) tuple to a SideAttribsRow (``cat(2, ..)``)."""
    row.IDs = np.concatenate([row.IDs, np.array([id], dtype=int)])
    row.m = np.concatenate([row.m, np.array([m], dtype=float)])
    row.mx = np.concatenate([row.mx, np.array([mx], dtype=float)])
    row.my = np.concatenate([row.my, np.array([my], dtype=float)])
    row.mz = np.concatenate([row.mz, np.array([mz], dtype=float)])

def _build_boundary_conditions(bndrs_attrbs: Sequence[SideAttribs],
                               bndrs_handles: Sequence[SideHandle]) -> BoundaryConditions:
    """Port of the MATLAB ``getFluidBoundaryConditions`` /
    ``getElasticBoundaryConditions`` loop (the two are identical)."""

    bc = BoundaryConditions(
        dirichlet=_new_side_row(),
        leading=_new_side_row(),
        floating=_new_side_row(),
    )

    n = len(bndrs_attrbs)
    for bb in range(1, n + 1):  # MATLAB 1-based
        attr = bndrs_attrbs[bb - 1]
        if bool(attr.isDirichlet):
            # Dirichlet side: zero magnitude / zero normal.
            _append(bc.dirichlet, bb, 0.0, 0.0, 0.0, 0.0)
        else:
            handle = bndrs_handles[bb - 1] if bb - 1 < len(bndrs_handles) else SideHandle()
            if bool(getattr(handle, "isPersistent", False)):
                # Leading side: configured magnitude + 3D versor.
                n3d = np.asarray(attr.n3D, dtype=float).ravel()
                if n3d.size != 3:
                    raise ValueError(
                        f"side {bb}: n3D must have 3 components, got {n3d.size}"
                    )
                _append(bc.leading, bb, float(attr.mag),
                        float(n3d[0]), float(n3d[1]), float(n3d[2]))
            else:
                # Floating side: unknown magnitude / versor -> NaNs.
                _append(bc.floating, bb, float("nan"),
                        float("nan"), float("nan"), float("nan"))

    return bc

# ---------------------------------------------------------------------------
# Public converters (literal ports of the two MATLAB files)
# ---------------------------------------------------------------------------

def convert_geom_data_2_fluid_data(GeomDATA) -> FluidData:
    """Port of ``convertGeomDATA2FluidDATA.m``.

    ``GeomDATA`` must expose ``mpHexa``, ``bndrsAttrbs``, ``bndrsHandles``
    and ``volSource`` attributes (dataclass or object with these fields).
    """
    FluidDATA = FluidData()
    FluidDATA.GeomFileName = GeomDATA.mpHexa
    FluidDATA.BoundaryConditions = _build_boundary_conditions(
        GeomDATA.bndrsAttrbs, GeomDATA.bndrsHandles
    )
    FluidDATA.VolumetricSource = GeomDATA.volSource
    return FluidDATA

def convert_geom_data_2_elastic_data(GeomDATA) -> ElasticData:
    """Port of ``convertGeomDATA2ElasticDATA.m``.

    ``GeomDATA`` must expose ``mpHexa``, ``bndrsAttrbs``, ``bndrsHandles``
    and ``volSource`` attributes.
    """
    ElasticDATA = ElasticData()
    ElasticDATA.GeomFileName = GeomDATA.mpHexa
    ElasticDATA.BoundaryConditions = _build_boundary_conditions(
        GeomDATA.bndrsAttrbs, GeomDATA.bndrsHandles
    )
    ElasticDATA.VolumetricSource = GeomDATA.volSource
    return ElasticDATA

def convert_geom_data_2_maxwell_data(GeomDATA) -> MaxwellData:
    """Port of ``convertGeomDATA2MaxwellDATA.m``.

    ``GeomDATA`` must expose ``mpHexa``, ``bndrsAttrbs``, ``bndrsHandles``
    and ``volSource`` attributes (dataclass or object with these fields).
    """
    MaxwellDATA = MaxwellData()
    MaxwellDATA.GeomFileName = GeomDATA.mpHexa
    MaxwellDATA.BoundaryConditions = _build_boundary_conditions(
        GeomDATA.bndrsAttrbs, GeomDATA.bndrsHandles
    )
    MaxwellDATA.VolumetricSource = GeomDATA.volSource
    return MaxwellDATA
