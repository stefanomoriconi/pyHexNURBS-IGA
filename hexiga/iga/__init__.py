"""
hexiga.iga
==========

Python port of the HexIGA IGA layer (``iga_utils/*.m``):

* :func:`convert_geom_data_2_fluid_data`   <- ``convertGeomDATA2FluidDATA.m``
* :func:`convert_geom_data_2_elastic_data` <- ``convertGeomDATA2ElasticDATA.m``
* :func:`map_boundaries_h_to_h_sbdv`       <- ``mapBoundariesHtoHsbdv.m``
* :func:`mp_solve_stokes_complete`         <- ``mp_solve_stokes_complete.m``
* :func:`mp_solve_linear_elasticity_complete` <- ``mp_solve_linear_elasticity_complete.m``
* :func:`convert_geom_data_2_maxwell_data` <- ``convertGeomDATA2MaxwellDATA.m``
* :func:`mp_solve_maxwell_eig_complete`    <- ``mp_solve_maxwell_eig_complete.m``
* :func:`run_fluid_simulation`             <- ``runFluidSim_GeoPDEsAPI.m`` (renamed)
* :func:`run_elastic_simulation`           <- ``runElastSim_GeoPDEsAPI.m`` (renamed)
* :func:`run_maxwell_simulation`           <- ``runMaxwellEigSim_GeoPDEsAPI.m`` (renamed)

The two solvers are thin orchestrators over an FEM backend (GeoPDEs in
MATLAB).  See :mod:`hexiga.iga.backend` for the protocol and the
:class:`~hexiga.iga.backend.MockGeoPDEsBackend` used by the unit tests.

MATLAB ground-truth sources: ``iga_utils/convertGeomDATA2{Fluid,Elastic}DATA.m``,
``iga_utils/mapBoundariesHtoHsbdv.m``, ``iga_utils/mp_solve_stokes_complete.m``,
``iga_utils/mp_solve_linear_elasticity_complete.m``.
"""

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
from .converters import (
    convert_geom_data_2_fluid_data,
    convert_geom_data_2_elastic_data,
    convert_geom_data_2_maxwell_data,
)
from .boundary_map import map_boundaries_h_to_h_sbdv
from .stokes import mp_solve_stokes_complete
from .elasticity import mp_solve_linear_elasticity_complete
from .maxwell import mp_solve_maxwell_eig_complete
from .simulations import (
    run_fluid_simulation,
    run_elastic_simulation,
    run_maxwell_simulation,
    mp_stats,
)
from .backend import GeoPDEsBackend, MockGeoPDEsBackend
from .geo_load import mp_geo_load, mp_geo_read_nurbs

__all__ = [
    "mp_geo_load",
    "mp_geo_read_nurbs",
    "BoundaryConditions",
    "ElasticData",
    "FluidData",
    "MaxwellData",
    "SideAttribs",
    "SideAttribsRow",
    "SideHandle",
    "VolumetricSource",
    "convert_geom_data_2_fluid_data",
    "convert_geom_data_2_elastic_data",
    "convert_geom_data_2_maxwell_data",
    "map_boundaries_h_to_h_sbdv",
    "mp_solve_stokes_complete",
    "mp_solve_linear_elasticity_complete",
    "mp_solve_maxwell_eig_complete",
    "run_fluid_simulation",
    "run_elastic_simulation",
    "run_maxwell_simulation",
    "mp_stats",
    "GeoPDEsBackend",
    "MockGeoPDEsBackend",
]
