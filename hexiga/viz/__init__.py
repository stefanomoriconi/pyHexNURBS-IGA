"""``hexiga.viz`` -- shared pyVista rendering core for HexIGA.

This subpackage mirrors the five visualisation modes of the MATLAB
``MAIN_gui/DemoSynthCAD.m`` GUIDE app (see that file, lines 624-727) as
headless-safe :class:`pyvista.Plotter` builders:

* :func:`plot_solid_domain`     <- ``plotSolidDomain``     (``nrbelmplotMP`` /
  ``nrbghostplotMP`` + GhostFLAG logic)
* :func:`plot_ctrl_pts`         <- ``plotSolidCtrlPts``    (``nrbghostplotMP``
  PLOTALL + ``nrbctrlcageplotMP``)
* :func:`plot_exploded_param`   <- ``plotExplodedParam``   (``nrbXplot`` SIDESPLOT)
* :func:`plot_reflection_lines` <- ``plotReflectionLines`` (``nrbZplot``)
* :func:`plot_trabecular`       <- ``plotTrabecular``      (``nrbcellplotMP``)

Lower-level mesh-building primitives (shared by the plot functions and by
the GUI apps) live in :mod:`hexiga.viz.mesh`; high-level renderers live in
:mod:`hexiga.viz.plots`.

All functions return a populated :class:`pyvista.Plotter`; call
``plotter.show()`` interactively or ``plotter.write_png("out.png")`` for
headless rendering (``off_screen=True`` is set by default in this package).
"""
from __future__ import annotations

from .mesh import (
    SIDE_COLORS,
    all_side_meshes,
    control_points,
    curve_mesh,
    exterior_boundary_meshes,
    surface_mesh,
)
from .plots import (
    get_explosion_transforms,
    plot_ctrl_pts,
    plot_exploded_param,
    plot_reflection_lines,
    plot_solid_domain,
    plot_trabecular,
)
from .visualise_individual_qfs import visualise_individual_qfs

__all__ = [
    # mesh helpers
    "SIDE_COLORS",
    "surface_mesh",
    "curve_mesh",
    "control_points",
    "exterior_boundary_meshes",
    "all_side_meshes",
    # high-level renderers
    "get_explosion_transforms",
    "plot_solid_domain",
    "plot_ctrl_pts",
    "plot_exploded_param",
    "plot_reflection_lines",
    "plot_trabecular",
    "visualise_individual_qfs",
]
