"""
hexiga.export
=============

Geometry import / export utilities.

This package contains faithful Python ports of the MATLAB export helpers
shipped with the HexIGA toolkit, in particular ``nrbExportSTL.m``.

Exports
-------
* :func:`nrb_export_stl` -- ``nrbExportSTL.m`` port: export a NURBS surface
  (or the surface of a NURBS solid / multi-patch) to a binary or ASCII STL
  file.

Note: the MATLAB ``nrbExportSTL.m`` source calls local helpers
(``getInputs``, ``getScalarPatchSurfs``, ``getMultiPatchSurfs``,
``evalSurface``, ``getXYZcatFromSrfs``, ``surf2stl``,
``checkFlipVol``).  These are ported here as module-level private helpers
with the same names.
"""

from .stl import nrb_export_stl, surf2stl

__all__ = [
    "nrb_export_stl",
    "surf2stl",
]
