"""
hexiga
======

Self-contained Python toolkit for solid NURBS conforming scaffolding and
isogeometric analysis (IGA), implementing the approach described in
https://arxiv.org/abs/2206.04421.

The package is organised in three layers:

* :mod:`hexiga.nurbs`  -- NURBS / B-spline geometry core, with an optional
  compiled C accelerator for the performance-critical kernels.
* :mod:`hexiga.iga`    -- isogeometric finite element analysis (multipatch
  geometry, spaces, operators, Stokes / elasticity solvers).
  *Milestone M2+ -- not yet implemented.*
* :mod:`hexiga.toolkit`-- the HexIGA application layer (N-junction
  generation, scaffold fitting, smoothing, tube utilities, I/O, CLI and
  later the high-fidelity GUI).  *Milestone M3+ -- not yet implemented.*

Design notes
------------
* The package is self-contained: it has no external MATLAB dependency at
  run time.
* Performance-critical kernels live behind :mod:`hexiga.nurbs._backend`,
  a thin seam that selects between a pure-NumPy reference implementation
  and an optional compiled C / OpenMP implementation without touching the
  high-level API.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
