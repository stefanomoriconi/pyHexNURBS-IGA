"""
hexiga.nurbs._backend
=====================

Performance-backend seam.

The heavy-weight NURBS kernels (span finding, basis evaluation, knot
insertion, degree elevation, derivatives) are implemented in pure NumPy in
:mod:`hexiga.nurbs._kernels`.  This module is the single point through which
the rest of the package reaches those kernels, so that a later compiled
backend (C with OpenMP parallelism) can be dropped in *without* touching any
call-site.

Activation of a compiled backend is opt-in and environment-driven:
setting ``HEXIGA_BACKEND=nurbs_c`` (or providing a module
``hexiga.nurbs._c`` with the same call signatures) will be preferred over
the NumPy reference.  This keeps the pure-Python path the deterministic
reference implementation, which the test-suite also uses as the oracle.
"""

from __future__ import annotations

import os

from . import _kernels

__all__ = [
    "backend_name",
    "findspan",
    "basisfun",
    "bspeval",
    "bspderiv",
    "bspkntins",
    "bspdegelev",
]

def backend_name() -> str:
    """Return the name of the active NURBS kernel backend.

    Currently ``"numpy"``; ``"nurbs_c"`` once the optional compiled backend
    is available and selected via ``HEXIGA_BACKEND``.
    """
    if os.environ.get("HEXIGA_BACKEND") == "nurbs_c":
        try:
            import hexiga.nurbs._c as _c  # noqa: F401  (optional)

            return "nurbs_c"
        except Exception:  # pragma: no cover - optional path
            pass
    return "numpy"

def _resolve():
    """Import and return the concrete kernel implementation to use."""
    if os.environ.get("HEXIGA_BACKEND") == "nurbs_c":
        try:
            import hexiga.nurbs._c as _c

            return _c
        except Exception:  # pragma: no cover - optional path
            pass
    return _kernels

_K = _resolve()

findspan = _K.findspan
basisfun = _K.basisfun
bspeval = _K.bspeval
bspderiv = _K.bspderiv
bspkntins = _K.bspkntins
bspdegelev = _K.bspdegelev
