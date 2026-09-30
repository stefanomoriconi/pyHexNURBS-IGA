"""
hexiga.nurbs._c
================

Optional ctypes wrapper around a small, original C implementation of the
performance-critical NURBS kernels (``hexiga/nurbs/_c/src/nurbs_kernels.c``):

* :func:`findspan`   -- knot-span location
* :func:`basisfun`   -- non-zero basis functions at a parameter value
* :func:`bspeval`    -- batched curve evaluation
* :func:`bspderiv`   -- derivative control points / knots
* :func:`bspkntins`  -- knot insertion
* :func:`bspdegelev` -- degree elevation

This module is selected transparently by :mod:`hexiga.nurbs._backend` when
the environment variable ``HEXIGA_BACKEND`` is set to ``"nurbs_c"`` *and*
the compiled shared library is present (build it with
``bash hexiga/nurbs/_c/build.sh``; it is not shipped pre-built). The
pure-NumPy reference in :mod:`hexiga.nurbs._kernels` is the default and the
deterministic oracle used by the test-suite; every function here matches
its call signature and return values exactly (cross-checked in
``tests/test_nurbs_c_backend.py``, skipped when the library isn't built).

The C kernels use a **row-major** ``(n_rows, n_ctrl)`` layout matching the
NumPy reference directly, so no transposition is needed at the FFI
boundary.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

__all__ = [
    "findspan",
    "basisfun",
    "bspeval",
    "bspderiv",
    "bspkntins",
    "bspdegelev",
]

def _find_library() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "libhexiga_nurbs_c.so"),
        os.path.join(here, "libhexiga_nurbs_c.dylib"),
        os.path.join(here, "hexiga_nurbs_c.dll"),
    ]
    env = os.environ.get("HEXIGA_NURBS_C_LIB")
    if env:
        candidates.insert(0, env)
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise ImportError(
        "hexiga.nurbs._c: compiled native library not found. "
        "Run `bash hexiga/nurbs/_c/build.sh` to build it from "
        "hexiga/nurbs/_c/src/nurbs_kernels.c, or unset HEXIGA_BACKEND to "
        "use the pure-NumPy reference backend."
    )

_LIB = ctypes.CDLL(_find_library())

_dbl_p = ctypes.POINTER(ctypes.c_double)
_int_p = ctypes.POINTER(ctypes.c_int)

def _as_c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)

def _ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return arr.ctypes.data_as(_dbl_p)

_LIB.hexiga_findspan.restype = ctypes.c_int
_LIB.hexiga_findspan.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, _dbl_p]

_LIB.hexiga_basisfun.restype = None
_LIB.hexiga_basisfun.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int, _dbl_p, _dbl_p]

_LIB.hexiga_bspeval.restype = None
_LIB.hexiga_bspeval.argtypes = [
    ctypes.c_int, _dbl_p, ctypes.c_int, ctypes.c_int, _dbl_p, _dbl_p, ctypes.c_int, _dbl_p,
]

_LIB.hexiga_bspderiv.restype = None
_LIB.hexiga_bspderiv.argtypes = [
    ctypes.c_int, _dbl_p, ctypes.c_int, ctypes.c_int, _dbl_p, _dbl_p, _dbl_p,
]

_LIB.hexiga_bspkntins.restype = None
_LIB.hexiga_bspkntins.argtypes = [
    ctypes.c_int, _dbl_p, ctypes.c_int, ctypes.c_int, _dbl_p, _dbl_p, ctypes.c_int, _dbl_p, _dbl_p,
]

_LIB.hexiga_bspdegelev.restype = None
_LIB.hexiga_bspdegelev.argtypes = [
    ctypes.c_int, _dbl_p, ctypes.c_int, ctypes.c_int, _dbl_p, ctypes.c_int,
    _dbl_p, ctypes.c_int, _dbl_p, _int_p, _int_p,
]

def findspan(n: int, p: int, u: float, U: np.ndarray) -> int:
    """Knot span of ``u`` for degree ``p`` with ``n`` control points (0-based
    last control index) and knot vector ``U``. Same contract as
    :func:`hexiga.nurbs._kernels.findspan`."""
    U = _as_c(U)
    return int(_LIB.hexiga_findspan(n + 1, p, float(u), _ptr(U)))

def basisfun(i: int, u: float, p: int, U: np.ndarray) -> np.ndarray:
    """Non-vanishing basis functions at ``u`` in span ``i`` (length ``p+1``)."""
    U = _as_c(U)
    out = np.zeros(p + 1, dtype=np.float64)
    _LIB.hexiga_basisfun(i, float(u), p, _ptr(U), _ptr(out))
    return out

def bspeval(d: int, c: np.ndarray, k: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate a univariate B-spline; see :func:`hexiga.nurbs._kernels.bspeval`."""
    c = _as_c(c)
    k = _as_c(k)
    u = _as_c(np.atleast_1d(u))
    mc, nc = c.shape
    nu = u.size
    out = np.zeros((mc, nu), dtype=np.float64)
    _LIB.hexiga_bspeval(d, _ptr(c), mc, nc, _ptr(k), _ptr(u), nu, _ptr(out))
    return out

def bspderiv(d: int, c: np.ndarray, k: np.ndarray):
    """Control points/knots of the first derivative; see
    :func:`hexiga.nurbs._kernels.bspderiv`."""
    c = _as_c(c)
    k = _as_c(k)
    mc, nc = c.shape
    dc = np.zeros((mc, nc - 1), dtype=np.float64)
    dk = np.zeros(nc + d - 1, dtype=np.float64)
    _LIB.hexiga_bspderiv(d, _ptr(c), mc, nc, _ptr(k), _ptr(dc), _ptr(dk))
    return dc, dk

def bspkntins(d: int, c: np.ndarray, k: np.ndarray, u: np.ndarray):
    """Insert knots ``u`` (non-decreasing); see
    :func:`hexiga.nurbs._kernels.bspkntins`."""
    c = _as_c(c)
    k = _as_c(k)
    u = _as_c(np.atleast_1d(u))
    mc, nc = c.shape
    nu = u.size
    ic = np.zeros((mc, nc + nu), dtype=np.float64)
    ik = np.zeros(nc + d + 1 + nu, dtype=np.float64)
    _LIB.hexiga_bspkntins(d, _ptr(c), mc, nc, _ptr(k), _ptr(u), nu, _ptr(ic), _ptr(ik))
    return ic, ik

def bspdegelev(d: int, c: np.ndarray, k: np.ndarray, t: int):
    """Degree-elevate from ``d`` to ``d + t``; see
    :func:`hexiga.nurbs._kernels.bspdegelev`."""
    c = _as_c(c)
    k = _as_c(k)
    mc, nc = c.shape
    n_knots = k.size
    # Worst case (every interior knot simple, i.e. maximal refinement):
    # at most nc + t*(n_knots) new control points/knots is a safe, generous
    # upper bound (actual output is always <= this).
    max_ic = nc + t * n_knots + d + 2
    max_ik = n_knots + t * n_knots + d + 2

    ic_buf = np.zeros((mc, max_ic), dtype=np.float64)
    ik_buf = np.zeros(max_ik, dtype=np.float64)
    n_ic = ctypes.c_int(0)
    n_ik = ctypes.c_int(0)

    _LIB.hexiga_bspdegelev(
        d, _ptr(c), mc, nc, _ptr(k), t,
        _ptr(ic_buf), max_ic, _ptr(ik_buf),
        ctypes.byref(n_ic), ctypes.byref(n_ik),
    )
    return ic_buf[:, : n_ic.value].copy(), ik_buf[: n_ik.value].copy()
