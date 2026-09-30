"""
hexiga.nurbs._kernels
=====================

Pure-NumPy reference implementations of the six performance-critical NURBS
kernels, implementing the standard algorithms from "The NURBS Book"
(Piegl & Tiller):

* :func:`findspan`   -- A2.1
* :func:`basisfun`   -- A2.2
* :func:`bspeval`    -- A3.1
* :func:`bspderiv`   -- A3.3
* :func:`bspkntins`  -- A5.4
* :func:`bspdegelev` -- A5.9

This module is the deterministic oracle used by the test suite; an
optional, original C implementation with the same call signatures lives
in :mod:`hexiga.nurbs._c` and can be activated as a drop-in accelerator
(see :mod:`hexiga.nurbs._backend`).

Conventions (0-based):
  * ``c`` is an ``(mc, nc)`` array: ``mc`` is the row count (``4`` for NURBS:
    x, y, z, weight) and ``nc`` the number of control points.
  * knot vectors include the full clamped ends, length ``nc + d + 1``.
  * ``degree`` is the B-spline degree ``d`` (order = ``d + 1``).
"""

from __future__ import annotations

from math import comb

import numpy as np

__all__ = [
    "findspan",
    "basisfun",
    "bspeval",
    "bspderiv",
    "bspkntins",
    "bspdegelev",
]

# ---------------------------------------------------------------------------
# A2.1 - findspan
# ---------------------------------------------------------------------------
def findspan(n: int, p: int, u: float, U: np.ndarray) -> int:
    """Knot span of ``u`` for a B-spline of degree ``p`` with ``n`` (0-based
    last control index) control points and knot vector ``U``.

    Port of ``findspan.c`` (Algorithm A2.1).
    """
    U = np.asarray(U, dtype=float)
    # special case
    if u == U[n + 1]:
        return n

    low = p
    high = n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid

# ---------------------------------------------------------------------------
# A2.2 - basisfun
# ---------------------------------------------------------------------------
def basisfun(i: int, u: float, p: int, U: np.ndarray) -> np.ndarray:
    """Non-vanishing basis functions at ``u`` in span ``i``.

    Returns an array of length ``p + 1``.  Port of ``basisfun.c`` (A2.2).
    """
    U = np.asarray(U, dtype=float)
    N = np.zeros(p + 1)
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)

    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N

# ---------------------------------------------------------------------------
# A3.1 - bspeval
# ---------------------------------------------------------------------------
def bspeval(d: int, c: np.ndarray, k: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate a univariate B-spline.

    Parameters
    ----------
    d : int
        Degree.
    c : (mc, nc) array
        Control points (row 3 is the weight for NURBS).
    k : (nc + d + 1,) array
        Knot vector (clamped).
    u : (nu,) array
        Parametric points.

    Returns
    -------
    (mc, nu) array of evaluated points.

    Port of ``bspeval.c`` (A3.1).
    """
    c = np.asarray(c, dtype=float)
    k = np.asarray(k, dtype=float)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    mc, nc = c.shape
    nu = u.size

    out = np.zeros((mc, nu))
    for col in range(nu):
        s = findspan(nc - 1, d, float(u[col]), k)
        N = basisfun(s, float(u[col]), d, k)
        tmp1 = s - d
        for row in range(mc):
            acc = 0.0
            for ii in range(d + 1):
                acc += N[ii] * c[row, tmp1 + ii]
            out[row, col] = acc
    return out

# ---------------------------------------------------------------------------
# A3.3 - bspderiv
# ---------------------------------------------------------------------------
def bspderiv(d: int, c: np.ndarray, k: np.ndarray):
    """Control points and knot vector of the first derivative.

    Returns ``(dc, dk)`` where ``dc`` is ``(mc, nc-1)`` and ``dk`` is
    ``(nc + d - 1,)``.  Port of ``bspderiv.c`` (A3.3).
    """
    c = np.asarray(c, dtype=float)
    k = np.asarray(k, dtype=float)
    mc, nc = c.shape

    dc = np.zeros((mc, nc - 1))
    for ii in range(nc - 1):
        tmp = d / (k[ii + d + 1] - k[ii + 1])
        for j in range(mc):
            dc[j, ii] = tmp * (c[j, ii + 1] - c[j, ii])
    dk = k[1:-1].copy()
    return dc, dk

# ---------------------------------------------------------------------------
# A5.4 - bspkntins
# ---------------------------------------------------------------------------
def bspkntins(d: int, c: np.ndarray, k: np.ndarray, u: np.ndarray):
    """Insert knots ``u`` (non-decreasing) into a B-spline.

    Returns ``(ic, ik)`` where ``ic`` is ``(mc, nc+nu)`` and ``ik`` is
    ``(nc + d + 1 + nu,)``.  Port of ``bspkntins.c`` (A5.4).

    Note: ``u`` must be provided in non-decreasing order.
    """
    c = np.asarray(c, dtype=float)
    k = np.asarray(k, dtype=float)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    mc, nc = c.shape
    nu = u.size

    n = nc - 1
    r = nu - 1

    m = n + d + 1
    a = findspan(n, d, float(u[0]), k)
    b = findspan(n, d, float(u[r]), k)
    b += 1

    ic = np.zeros((mc, nc + nu))
    for q in range(mc):
        for j in range(0, a - d + 1):
            ic[q, j] = c[q, j]
        for j in range(b - 1, n + 1):
            ic[q, j + r + 1] = c[q, j]
    ik = np.zeros(nc + d + 1 + nu)
    for j in range(0, a + 1):
        ik[j] = k[j]
    for j in range(b + d, m + 1):
        ik[j + r + 1] = k[j]

    i = b + d - 1
    s = b + d + r
    for j in range(r, -1, -1):
        while u[j] <= k[i] and i > a:
            for q in range(mc):
                ic[q, s - d - 1] = c[q, i - d - 1]
            ik[s] = k[i]
            s -= 1
            i -= 1
        for q in range(mc):
            ic[q, s - d - 1] = ic[q, s - d]
        for l in range(1, d + 1):
            ind = s - d + l
            alfa = ik[s + l] - u[j]
            if abs(alfa) == 0.0:
                for q in range(mc):
                    ic[q, ind - 1] = ic[q, ind]
            else:
                alfa = alfa / (ik[s + l] - k[i - d + l])
                for q in range(mc):
                    ic[q, ind - 1] = alfa * ic[q, ind - 1] + (1.0 - alfa) * ic[q, ind]
        ik[s] = u[j]
        s -= 1
    return ic, ik

# ---------------------------------------------------------------------------
# A5.9 - bspdegelev
# ---------------------------------------------------------------------------
def bspdegelev(d: int, c: np.ndarray, k: np.ndarray, t: int):
    """Degree-elevate a B-spline from degree ``d`` to ``d + t``.

    Returns ``(ic, ik)`` where ``ic`` is ``(mc, nh+1)`` and ``ik`` the new
    clamped knot vector.  Port of ``bspdegelev.c`` (A5.9).

    Notes
    -----
    This NumPy implementation is the **geometrically correct** reference and
    is exact for all input degrees.  The original GeoPDEs kernel (C and
    pure-MATLAB) has a verified bug for **input degree ``d >= 4``**: it
    produces the same (correct) knot vector ``ik`` but control points whose
    elevated curve deviates from the original by ~1e-2..1e-1, and its shipped
    ``bspdegelev.mexw64`` even corrupts the heap at ``d=4``.  HexIGA is a
    cubic codebase, so every production elevation pair has input degree
    ``d <= 3`` (e.g. linear->cubic ``[2 2 2]``, quadratic->quartic ``[2 2 0]``,
    cubic->quintic ``[2 2 2]``, degree-2->3 ``[1 1 1]``), all of which agree
    exactly between this kernel and the C backend.
    """
    c = np.asarray(c, dtype=float)
    k = np.asarray(k, dtype=float)
    mc, nc = c.shape

    n = nc - 1
    m = n + d + 1
    ph = d + t
    ph2 = ph // 2

    def bincoeff(nn, kk):
        return float(comb(nn, kk))

    # bezalfs[i][j],  i in 0..ph, j in 0..d
    bezalfs = [[0.0] * (d + 1) for _ in range(ph + 1)]
    bezalfs[0][0] = 1.0
    bezalfs[ph][d] = 1.0

    for i in range(1, ph2 + 1):
        inv = 1.0 / bincoeff(ph, i)
        mpi = min(d, i)
        for j in range(max(0, i - t), mpi + 1):
            bezalfs[i][j] = inv * bincoeff(d, j) * bincoeff(t, i - j)

    for i in range(ph2 + 1, ph):
        mpi = min(d, i)
        for j in range(max(0, i - t), mpi + 1):
            bezalfs[i][j] = bezalfs[ph - i][d - j]

    mh = ph
    kind = ph + 1
    r = -1
    a = d
    b = d + 1
    cind = 1
    ua = k[0]

    # ic rows accumulate; row 0 is the first control point
    ic = [np.array([c[ii, 0] for ii in range(mc)])]
    ik = [ua] * (ph + 1)

    # first bezier segment (d+1 control points)
    bpts = [np.array([c[ii, i] for ii in range(mc)]) for i in range(d + 1)]
    ebpts = [np.zeros(mc) for _ in range(ph + 1)]
    Nextbpts = [np.zeros(mc) for _ in range(d + 1)]
    alfs = [0.0] * (d + 1)

    while b < m:
        i = b
        while b < m and k[b] == k[b + 1]:
            b += 1

        mul = b - i + 1
        mh += mul + t
        ub = k[b]
        oldr = r
        r = d - mul

        if oldr > 0:
            lbz = (oldr + 2) // 2
        else:
            lbz = 1

        if r > 0:
            rbz = ph - (r + 1) // 2
        else:
            rbz = ph

        if r > 0:
            numer = ub - ua
            for q in range(d, mul, -1):
                alfs[q - mul - 1] = numer / (k[a + q] - ua)
            for j in range(1, r + 1):
                save = r - j
                s = mul + j
                for q in range(d, s - 1, -1):
                    for ii in range(mc):
                        bpts[q][ii] = (
                            alfs[q - s] * bpts[q][ii]
                            + (1.0 - alfs[q - s]) * bpts[q - 1][ii]
                        )
                for ii in range(mc):
                    Nextbpts[save][ii] = bpts[d][ii]

        # degree elevate bezier
        for i in range(lbz, ph + 1):
            for ii in range(mc):
                ebpts[i][ii] = 0.0
            mpi = min(d, i)
            for j in range(max(0, i - t), mpi + 1):
                for ii in range(mc):
                    ebpts[i][ii] += bezalfs[i][j] * bpts[j][ii]

        if oldr > 1:
            first = kind - 2
            last = kind
            den = ub - ua
            bet = (ub - ik[kind - 1]) / den

            for tr in range(1, oldr):
                i = first
                j = last
                kj = j - kind + 1
                while j - i > tr:
                    if i < cind:
                        alf = (ub - ik[i]) / (ua - ik[i])
                        for ii in range(mc):
                            ic[i][ii] = alf * ic[i][ii] + (1.0 - alf) * ic[i - 1][ii]
                    if j >= lbz:
                        if j - tr <= kind - ph + oldr:
                            gam = (ub - ik[j - tr]) / den
                            for ii in range(mc):
                                ebpts[kj][ii] = (
                                    gam * ebpts[kj][ii]
                                    + (1.0 - gam) * ebpts[kj + 1][ii]
                                )
                        else:
                            for ii in range(mc):
                                ebpts[kj][ii] = (
                                    bet * ebpts[kj][ii]
                                    + (1.0 - bet) * ebpts[kj + 1][ii]
                                )
                    i += 1
                    j -= 1
                    kj -= 1
                first -= 1
                last += 1

        # load the knot ua
        if a != d:
            for _ in range(ph - oldr):
                ik.append(ua)
                kind += 1

        # load ctrl pts into ic
        for j in range(lbz, rbz + 1):
            ic.append(np.array([ebpts[j][ii] for ii in range(mc)]))
            cind += 1

        if b < m:
            for j in range(r):
                for ii in range(mc):
                    bpts[j][ii] = Nextbpts[j][ii]
            for j in range(r, d + 1):
                for ii in range(mc):
                    bpts[j][ii] = c[ii, b - d + j]
            a = b
            b += 1
            ua = ub
        else:
            for _ in range(ph + 1):
                ik.append(ub)

    nic = mh - ph
    nik = nic + d + t + 1
    ik = ik[:nik]

    ic_arr = np.stack([np.array(row) for row in ic[:nic]], axis=1)
    ik_arr = np.array(ik, dtype=float)
    return ic_arr, ik_arr
