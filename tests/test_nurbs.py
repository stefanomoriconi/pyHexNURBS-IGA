#!/usr/bin/env python3
"""
tests/test_nurbs.py — M1d: stdlib ``unittest`` suite for hexiga.nurbs.

Covers:
  * kernel invariants (findspan, basisfun, bspeval, bspderiv, bspkntins,
    bspdegelev)
  * NRB object construction (1D, 2D, 3D)
  * nrbmak, nrbeval, nrbderiv, nrbdeval
  * nrbkntins, nrbextract, nrbreverse, nrbtransp
  * nrbmultipatch (interface / boundary detection)
  * io: nrblead / nrbsave round-trip on a synthetic 2-patch 1D mesh
        and on Junct3.txt (if present)

Oracle conventions:
  * MATLAB is ground truth.
  * Coords stored in homogeneous form (4, *number); last row = weights.
  * ``nrbeval`` default is homogeneous → returns (cp, cw).
"""

from __future__ import annotations
import os
import sys
import unittest
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexiga.nurbs import (  # noqa: E402
    Nrb,
    nrbmak,
    nrbeval,
    nrbderiv,
    nrbdeval,
    nrbkntins,
    nrbextract,
    nrbreverse,
    nrbtransp,
    nrbmultipatch,
    nrblead,
    nrbsave,
)
from hexiga.nurbs import _backend  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_bezier_curve(P, deg):
    """Build a Bezier curve (order deg+1) from 1-D or 2-D control points.

    The NURBS storage is homogeneous with 3 spatial rows + weight; 1-D and
    2-D inputs are padded with a zero z-row, which is invisible to any
    evaluation/transform that does not touch the third coordinate.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    if P.shape[0] == 1:
        P = np.vstack([P, np.zeros((1, P.shape[1])), np.zeros((1, P.shape[1]))])
    elif P.shape[0] == 2:
        P = np.vstack([P, np.zeros((1, P.shape[1]))])
    n = P.shape[1]
    assert n == deg + 1
    k = np.r_[np.zeros(deg + 1), np.ones(deg + 1)]
    c = np.zeros((4, n))
    c[:3, :] = P
    c[3, :] = 1.0
    return Nrb(number=(n,), order=(deg + 1,), knots=(k,), coefs=c)

def make_bicubic_surface(P):
    nu, nv = P.shape[1:]
    assert nu == 5 and nv == 5
    k = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float)
    c = np.zeros((4, nu, nv))
    c[:3, :, :] = P
    c[3, :, :] = 1.0
    return Nrb(number=(nu, nv), order=(4, 4), knots=(k, k.copy()), coefs=c)

def make_bicubic_volume(P):
    nu, nv, nw = P.shape[1:]
    assert nu == nv == nw == 5
    k = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float)
    c = np.zeros((4, nu, nv, nw))
    c[:3, :, :, :] = P
    c[3, :, :, :] = 1.0
    return Nrb(number=(nu, nv, nw), order=(4, 4, 4),
               knots=(k, k.copy(), k.copy()), coefs=c)

def _grid_pnts(nu, nv, f):
    U, V = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="ij")
    return f(U, V)

# ---------------------------------------------------------------------------
# Kernel invariants
# ---------------------------------------------------------------------------
class TestKernels(unittest.TestCase):
    def test_bspeval_partition_of_unity(self):
        # degree 3, n=5, clamped knots
        k = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        d = 3
        n = 4  # control points = knot_len - d - 1 = 8 - 3 - 1
        c = np.ones((1, n))
        us = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        B = _backend.bspeval(d, c, k, us)
        # B has shape (1, nu) and each column sums to 1 across basis functions
        # (c has only one row → B == basisfun evaluated at us).
        # For a single CP row with weight 1, the sum over CPs of N_i(u)*1 = 1.
        s = B[0, :]  # shape (nu,) = sum over CPs
        self.assertTrue(np.allclose(s, 1.0), f"partition of unity violated: {s}")

    def test_bspderiv_cubic_bezier_derivative_coefficients(self):
        # Cubic Bezier (clamped knots, no interior knots):
        #   P'(0) = d * (P1 - P0),  P'(1) = d * (P3 - P2).
        # The 1st-derivative curve has d control points; for a Bezier the
        # derivative control points are d*(c[:, 1:] - c[:, :-1]).
        k = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        d = 3
        n = 4  # knot_len - d - 1 = 8 - 3 - 1
        c = np.array([[1.0, 2.0, 4.0, 7.0]])
        dc, dk = _backend.bspderiv(d, c, k)
        expected = d * (c[:, 1:] - c[:, :-1])  # [[3, 6, 9]]
        self.assertTrue(np.allclose(dc, expected),
                        f"dc={dc} expected={expected}")
        # dk is the derivative knot vector (length = nc + d - 1).
        self.assertEqual(dk.size, n + d - 1)

    def test_bspkntins_preserves_value(self):
        # Inserting a knot does not change the evaluated value.
        k = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        d = 3
        n = 4  # control points = knot_len - d - 1 = 8 - 3 - 1
        rng = np.random.default_rng(0)
        c = rng.random((1, n))
        us = np.array([0.2, 0.5, 0.8])
        B0 = _backend.bspeval(d, c, k, us)

        # Insert one knot at 0.5 via bspkntins
        ic, ik = _backend.bspkntins(d, c, k, np.array([0.5]))
        B1 = _backend.bspeval(d, ic, ik, us)
        self.assertTrue(np.allclose(B0, B1), f"{B0} != {B1}")

    def test_bspdegelev_preserves_value(self):
        k = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        d = 3
        n = 4  # control points = knot_len - d - 1 = 8 - 3 - 1
        rng = np.random.default_rng(1)
        c = rng.random((1, n))
        us = np.array([0.1, 0.5, 0.9])
        B0 = _backend.bspeval(d, c, k, us)
        ic, ik = _backend.bspdegelev(d, c, k, 1)
        B1 = _backend.bspeval(d + 1, ic, ik, us)
        self.assertTrue(np.allclose(B0, B1))

    def test_findspan_within_bounds(self):
        k = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1])
        n = 5
        d = 3
        for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r = _backend.findspan(n, d, u, k)
            self.assertGreaterEqual(r, d)
            self.assertLessEqual(r, n)

# ---------------------------------------------------------------------------
# NRB construction
# ---------------------------------------------------------------------------
class TestNrbConstruction(unittest.TestCase):
    def test_bezier_curve_1d(self):
        P = np.array([[0.0, 1.0, 2.0, 3.0],
                      [0.0, 2.0, 4.0, 6.0]])
        nrb = make_bezier_curve(P, 3)
        self.assertEqual(nrb.number, (4,))
        self.assertEqual(nrb.order, (4,))
        self.assertEqual(len(nrb.knots), 1)
        self.assertEqual(nrb.coefs.shape, (4, 4))

    def test_bicubic_surface_2d(self):
        rng = np.random.default_rng(2)
        P = rng.random((3, 5, 5))
        nrb = make_bicubic_surface(P)
        self.assertEqual(nrb.number, (5, 5))
        self.assertEqual(nrb.order, (4, 4))
        self.assertEqual(len(nrb.knots), 2)
        self.assertEqual(nrb.coefs.shape, (4, 5, 5))

    def test_bicubic_volume_3d(self):
        rng = np.random.default_rng(3)
        P = rng.random((3, 5, 5, 5))
        nrb = make_bicubic_volume(P)
        self.assertEqual(nrb.number, (5, 5, 5))
        self.assertEqual(nrb.order, (4, 4, 4))
        self.assertEqual(len(nrb.knots), 3)
        self.assertEqual(nrb.coefs.shape, (4, 5, 5, 5))

# ---------------------------------------------------------------------------
# nrbmak / nrbeval / nrbderiv / nrbdeval
# ---------------------------------------------------------------------------
class TestEval(unittest.TestCase):
    def test_nrbeval_1d_homogeneous(self):
        P = np.array([[0.0, 1.0, 2.0, 3.0],
                      [0.0, 2.0, 4.0, 6.0]])
        Ppad = np.vstack([P, np.zeros((1, P.shape[1]))])
        nrb = make_bezier_curve(P, 3)
        us = np.array([0.0, 0.5, 1.0])
        cp, cw = nrbeval(nrb, us)
        # Homogeneous form: cp has shape (rdim+1, nu), cw = weights = 1
        self.assertEqual(cp.shape, (3, 3))
        self.assertTrue(np.allclose(cw, 1.0))
        # Bezier endpoint property: curve(t=0)=CP0, curve(t=1)=CPn-1.
        self.assertTrue(np.allclose(cp[:, 0], Ppad[:, 0]))
        self.assertTrue(np.allclose(cp[:, -1], Ppad[:, -1]))

    def test_nrbeval_2d_homogeneous(self):
        rng = np.random.default_rng(4)
        P = rng.random((3, 5, 5))
        nrb = make_bicubic_surface(P)
        us = np.array([0.0, 0.5, 1.0])
        vs = np.array([0.0, 0.5, 1.0])
        cp, cw = nrbeval(nrb, [us, vs])
        # 2D grid: nu=3, nv=3 → (3, 3, 3)
        self.assertEqual(cp.shape, (3, 3, 3))
        self.assertEqual(cw.shape, (3, 3))

    def test_nrbeval_3d_homogeneous(self):
        rng = np.random.default_rng(5)
        P = rng.random((3, 5, 5, 5))
        nrb = make_bicubic_volume(P)
        us = np.array([0.0, 0.5, 1.0])
        vs = np.array([0.0, 0.5, 1.0])
        ws = np.array([0.0, 0.5, 1.0])
        cp, cw = nrbeval(nrb, [us, vs, ws])
        self.assertEqual(cp.shape, (3, 3, 3, 3))
        self.assertEqual(cw.shape, (3, 3, 3))

    def test_nrbderiv_1d_first_deriv(self):
        # Cubic Bezier P(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
        # P'(t) = 3(1-t)^2*(P1-P0) + 6(1-t)*t*(P2-P1) + 3t^2*(P3-P2)
        P = np.array([[0.0, 1.0, 3.0, 6.0]])  # 1D
        nrb = make_bezier_curve(P, 3)
        us = np.array([0.0, 0.5, 1.0])
        D = nrbderiv(nrb)
        # D is a list with one Nrb per requested direction.
        # For 1D with default idir, D[0] is the first derivative (degree 2).
        cp1, cw1 = nrbeval(D[0], us)
        # At t=0: P'(0) = 3*(P1-P0) = 3
        self.assertAlmostEqual(cp1[0, 0], 3.0, places=10)
        # At t=1: P'(1) = 3*(P3-P2) = 9
        self.assertAlmostEqual(cp1[0, -1], 9.0, places=10)

    def test_nrbdeval_1d(self):
        P = np.array([[0.0, 1.0, 3.0, 6.0]])
        nrb = make_bezier_curve(P, 3)
        us = np.array([0.0, 0.5, 1.0])
        dnurbs = nrbderiv(nrb, [1])
        pnt, jac, hess = nrbdeval(nrb, dnurbs, tt=[us])
        # Bezier derivative: dP/dt(0)=3*(P1-P0)=3, dP/dt(1)=3*(P3-P2)=9.
        self.assertAlmostEqual(jac[0][0, 0], 3.0, places=10)
        self.assertAlmostEqual(jac[0][0, -1], 9.0, places=10)

# ---------------------------------------------------------------------------
# nrbkntins / nrbextract / nrbreverse / nrbtransp
# ---------------------------------------------------------------------------
class TestTransforms(unittest.TestCase):
    def test_nrbkntins_1d_preserves_geometry(self):
        P = np.array([[0.0, 1.0, 3.0, 6.0],
                      [0.0, 4.0, 9.0, 16.0]])
        nrb = make_bezier_curve(P, 3)
        nrb2 = nrbkntins(nrb, np.array([0.25, 0.5, 0.75]))
        us = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        c1, _ = nrbeval(nrb, us)
        c2, _ = nrbeval(nrb2, us)
        self.assertTrue(np.allclose(c1, c2, atol=1e-12))

    def test_nrbreverse_1d_preserves_geometry(self):
        P = np.array([[0.0, 1.0, 3.0, 6.0]])
        nrb = make_bezier_curve(P, 3)
        nrb_r = nrbreverse(nrb)
        us = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        c1, _ = nrbeval(nrb, us)
        c2, _ = nrbeval(nrb_r, 1.0 - us)
        self.assertTrue(np.allclose(c1, c2, atol=1e-12))

    def test_nrbextract_1d(self):
        P = np.array([[0.0, 1.0, 3.0, 6.0],
                      [0.0, 4.0, 9.0, 16.0]])
        Ppad = np.vstack([P, np.zeros((1, P.shape[1]))])
        nrb = make_bezier_curve(P, 3)
        # Extract the two endpoints (side 1 and side 2 of a 1D curve).
        # nrbextract returns a list of Nrb objects.
        e1 = nrbextract(nrb, [1])[0]
        e2 = nrbextract(nrb, [2])[0]
        # Side 1 = first CP, side 2 = last CP (3 spatial rows).
        self.assertTrue(np.allclose(e1.coefs[:3, 0], Ppad[:, 0], atol=1e-12))
        self.assertTrue(np.allclose(e2.coefs[:3, 0], Ppad[:, -1], atol=1e-12))

    def test_nrbtransp_2d(self):
        # Transpose a 2D surface: swap u and v
        rng = np.random.default_rng(6)
        P = rng.random((3, 5, 5))
        # Make it asymmetric so transposition is detectable
        P[0, 0, 0] = 1.0
        nrb = make_bicubic_surface(P)
        nrb_t = nrbtransp(nrb)
        self.assertEqual(nrb_t.number, (5, 5))
        # Transposed coefs: nrb_t.coefs[idim, i, j] == nrb.coefs[idim, j, i]
        for idim in range(3):
            self.assertTrue(np.allclose(nrb_t.coefs[idim, :, :],
                                        nrb.coefs[idim, :, :].T, atol=1e-12))

# ---------------------------------------------------------------------------
# nrbmultipatch
# ---------------------------------------------------------------------------
class TestMultipatch(unittest.TestCase):
    def test_two_connected_cubes(self):
        # Two unit cubes sharing a face: patch1 [0,1]^3, patch2 [1,2]x[0,1]x[0,1]
        def _cube(x0):
            # 2x2x2 Bezier volume (cubic)
            rng = np.random.default_rng(7)
            P = np.zeros((3, 2, 2, 2))
            # Corner coords
            for i in range(2):
                for j in range(2):
                    for l in range(2):
                        P[0, i, j, l] = x0 + [0, 1][i]
                        P[1, i, j, l] = [0, 1][j]
                        P[2, i, j, l] = [0, 1][l]
            k = np.r_[np.zeros(2), np.ones(2)]  # deg=1 → order=2
            c = np.zeros((4, 2, 2, 2))
            c[:3] = P
            c[3] = 1.0
            return Nrb(number=(2, 2, 2), order=(2, 2, 2),
                       knots=(k, k.copy(), k.copy()), coefs=c)

        p1 = _cube(0.0)
        p2 = _cube(1.0)
        interfaces, boundary = nrbmultipatch([p1, p2])
        # Two cubes sharing one face → 1 interface, 10 boundary faces (6+6-2)
        self.assertEqual(len(interfaces), 1, f"got {len(interfaces)}")
        self.assertEqual(len(boundary), 10, f"got {len(boundary)}")

# ---------------------------------------------------------------------------
# I/O: round-trip
# ---------------------------------------------------------------------------
class TestIO(unittest.TestCase):
    def test_roundtrip_1d_two_patches(self):
        # Two connected cubic Bezier curves sharing an endpoint
        P1 = np.array([[0.0, 1.0, 2.0, 3.0]])
        P2 = np.array([[3.0, 4.0, 6.0, 9.0]])
        n1 = make_bezier_curve(P1, 3)
        n2 = make_bezier_curve(P2, 3)

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        tmp.close()

        try:
            # 1-D decks carry no interface orientation in the v2.1
            # format (writer omits the ornt line), so round-trip the
            # patches-only deck and verify both patches survive.
            nrbsave(tmp.name, [n1, n2])
            p, i, s, b = nrblead(tmp.name)
            self.assertEqual(len(p), 2)
            self.assertEqual(len(i), 0)
            for a, b_ in zip([n1, n2], p):
                self.assertEqual(a.number, b_.number)
                self.assertEqual(a.order, b_.order)
                self.assertTrue(np.allclose(a.coefs, b_.coefs))
        finally:
            os.remove(tmp.name)

    def test_junct3_read(self):
        # Reference geometry from the MATLAB ground-truth tree (dev-machine
        # only; not distributed with this repo). Skipped when absent.
        junct3 = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "HexIGA_Master_Orig",
            "test_geometry", "Junct3.txt"))
        if not os.path.isfile(junct3):
            self.skipTest("Junct3.txt not found")
        p, i, s, b = nrblead(junct3)
        self.assertEqual(len(p), 12)
        self.assertEqual(len(i), 18)
        self.assertEqual(len(b), 36)
        for patch in p:
            self.assertEqual(patch.number, (4, 4, 7))
            self.assertEqual(patch.order, (4, 4, 4))

if __name__ == "__main__":
    unittest.main(verbosity=2)
