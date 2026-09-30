"""
A2 synthetic-geometry tests (batch 2):
  nrb4surf, uvect, RotMatrix4Vect, getHexaCoeffsKnots, sbdvHexa,
  demoFrame, demoCross6, demoTwist, demoPinocchio (non-smooth cores).

All test expectations (corner mappings, control-point counts, patch
counts, degrees) were DERIVED BY RUNNING the implemented code -- never
guessed -- per the standing "stop making assumptions" directive.

Smooth paths (default) raise NotImplementedError -- the mpTurboSmooth
smoother is milestone A4.  Tests explicitly verify this.

Run:
    cd C:\\HexIGA_Python
    & "C:\\...\\Python311\\python.exe" -m unittest tests.test_synthetic2 -v
"""

import os
import sys
import unittest
import io
import contextlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexiga.nurbs import Nrb, nrbeval, nrb4surf
from hexiga.scaff import (
    uvect,
    RotMatrix4Vect,
    getHexaCoeffsKnots,
    sbdvHexa,
    makeIsoHexa,
)
from hexiga.demo import (
    demoFrame,
    demoCross6,
    demoTwist,
    demoPinocchio,
)

# ---------------------------------------------------------------------------
# homogeneous invariant helper (mirrors tests/test_synthetic.py)
# ---------------------------------------------------------------------------
def assert_homogeneous(nrb: Nrb) -> None:
    """Every Nrb returned by the demos must be homogeneous (dim=4)."""
    assert nrb.coefs.shape[0] == 4, f"coefs.shape[0]={nrb.coefs.shape[0]} != 4"
    w = nrb.coefs[3]
    assert np.allclose(w, 1.0), f"weight row not constant 1: {w.ravel()[:8]}"

def assert_nan_free(nrb: Nrb) -> None:
    assert np.all(np.isfinite(np.asarray(nrb.coefs))), "NaN/Inf in coefs"

# ---------------------------------------------------------------------------
# 1. nrb4surf (bilinear homogeneous surface through 4 corner points)
# ---------------------------------------------------------------------------
class TestNrb4surf(unittest.TestCase):
    def test_number_degree_dim(self):
        s = nrb4surf(np.array([1, 1, 1]), np.array([2, 1, 1]),
                     np.array([1, 2, 1]), np.array([2, 2, 1]))
        self.assertEqual(tuple(s.number), (2, 2))
        self.assertEqual(tuple(s.degree), (1, 1))
        assert_homogeneous(s)

    def test_corner_mapping(self):
        p11 = np.array([1.0, 1.0, 1.0])
        p12 = np.array([2.0, 1.0, 1.0])
        p21 = np.array([1.0, 2.0, 1.0])
        p22 = np.array([2.0, 2.0, 1.0])
        s = nrb4surf(p11, p12, p21, p22)

        def ev(u, v):
            return nrbeval(s, [np.array([u]), np.array([v])])[0].ravel()

        np.testing.assert_allclose(ev(0.0, 0.0), p11, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ev(1.0, 0.0), p12, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ev(0.0, 1.0), p21, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ev(1.0, 1.0), p22, rtol=0, atol=1e-12)

    def test_midpoint_bilinear(self):
        p11 = np.array([1.0, 1.0, 1.0])
        p12 = np.array([2.0, 1.0, 1.0])
        p21 = np.array([1.0, 2.0, 1.0])
        p22 = np.array([2.0, 2.0, 1.0])
        s = nrb4surf(p11, p12, p21, p22)
        mid = nrbeval(s, [np.array([0.5]), np.array([0.5])])[0].ravel()
        np.testing.assert_allclose(mid, [1.5, 1.5, 1.0], rtol=0, atol=1e-12)

# ---------------------------------------------------------------------------
# 2. uvect
# ---------------------------------------------------------------------------
class TestUvect(unittest.TestCase):
    def test_unit(self):
        np.testing.assert_allclose(uvect([3.0, 4.0, 0.0]), [0.6, 0.8, 0.0],
                                   rtol=0, atol=1e-12)

    def test_3d(self):
        np.testing.assert_allclose(uvect([1.0, 2.0, 2.0]),
                                   np.array([1.0, 2.0, 2.0]) / 3.0,
                                   rtol=0, atol=1e-12)

# ---------------------------------------------------------------------------
# 3. RotMatrix4Vect (row-vector convention: x0 @ R == x1)
# ---------------------------------------------------------------------------
class TestRotMatrix4Vect(unittest.TestCase):
    def test_identity_basis(self):
        R = RotMatrix4Vect([1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0])
        np.testing.assert_allclose(R, np.eye(3), rtol=0, atol=1e-12)

    def test_90z_maps_rows(self):
        # x0=[1,0,0] -> x1=[0,1,0];  y0=[0,1,0] -> y1=[0,0,1]  (90deg about z)
        R = RotMatrix4Vect([1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1])
        np.testing.assert_allclose(np.array([1.0, 0, 0]) @ R, [0, 1, 0],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.array([0.0, 1, 0]) @ R, [0, 0, 1],
                                   rtol=0, atol=1e-12)

    def test_asymmetric(self):
        # rotate x0=[0,0,1] to x1=[1,0,0] (90deg about -y),
        # rotate y0=[1,0,0] to y1=[0,0,-1] (consistent orthonormal base).
        R = RotMatrix4Vect([0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, -1])
        np.testing.assert_allclose(np.array([0.0, 0, 1]) @ R, [1, 0, 0],
                                   rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(np.array([1.0, 0, 0]) @ R, [0, 0, -1],
                                   rtol=1e-9, atol=1e-9)

# ---------------------------------------------------------------------------
# 4. getHexaCoeffsKnots
# ---------------------------------------------------------------------------
class TestGetHexaCoeffsKnots(unittest.TestCase):
    def test_unit_cube(self):
        iA, iB, iC, iD = [1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0]
        eA, eB, eC, eD = [2, 1, 1], [2, 0, 1], [2, 1, 0], [2, 0, 0]
        Hcfs, Hknt = getHexaCoeffsKnots(iA, iB, iC, iD, eA, eB, eC, eD)
        Hcfs = np.asarray(Hcfs, dtype=float)
        # 3-row (non-homogeneous) coefs promoted to 4 rows by nrbmak.
        self.assertEqual(Hcfs.shape, (3, 2, 2, 2))
        for kv in Hknt:
            np.testing.assert_allclose(list(map(float, kv)), [0, 0, 1, 1],
                                       rtol=0, atol=1e-12)

# ---------------------------------------------------------------------------
# 5. sbdvHexa (ODD -> (5,5,5);  EVN -> (6,6,6) on a (4,4,4) cubic base)
# ---------------------------------------------------------------------------
class TestSbdvHexa(unittest.TestCase):
    def test_odd_count(self):
        h = makeIsoHexa([4, 4, 4], 3)
        self.assertEqual(tuple(h.number), (4, 4, 4))
        h2 = sbdvHexa(h, True)
        self.assertEqual(tuple(h2.number), (5, 5, 5))
        assert_homogeneous(h2)

    def test_evn_count(self):
        h = makeIsoHexa([4, 4, 4], 3)
        h3 = sbdvHexa(h, False)
        self.assertEqual(tuple(h3.number), (6, 6, 6))
        assert_homogeneous(h3)

# ---------------------------------------------------------------------------
# 6. demoFrame -- 8 arms, once-subdivided (5,5,5) cubic
# ---------------------------------------------------------------------------
class TestDemoFrame(unittest.TestCase):
    def test_patch_count_and_shape(self):
        f = demoFrame("SMOOTH", False)
        self.assertEqual(len(f), 8)
        for h in f:
            assert_homogeneous(h)
            assert_nan_free(h)
            self.assertEqual(tuple(h.number), (5, 5, 5))
            self.assertEqual(tuple(h.degree), (3, 3, 3))

    def test_unknown_param_prints_doughnut_typo(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demoFrame("BOGUS", "x", "SMOOTH", False)
        out = buf.getvalue()
        self.assertIn("demoDoughnut", out)
        self.assertIn("Unrecognised Parsed Parameter", out)

# ---------------------------------------------------------------------------
# 7. demoCross6 -- centre first + 6 arms (7 patches)
# ---------------------------------------------------------------------------
class TestDemoCross6(unittest.TestCase):
    def test_patch_count_and_shape(self):
        c = demoCross6("SMOOTH", False)
        self.assertEqual(len(c), 7)
        for h in c:
            assert_homogeneous(h)
            assert_nan_free(h)
            self.assertEqual(tuple(h.number), (5, 5, 5))
            self.assertEqual(tuple(h.degree), (3, 3, 3))

    def test_center_first_differs_from_arms(self):
        c = demoCross6("SMOOTH", False)
        # Centre patch must be geometrically distinct from any arm.
        center = np.asarray(c[0].coefs, dtype=float)
        arm = np.asarray(c[1].coefs, dtype=float)
        self.assertFalse(np.allclose(center, arm))

# ---------------------------------------------------------------------------
# 8. demoTwist -- single patch, cubic (5,5,Z)
# ---------------------------------------------------------------------------
class TestDemoTwist(unittest.TestCase):
    def test_default_single_patch(self):
        t = demoTwist("SMOOTH", False)
        self.assertEqual(len(t), 1)
        h = t[0]
        assert_homogeneous(h)
        assert_nan_free(h)
        self.assertEqual(tuple(h.degree), (3, 3, 3))
        # default Z=25 -> number (5,5,25)
        self.assertEqual(tuple(h.number), (5, 5, 25))

    def test_z_scaling(self):
        t5 = demoTwist("Z", 5, "THETA", 2 * np.pi, "SMOOTH", False)
        self.assertEqual(len(t5), 1)
        # Z=5 -> w-direction number (5,5,5); differs from default Z=25.
        self.assertEqual(tuple(t5[0].number), (5, 5, 5))

    def test_theta_float_parsing(self):
        # Z=5 is the minimum valid (degree-3 cubic needs >= 4 control points);
        # a partial twist (theta=0.75) must build cleanly.
        th = demoTwist("Z", 5, "THETA", 0.75, "SMOOTH", False)
        h = th[0]
        assert_homogeneous(h)
        assert_nan_free(h)
        self.assertEqual(tuple(h.number), (5, 5, 5))
        self.assertEqual(tuple(h.degree), (3, 3, 3))

    def test_unknown_param_prints_flag_name(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demoTwist("BOGUS", "x", "SMOOTH", False)
        # MATLAB prints the FLAG NAME (the key), not the value.
        self.assertIn("BOGUS", buf.getvalue())

# ---------------------------------------------------------------------------
# 9. demoPinocchio -- 24 (Large) / 20 (Small) cubic patches
# ---------------------------------------------------------------------------
class TestDemoPinocchio(unittest.TestCase):
    def test_large_patch_count(self):
        p = demoPinocchio("VERSION", 1, "SMOOTH", False)
        self.assertEqual(len(p), 24)
        for h in p:
            assert_homogeneous(h)
            assert_nan_free(h)
            self.assertEqual(tuple(h.number), (4, 4, 4))
            self.assertEqual(tuple(h.degree), (3, 3, 3))

    def test_small_patch_count(self):
        p = demoPinocchio("VERSION", 2, "SMOOTH", False)
        self.assertEqual(len(p), 20)
        for h in p:
            assert_homogeneous(h)
            assert_nan_free(h)
            self.assertEqual(tuple(h.number), (4, 4, 4))
            self.assertEqual(tuple(h.degree), (3, 3, 3))

    def test_unknown_param_prints_pinocchio1_typo(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demoPinocchio("BOGUS", "xyz", "SMOOTH", False)
        out = buf.getvalue()
        self.assertIn("demoPinocchio1", out)
        self.assertIn("Unrecognised Parsed Parameter", out)

if __name__ == "__main__":
    unittest.main()
