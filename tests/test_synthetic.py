"""
A2 synthetic-geometry tests (batch 1):
  getUniformKnotVect, makeIsoHexa, mpMerge, makeCuboid,
  demoTorus, demoQuadball, demoEggLW (non-smooth cores).

Faithful literal port of the corresponding MATLAB ground truth:
  smth_utils/getUniformKnotVect.m
  smth_utils/makeIsoHexa.m
  scaff_utils/mpMerge.m
  smth_utils/multipatch/makeCuboid.m
  demo_utils/Synthetic/demoTorus.m
  demo_utils/Synthetic/demoQuadball.m
  demo_utils/Synthetic/demoEggLW.m

Smooth paths (default) raise NotImplementedError -- the mpTurboSmooth
smoother is milestone A4.  Tests explicitly verify this.

Run:
    cd C:\\HexIGA_Python
    & "C:\\...\\Python311\\python.exe" -m unittest tests.test_synthetic -v
"""

import os
import sys
import unittest
import io
import contextlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexiga.nurbs import Nrb, nrbeval, nrbdegelev
from hexiga.scaff import getUniformKnotVect, makeIsoHexa, mpMerge
from hexiga.demo import makeCuboid, demoTorus, demoQuadball, demoEggLW
from hexiga.demo.demo_torus import genTorus
from hexiga.demo.demo_egg_lw import genEgg

# ---------------------------------------------------------------------------
# homogeneous invariant helper
# ---------------------------------------------------------------------------
def assert_homogeneous(nrb: Nrb) -> None:
    """Every Nrb returned by the demos must be homogeneous (dim=4)."""
    assert nrb.coefs.shape[0] == 4, f"coefs.shape[0]={nrb.coefs.shape[0]} != 4"
    w = nrb.coefs[3]
    assert np.allclose(w, 1.0), f"weight row not constant 1: {w.ravel()[:8]}"

def grid_sample(nrb: Nrb, n: int = 3):
    """Return a small parametric grid over [0,1]^ndim (homogeneous eval)."""
    u = np.linspace(0, 1, n)
    grids = [u] * nrb.ndim
    return grids

# ---------------------------------------------------------------------------
# 1. getUniformKnotVect
# ---------------------------------------------------------------------------
class TestGetUniformKnotVect(unittest.TestCase):
    def test_linear_length_and_clamps(self):
        for dim in (2, 3, 5):
            kv = getUniformKnotVect(dim, 1)
            self.assertEqual(len(kv), dim + 2, f"dim={dim}")
            self.assertEqual(kv[0], 0.0)
            self.assertEqual(kv[-1], 1.0)
            self.assertEqual(kv[1], 0.0)  # first interior = 0
            self.assertEqual(kv[-2], 1.0)  # last interior = 1

    def test_cubic_length_and_clamps(self):
        for dim in (4, 5, 6):
            kv = getUniformKnotVect(dim, 3)
            self.assertEqual(len(kv), dim + 4, f"dim={dim}")
            self.assertEqual(kv[0], 0.0)
            self.assertEqual(kv[-1], 1.0)
            self.assertEqual(kv[1], 0.0)
            self.assertEqual(kv[2], 0.0)
            self.assertEqual(kv[3], 0.0)
            self.assertEqual(kv[-2], 1.0)
            self.assertEqual(kv[-3], 1.0)
            self.assertEqual(kv[-4], 1.0)

    def test_quadratic_length_and_clamps(self):
        for dim in (3, 4, 5):
            kv = getUniformKnotVect(dim, 2)
            self.assertEqual(len(kv), dim + 3, f"dim={dim}")
            self.assertEqual(kv[0], 0.0)
            self.assertEqual(kv[1], 0.0)
            self.assertEqual(kv[-1], 1.0)
            self.assertEqual(kv[-2], 1.0)

    def test_error_deg_gte_dim(self):
        with self.assertRaises(ValueError):
            getUniformKnotVect(3, 3)
        with self.assertRaises(ValueError):
            getUniformKnotVect(2, 2)

# ---------------------------------------------------------------------------
# 2. makeIsoHexa
# ---------------------------------------------------------------------------
class TestMakeIsoHexa(unittest.TestCase):
    def test_cube(self):
        H = makeIsoHexa([3, 3, 3], 2)
        assert_homogeneous(H)
        self.assertEqual(H.ndim, 3)
        self.assertEqual(H.number, (3, 3, 3))
        # coefs: (4, 3, 3, 3)
        self.assertEqual(H.coefs.shape, (4, 3, 3, 3))
        # Linear control points form the exact grid on [-1,1]^3.
        xs = np.linspace(-1, 1, 3)
        ys = np.linspace(-1, 1, 3)
        zs = np.linspace(-1, 1, 3)
        xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
        np.testing.assert_allclose(H.coefs[0], xg)
        np.testing.assert_allclose(H.coefs[1], yg)
        np.testing.assert_allclose(H.coefs[2], zg)
        np.testing.assert_allclose(H.coefs[3], np.ones((3, 3, 3)))

    def test_nonuniform_dims(self):
        H = makeIsoHexa([3, 4, 5], 2)
        assert_homogeneous(H)
        self.assertEqual(H.number, (3, 4, 5))
        self.assertEqual(H.coefs.shape, (4, 3, 4, 5))
        # Per-axis scale is dims[i]/maxDim (see makeIsoHexa.m).
        # maxDim = 5 => x scales by 3/5, y by 4/5, z by 5/5.
        xlo, xhi = -3.0 / 5, 3.0 / 5
        ylo, yhi = -4.0 / 5, 4.0 / 5
        zlo, zhi = -1.0, 1.0
        np.testing.assert_allclose(H.coefs[0].min(), xlo)
        np.testing.assert_allclose(H.coefs[0].max(), xhi)
        np.testing.assert_allclose(H.coefs[1].min(), ylo)
        np.testing.assert_allclose(H.coefs[1].max(), yhi)
        np.testing.assert_allclose(H.coefs[2].min(), zlo)
        np.testing.assert_allclose(H.coefs[2].max(), zhi)

    def test_knot_lengths(self):
        H = makeIsoHexa([3, 4, 5], 2)
        self.assertEqual(len(H.knots[0]), 3 + 2 + 1)
        self.assertEqual(len(H.knots[1]), 4 + 2 + 1)
        self.assertEqual(len(H.knots[2]), 5 + 2 + 1)
        # Each knot vector is clamped.
        for k in H.knots:
            self.assertEqual(k[0], 0.0)
            self.assertEqual(k[-1], 1.0)

    def test_error_dims_not_3(self):
        with self.assertRaises(ValueError):
            makeIsoHexa([3, 3], 2)
        with self.assertRaises(ValueError):
            makeIsoHexa([3, 3, 3, 3], 2)

# ---------------------------------------------------------------------------
# 3. makeCuboid
# ---------------------------------------------------------------------------
class TestMakeCuboid(unittest.TestCase):
    def test_unit_cube_eval(self):
        A0 = np.array([0.0, 0.0, 0.0])
        B0 = np.array([1.0, 0.0, 0.0])
        C0 = np.array([1.0, 1.0, 0.0])
        D0 = np.array([0.0, 1.0, 0.0])
        A1 = np.array([0.0, 0.0, 1.0])
        B1 = np.array([1.0, 0.0, 1.0])
        C1 = np.array([1.0, 1.0, 1.0])
        D1 = np.array([0.0, 1.0, 1.0])
        c = makeCuboid(A0, B0, C0, D0, A1, B1, C1, D1)
        assert_homogeneous(c)
        self.assertEqual(c.number, (2, 2, 2))
        self.assertEqual(c.order, (2, 2, 2))
        self.assertEqual(len(c.knots[0]), 4)
        self.assertEqual(len(c.knots[1]), 4)
        self.assertEqual(len(c.knots[2]), 4)
        np.testing.assert_allclose(c.knots[0], [0, 0, 1, 1])

    def test_cuboid_corner_eval(self):
        # Build a non-trivial box: A0..D1 at known positions.
        A0 = np.array([0.0, 0.0, 0.0])
        B0 = np.array([2.0, 0.0, 0.0])
        C0 = np.array([2.0, 1.0, 0.0])
        D0 = np.array([0.0, 1.0, 0.0])
        A1 = np.array([0.0, 0.0, 3.0])
        B1 = np.array([2.0, 0.0, 3.0])
        C1 = np.array([2.0, 1.0, 3.0])
        D1 = np.array([0.0, 1.0, 3.0])
        c = makeCuboid(A0, B0, C0, D0, A1, B1, C1, D1)
        # Evaluate at (u,v,w) in {0,1}^3 (all 8 corners).
        for uu in (0.0, 1.0):
            for vv in (0.0, 1.0):
                for ww in (0.0, 1.0):
                    cp, cw = nrbeval(c, [[uu], [vv], [ww]], homogeneous=True)
                    self.assertEqual(cp.shape, (3, 1, 1, 1))
                    self.assertAlmostEqual(cp[0, 0, 0, 0], uu * 2.0)
                    self.assertAlmostEqual(cp[1, 0, 0, 0], vv * 1.0)
                    self.assertAlmostEqual(cp[2, 0, 0, 0], ww * 3.0)

# ---------------------------------------------------------------------------
# 4. mpMerge
# ---------------------------------------------------------------------------
class TestMpMerge(unittest.TestCase):
    def _make_pair(self):
        """Two adjacent unit-cube hexahedra along w."""
        A0 = np.array([0.0, 0.0, 0.0])
        B0 = np.array([1.0, 0.0, 0.0])
        C0 = np.array([1.0, 1.0, 0.0])
        D0 = np.array([0.0, 1.0, 0.0])
        A1 = np.array([0.0, 0.0, 1.0])
        B1 = np.array([1.0, 0.0, 1.0])
        C1 = np.array([1.0, 1.0, 1.0])
        D1 = np.array([0.0, 1.0, 1.0])
        h0 = makeCuboid(A0, B0, C0, D0, A1, B1, C1, D1)
        # Second cube shifted up by 1 in z.
        A0b = A0 + np.array([0, 0, 1.0])
        B0b = B0 + np.array([0, 0, 1.0])
        C0b = C0 + np.array([0, 0, 1.0])
        D0b = D0 + np.array([0, 0, 1.0])
        A1b = A1 + np.array([0, 0, 1.0])
        B1b = B1 + np.array([0, 0, 1.0])
        C1b = C1 + np.array([0, 0, 1.0])
        D1b = D1 + np.array([0, 0, 1.0])
        h1 = makeCuboid(A0b, B0b, C0b, D0b, A1b, B1b, C1b, D1b)
        return h0, h1

    def test_merge_dim3_eval_matches_patch0(self):
        h0, h1 = self._make_pair()
        m = mpMerge([h0, h1], 3)
        self.assertIsNotNone(m)
        assert_homogeneous(m)
        # Merged patch: number w = 3 (2 + 2 - 1).
        self.assertEqual(m.number, (2, 2, 3))
        # Evaluate merged patch at w=0: must match h0 at w=0.
        for uu in (0.25, 0.75):
            for vv in (0.25, 0.75):
                cp_m, _ = nrbeval(m, [[uu], [vv], [0.0]], homogeneous=True)
                cp_0, _ = nrbeval(h0, [[uu], [vv], [0.0]], homogeneous=True)
                np.testing.assert_allclose(cp_m, cp_0)
        # Evaluate merged patch at w=0.5: must match h0 at w=1 (= shared face).
        for uu in (0.25, 0.75):
            for vv in (0.25, 0.75):
                cp_m, _ = nrbeval(m, [[uu], [vv], [0.5]], homogeneous=True)
                cp_0, _ = nrbeval(h0, [[uu], [vv], [1.0]], homogeneous=True)
                np.testing.assert_allclose(cp_m, cp_0, rtol=1e-12, atol=1e-12)

    def test_merge_knot_normalisation(self):
        h0, h1 = self._make_pair()
        m = mpMerge([h0, h1], 3)
        self.assertEqual(m.knots[2][-1], 1.0)
        self.assertEqual(m.knots[2][0], 0.0)

    def test_merge_4_patches_dim3(self):
        h0, h1 = self._make_pair()
        # h2, h3: continue stacking along z.
        off2 = np.array([0.0, 0.0, 2.0])
        off3 = np.array([0.0, 0.0, 3.0])
        A0 = np.array([0.0, 0.0, 0.0])
        B0 = np.array([1.0, 0.0, 0.0])
        C0 = np.array([1.0, 1.0, 0.0])
        D0 = np.array([0.0, 1.0, 0.0])
        A1 = np.array([0.0, 0.0, 1.0])
        B1 = np.array([1.0, 0.0, 1.0])
        C1 = np.array([1.0, 1.0, 1.0])
        D1 = np.array([0.0, 1.0, 1.0])
        h2 = makeCuboid(A0 + off2, B0 + off2, C0 + off2, D0 + off2,
                        A1 + off2, B1 + off2, C1 + off2, D1 + off2)
        h3 = makeCuboid(A0 + off3, B0 + off3, C0 + off3, D0 + off3,
                        A1 + off3, B1 + off3, C1 + off3, D1 + off3)
        m = mpMerge([h0, h1, h2, h3], 3)
        self.assertIsNotNone(m)
        assert_homogeneous(m)
        # w-count: 2 + 2 + 2 + 2 - 3 = 5.
        self.assertEqual(m.number[2], 5)

    def test_merge_dim1(self):
        h0, h1 = self._make_pair()
        # Shift h1 in x (not z) for a U-dir merge: h1 at [1,2]x[0,1]x[0,1].
        off = np.array([1.0, 0.0, 0.0])
        A0 = np.array([0.0, 0.0, 0.0])
        B0 = np.array([1.0, 0.0, 0.0])
        C0 = np.array([1.0, 1.0, 0.0])
        D0 = np.array([0.0, 1.0, 0.0])
        A1 = np.array([0.0, 0.0, 1.0])
        B1 = np.array([1.0, 0.0, 1.0])
        C1 = np.array([1.0, 1.0, 1.0])
        D1 = np.array([0.0, 1.0, 1.0])
        h1u = makeCuboid(A0 + off, B0 + off, C0 + off, D0 + off,
                         A1 + off, B1 + off, C1 + off, D1 + off)
        m = mpMerge([h0, h1u], 1)
        self.assertIsNotNone(m)
        self.assertEqual(m.number[0], 3)
        self.assertEqual(m.knots[0][-1], 1.0)

# ---------------------------------------------------------------------------
# 5. demoTorus
# ---------------------------------------------------------------------------
class TestDemoTorus(unittest.TestCase):
    def test_gen_torus_patch_count_and_deg(self):
        t = genTorus()
        self.assertEqual(len(t), 8)
        for h in t:
            assert_homogeneous(h)
            # Cubic: nrbdegelev(linear, [2,2,2]) -> degree 3 = order 4.
            self.assertEqual(h.degree, (3, 3, 3))
            self.assertEqual(h.order, (4, 4, 4))

    def test_demo_torus_merged(self):
        t = demoTorus("MERGE", True, "SMOOTH", False)
        self.assertEqual(len(t), 4)
        for h in t:
            assert_homogeneous(h)
            self.assertEqual(h.degree, (3, 3, 3))

    def test_demo_torus_non_merged(self):
        t = demoTorus("MERGE", False, "SMOOTH", False)
        self.assertEqual(len(t), 8)
        for h in t:
            assert_homogeneous(h)

# ---------------------------------------------------------------------------
# 6. demoQuadball
# ---------------------------------------------------------------------------
class TestDemoQuadball(unittest.TestCase):
    def test_non_smooth(self):
        q = demoQuadball("SMOOTH", False)
        self.assertEqual(len(q), 1)
        h = q[0]
        assert_homogeneous(h)
        # makeIsoHexa([3,3,3],[2,2,2]) then degelev [1,1,1] -> cubic (deg 3).
        self.assertEqual(h.degree, (3, 3, 3))
        self.assertEqual(h.order, (4, 4, 4))

# ---------------------------------------------------------------------------
# 7. demoEggLW
# ---------------------------------------------------------------------------
class TestDemoEggLW(unittest.TestCase):
    def test_gen_egg_counts(self):
        HexaQ, HexaW = genEgg()
        self.assertEqual(len(HexaQ), 4)
        self.assertEqual(len(HexaW), 16)
        for h in HexaQ + HexaW:
            assert_homogeneous(h)

    def test_demo_egg_non_smooth(self):
        EggL, EggW = demoEggLW("SMOOTH", False)
        self.assertEqual(len(EggL), 4)
        self.assertEqual(len(EggW), 16)
        for h in EggL + EggW:
            assert_homogeneous(h)
            # Non-smooth path returns the original LINEAR hexes (deg 1).
            self.assertEqual(h.degree, (1, 1, 1))
            self.assertEqual(h.order, (2, 2, 2))

# ---------------------------------------------------------------------------
# 8. degelev invariance on a demo patch (spot check)
# ---------------------------------------------------------------------------
class TestDegelevInvariance(unittest.TestCase):
    def test_torus_patch_degelev_eval(self):
        """Degree-elevated patch must evaluate identically to original at
        sampled parameter values (up to ~1e-12)."""
        t = genTorus()
        h = t[0]
        h2 = nrbdegelev(h, [1, 1, 1])
        self.assertEqual(h2.degree, (4, 4, 4))
        us = np.array([0.25, 0.5, 0.75])
        for u in us:
            for v in us:
                for w in us:
                    cp1, _ = nrbeval(h, [[u], [v], [w]], homogeneous=True)
                    cp2, _ = nrbeval(h2, [[u], [v], [w]], homogeneous=True)
                    np.testing.assert_allclose(cp1, cp2, rtol=1e-12, atol=1e-12)

if __name__ == "__main__":
    unittest.main()
