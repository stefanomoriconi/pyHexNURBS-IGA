"""
A1 core-operations tests:
  nrbtform, nrbpermute, nrbglue, mpStats, nrb_export_stl (surf2stl),
  mp_geo_load / mp_geo_read_nurbs.

Literal port of the corresponding MATLAB NURBS-toolbox / scaff / export /
GeoPDEs behaviour.  Covers 1D (curve), 2D (surface) and 3D (volume)
embeddings as well as real .txt data when available (Junct3.txt).

Run:
    cd C:\\HexIGA_Python
    & "C:\\...\\Python311\\python.exe" -m unittest tests.test_core_ops -v
"""

import io
import os
import struct
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexiga.nurbs import (Nrb, nrbeval, nrbderiv, nrbdeval, nrbtform,
                          nrbpermute, nrbglue, nrbkntins, nrbdegelev)
from hexiga.iga import mp_geo_load, mp_geo_read_nurbs
from hexiga.scaff import mpStats
from hexiga.export import nrb_export_stl

# Reference geometry shipped with the MATLAB ground-truth tree (dev-machine
# only; not distributed with this repo). Skipped when absent.
TEST_GEOM = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "HexIGA_Master_Orig",
    "test_geometry", "Junct3.txt"))

# ---------------------------------------------------------------------------
# fixtures (same style as tests.test_nurbs)
# ---------------------------------------------------------------------------
def make_bezier_curve(P, deg=3):
    if P.ndim == 1:
        P = P.reshape(1, -1)
    n = P.shape[1]
    k = np.r_[np.zeros(deg + 1), np.ones(deg + 1)]
    c = np.zeros((4, n))
    c[:P.shape[0], :] = P
    c[P.shape[0]:, :] = 0
    c[3, :] = 1.0
    return Nrb(number=(n,), order=(deg + 1,), knots=(k,), coefs=c)

def make_bicubic_surface(P):
    assert P.ndim == 3 and P.shape[1:] == (5, 5)
    k = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float)
    c = np.zeros((4, 5, 5))
    c[:3, :, :] = P
    c[3, :, :] = 1.0
    return Nrb(number=(5, 5), order=(4, 4), knots=(k, k.copy()), coefs=c)

def make_bicubic_volume(P):
    assert P.ndim == 4 and P.shape[1:] == (5, 5, 5)
    k = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float)
    c = np.zeros((4, 5, 5, 5))
    c[:3, :, :, :] = P
    c[3, :, :, :] = 1.0
    return Nrb(number=(5, 5, 5), order=(4, 4, 4),
               knots=(k, k.copy(), k.copy()), coefs=c)

def make_box(lox, uax):
    """Unit-cube Bezier volume spanning [lox,uax]x[0,1]x[0,1] with a
    simple x-dependent x coordinate (so evaluation points are identifiable).
    Control points: x = lox + (uax-lox)*i/4, y=j/4, z=k/4."""
    P = np.zeros((3, 5, 5, 5))
    for i in range(5):
        for j in range(5):
            for k in range(5):
                P[0, i, j, k] = lox + (uax - lox) * i / 4.0
                P[1, i, j, k] = j / 4.0
                P[2, i, j, k] = k / 4.0
    return make_bicubic_volume(P)

def _capture_stdout():
    old = sys.stdout
    sys.stdout = buf = io.StringIO()
    return old, buf

def _restore_stdout(old):
    sys.stdout = old

# ---------------------------------------------------------------------------
# nrbtform
# ---------------------------------------------------------------------------
class TestNrbTform(unittest.TestCase):
    def test_identity_leaves_eval_unchanged_1d_2d_3d(self):
        T = np.eye(4)
        rng = np.random.default_rng(0)
        for nrb in [make_bezier_curve(rng.random((3, 4))),
                    make_bicubic_surface(rng.random((3, 5, 5))),
                    make_bicubic_volume(rng.random((3, 5, 5, 5)))]:
            t = nrbtform(nrb, T)
            ndim = nrb.ndim
            tt = [np.array([0.0, 0.5, 1.0]) for _ in range(ndim)]
            c0, w0 = nrbeval(nrb, tt)
            c1, w1 = nrbeval(t, tt)
            self.assertTrue(np.allclose(c0, c1) and np.allclose(w0, w1))

    def test_translation_shifts_1d(self):
        nrb = make_bezier_curve(np.array([[0.0, 1.0, 2.0, 3.0]]))
        T = np.array([[1, 0, 0, 10.0],
                      [0, 1, 0, -5.0],
                      [0, 0, 1, 2.0],
                      [0, 0, 0, 1.0]])
        t = nrbtform(nrb, T)
        us = np.array([0.0, 0.3, 0.7, 1.0])
        c0, _ = nrbeval(nrb, us)
        c1, _ = nrbeval(t, us)
        self.assertTrue(np.allclose(c1[0], c0[0] + 10.0))
        self.assertTrue(np.allclose(c1[1], c0[1] - 5.0))
        self.assertTrue(np.allclose(c1[2], c0[2] + 2.0))

    def test_translation_2d_and_3d(self):
        T = np.array([[1, 0, 0, 1.5],
                      [0, 1, 0, -2.0],
                      [0, 0, 1, 4.0],
                      [0, 0, 0, 1.0]])
        tt2 = [np.array([0.2, 0.8]), np.array([0.1, 0.9])]
        for nrb in [make_bicubic_surface(np.random.default_rng(1).random((3, 5, 5))),
                    make_bicubic_volume(np.random.default_rng(2).random((3, 5, 5, 5)))]:
            tt = [np.array([0.2, 0.8])] * nrb.ndim
            c0, _ = nrbeval(nrb, tt)
            c1, _ = nrbeval(nrbtform(nrb, T), tt)
            self.assertTrue(np.allclose(c1[0], c0[0] + 1.5))
            self.assertTrue(np.allclose(c1[1], c0[1] - 2.0))
            self.assertTrue(np.allclose(c1[2], c0[2] + 4.0))

    def test_rotation_90z_2d(self):
        rng = np.random.default_rng(3)
        Q = rng.random((3, 5, 5))
        nrb = make_bicubic_surface(Q)
        T = np.array([[0.0, -1.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        t = nrbtform(nrb, T)
        tt = [np.array([0.0, 0.5, 1.0]) for _ in range(2)]
        c0, _ = nrbeval(nrb, tt)
        c1, _ = nrbeval(t, tt)
        self.assertTrue(np.allclose(c1[0], -c0[1]))
        self.assertTrue(np.allclose(c1[1], c0[0]))
        self.assertTrue(np.allclose(c1[2], c0[2]))

    def test_tmat_none_raises(self):
        nrb = make_bezier_curve(np.array([[0.0, 1.0, 2.0, 3.0]]))
        with self.assertRaises(ValueError):
            nrbtform(nrb, None)

    def test_bad_tmat_shape_raises(self):
        nrb = make_bezier_curve(np.array([[0.0, 1.0, 2.0, 3.0]]))
        with self.assertRaises(ValueError):
            nrbtform(nrb, np.eye(3))

# ---------------------------------------------------------------------------
# nrbpermute
# ---------------------------------------------------------------------------
class TestNrbPermute(unittest.TestCase):
    def test_curve_raises(self):
        nrb = make_bezier_curve(np.array([[0.0, 1.0, 2.0, 3.0]]))
        with self.assertRaises(ValueError):
            nrbpermute(nrb, [1, 2])

    def test_2d_swap_eval_symmetry(self):
        rng = np.random.default_rng(4)
        P = rng.random((3, 5, 5))
        nrb = make_bicubic_surface(P)
        Psw = np.swapaxes(P, 1, 2)
        nrb_sw = make_bicubic_surface(Psw)
        sw = nrbpermute(nrb, [2, 1])
        tt = [np.array([0.1, 0.4, 0.7]), np.array([0.2, 0.5, 0.8])]
        c0, w0 = nrbeval(sw, tt)
        c1, w1 = nrbeval(nrb_sw, tt)
        self.assertTrue(np.allclose(c0, c1) and np.allclose(w0, w1))

    def test_3d_permute_number_and_eval(self):
        rng = np.random.default_rng(5)
        P = rng.random((3, 5, 6, 7))
        k5 = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float)        # len 9 = 5+4
        k6 = np.array([0, 0, 0, 0, 0.4, 0.6, 1, 1, 1, 1], dtype=float)   # len 10 = 6+4
        k7 = np.array([0, 0, 0, 0, 0.3, 0.5, 0.7, 1, 1, 1, 1], dtype=float)  # len 11 = 7+4
        c = np.zeros((4, 5, 6, 7))
        c[:3] = P
        c[3] = 1.0
        nrb = Nrb(number=(5, 6, 7), order=(4, 4, 4),
                  knots=(k5, k6, k7), coefs=c)
        perm = nrbpermute(nrb, [1, 3, 2])
        self.assertEqual(perm.number, (5, 7, 6))
        # Reference: transpose CPs (keep weight row) and swap knot dirs 2<->3.
        c_ref = np.zeros((4, 5, 7, 6))
        c_ref[:3] = np.transpose(P, (0, 1, 3, 2))
        c_ref[3] = 1.0
        nrb_ref = Nrb(number=(5, 7, 6), order=(4, 4, 4),
                      knots=(nrb.knots[0], nrb.knots[2], nrb.knots[1]),
                      coefs=c_ref)
        tt = [np.array([0.3]), np.array([0.4]), np.array([0.5])]
        cp, wp = nrbeval(perm, tt)
        cq, wq = nrbeval(nrb_ref, tt)
        self.assertTrue(np.allclose(cp, cq) and np.allclose(wp, wq))
        # knots order is permuted accordingly
        self.assertTrue(np.array_equal(perm.knots[1], nrb.knots[2]))
        self.assertTrue(np.array_equal(perm.knots[2], nrb.knots[1]))

    def test_roundtrip_permute(self):
        rng = np.random.default_rng(6)
        P = rng.random((3, 5, 5))
        nrb = make_bicubic_surface(P)
        back = nrbpermute(nrbpermute(nrb, [2, 1]), [2, 1])
        tt = [np.array([0.25, 0.75]), np.array([0.1, 0.9])]
        c0, _ = nrbeval(nrb, tt)
        c1, _ = nrbeval(back, tt)
        self.assertTrue(np.allclose(c0, c1))

# ---------------------------------------------------------------------------
# nrbglue
# ---------------------------------------------------------------------------
class TestNrbGlue(unittest.TestCase):
    def test_glue_two_cubic_lines_end_to_end(self):
        P1 = np.array([[0.0, 1.0, 2.0, 3.0]])
        P2 = np.array([[3.0, 4.0, 5.0, 6.0]])
        n1 = make_bezier_curve(P1, 3)
        n2 = make_bezier_curve(P2, 3)
        g = nrbglue(n1, n2)
        # two Bezier segments → 7 CPs (shared endpoint dropped), internal knot at 0.5
        self.assertEqual(g.number, (7,))
        # glued CPs = a.coefs + b.coefs(:,2:end) = [P1] + [P2][1:]
        self.assertTrue(np.allclose(g.coefs[0], np.r_[P1[0], P2[0][1:]]))
        # glued parameter u: the glued curve spans [0, 2] -- n1 occupies [0, 1]
        # (local t = u) and n2 occupies [1, 2] (local t = u - 1).
        us = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        cg, _ = nrbeval(g, us)
        ref = np.empty_like(us)
        for i, u in enumerate(us):
            if u <= 1.0:
                v, _ = nrbeval(n1, np.array([u]))
            else:
                v, _ = nrbeval(n2, np.array([u - 1.0]))
            ref[i] = v[0]
        self.assertTrue(np.allclose(cg[0], ref))

    def test_glue_two_boxes_3d(self):
        A = make_box(0.0, 1.0)
        B = make_box(1.0, 2.0)
        g = nrbglue(A, B)
        # A: 5 CPs in x, B: 5 CPs in x, shared face merged → 9; other axes stay 5 x 5.
        self.assertEqual(g.number, (9, 5, 5))
        # glued box spans [0, 2] in x: A occupies [0, 1] (local t = u),
        # B occupies [1, 2] (local t = u - 1).
        # value at u=0 → A left face (t=0)
        self.assertTrue(np.allclose(
            nrbeval(g, [np.array([0.0]), np.array([0.5]), np.array([0.5])])[0],
            nrbeval(A, [np.array([0.0]), np.array([0.5]), np.array([0.5])])[0]))
        # value at u=1 → shared face = A(t=1) = B(t=0)
        c_g1 = nrbeval(g, [np.array([1.0]), np.array([0.5]), np.array([0.5])])[0]
        c_a1 = nrbeval(A, [np.array([1.0]), np.array([0.5]), np.array([0.5])])[0]
        c_b0 = nrbeval(B, [np.array([0.0]), np.array([0.5]), np.array([0.5])])[0]
        self.assertTrue(np.allclose(c_g1, c_a1))
        self.assertTrue(np.allclose(c_g1, c_b0))
        # value at u=2 → B right face (t=1)
        self.assertTrue(np.allclose(
            nrbeval(g, [np.array([2.0]), np.array([0.5]), np.array([0.5])])[0],
            nrbeval(B, [np.array([1.0]), np.array([0.5]), np.array([0.5])])[0]))

    def test_glue_degree_elevation_preserves_values(self):
        P1 = np.array([[0.0, 1.0, 2.0]])       # quadratic
        P2 = np.array([[2.0, 3.0, 4.0, 5.0]])  # cubic
        n1 = make_bezier_curve(P1, 2)
        n2 = make_bezier_curve(P2, 3)
        g = nrbglue(n1, n2)
        self.assertEqual(g.order, (4,))  # both elevated to cubic
        # glued u: first patch spans [0, 1] (local t = u),
        # second spans [1, 2] (local t = u - 1)
        us = np.array([0.4, 1.0, 1.6])
        cg, _ = nrbeval(g, us)
        c1, _ = nrbeval(n1, np.array([0.4, 1.0]))
        c2, _ = nrbeval(n2, np.array([0.6]))
        ref = np.r_[c1[0], c2[0]]
        self.assertTrue(np.allclose(cg[0], ref))

    def test_non_touching_patches_raise(self):
        A = make_box(0.0, 1.0)
        B = make_box(3.0, 4.0)
        with self.assertRaises(ValueError):
            nrbglue(A, B)

    def test_wrong_side_number_raises(self):
        A = make_box(0.0, 1.0)
        B = make_box(1.0, 2.0)
        with self.assertRaises(ValueError):
            nrbglue(A, B, 7, 1)

# ---------------------------------------------------------------------------
# mpStats
# ---------------------------------------------------------------------------
class TestMpStats(unittest.TestCase):
    def test_stats_on_two_boxes(self):
        A = make_box(0.0, 1.0)
        B = make_box(1.0, 2.0)
        old, buf = _capture_stdout()
        try:
            numH, numP = mpStats([A, B], 'VERBOSE', True)
        finally:
            _restore_stdout(old)
        self.assertEqual(numH, 2)
        # 5x5x5 control-point volume per box
        self.assertEqual(numP, 2 * 5 * 5 * 5)
        self.assertIn("multi-patch Stats: 2 Element(s) ; 250 Ctrl-pts", buf.getvalue())

    def test_incompatible_input(self):
        old, buf = _capture_stdout()
        try:
            numH, numP = mpStats(42)
        finally:
            _restore_stdout(old)
        self.assertIsNone(numH)
        self.assertIsNone(numP)
        self.assertIn("Incompatible Input Data", buf.getvalue())

# ---------------------------------------------------------------------------
# STL export (surf2stl path)
# ---------------------------------------------------------------------------
class TestStlExport(unittest.TestCase):
    def _plane_surface(self):
        P = np.zeros((3, 5, 5))
        for i in range(5):
            for j in range(5):
                P[0, i, j] = i
                P[1, i, j] = j
                P[2, i, j] = 0.0
        return make_bicubic_surface(P)

    @staticmethod
    def _read_stl(path):
        # Binary STL facet (50 bytes): normal (3f) + 3 vertices (9f) + uint16.
        # '<12fH' reads 12 floats (normal + 3*vertex) + 1 ushort = 50 bytes.
        with open(path, 'rb') as f:
            data = f.read()
        ntr = int.from_bytes(data[80:84], 'little')
        # verts[i] = (x1,y1,z1, x2,y2,z2, x3,y3,z3) -- the 9 vertex floats.
        verts = np.array([struct.unpack_from('<12fH', data, 84 + i * 50)[3:12]
                          for i in range(ntr)]).reshape(-1, 3)
        return ntr, verts, data

    def test_plane_fan_count_and_roundtrip(self):
        nrb = self._plane_surface()
        subs = 2  # 2 * 5 = 10 samples per parametric direction
        with tempfile.TemporaryDirectory() as td:
            # FILENAME must be a basename; FOLDERPATH (ending in sep) is the dir.
            old, buf = _capture_stdout()
            try:
                out = nrb_export_stl(nrb, 'SUBDIVISIONS', subs,
                                     'FILENAME', 'plane.stl',
                                     'FOLDERPATH', td + os.sep)
            finally:
                _restore_stdout(old)
            fn = out['path']
            self.assertTrue(os.path.isfile(fn))
            ntr, flat, data = self._read_stl(fn)
            # single surface: 9x9 quads = 162 triangles
            self.assertEqual(ntr, 9 * 9 * 2)
            self.assertEqual(len(data), 84 + ntr * 50)
            # all points on the z=0 plane, x in [0,4], y in [0,4]
            self.assertTrue(np.allclose(flat[:, 2], 0.0, atol=1e-9))
            self.assertTrue(np.all(flat[:, 0] >= -1e-9) and np.all(flat[:, 0] <= 4 + 1e-9))
            self.assertTrue(np.all(flat[:, 1] >= -1e-9) and np.all(flat[:, 1] <= 4 + 1e-9))

    def test_nan_row_skipped(self):
        nrb = self._plane_surface()
        # introduce NaN control points on one control-point row of the surface
        c = nrb.coefs.copy()
        c[:3, 1, :] = np.nan
        nrb_nan = Nrb(number=nrb.number, order=nrb.order, knots=nrb.knots, coefs=c)
        subs = 2  # 10 samples per direction
        with tempfile.TemporaryDirectory() as td:
            old, buf = _capture_stdout()
            try:
                out = nrb_export_stl(nrb_nan, 'SUBDIVISIONS', subs,
                                     'FILENAME', 'nan.stl',
                                     'FOLDERPATH', td + os.sep)
            finally:
                _restore_stdout(old)
            fn = out['path']
            ntr, flat, data = self._read_stl(fn)
            # The plane is a Bezier surface (global support): a single NaN
            # control point poisons every evaluated sample, so the writer
            # (MATLAB 'if isnan(...) continue') skips every quad -> 0 facets.
            # The faithful invariants: well-formed file, no NaN vertices.
            self.assertLessEqual(ntr, 8 * 8 * 2)  # never more than the full grid
            self.assertEqual(len(data), 84 + ntr * 50)
            self.assertTrue(np.all(np.isfinite(flat)))
            # and specifically: no facet may carry a NaN vertex (writer skips
            # NaN samples rather than writing them).
            self.assertEqual(ntr, 0)

# ---------------------------------------------------------------------------
# mp_geo_load / mp_geo_read_nurbs
# ---------------------------------------------------------------------------
class TestMpGeoLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(TEST_GEOM):
            raise unittest.SkipTest(f"Junct3.txt not found: {TEST_GEOM}")

    def test_read_nurbs_junct3(self):
        patches, boundaries, interfaces, subdomains = mp_geo_read_nurbs(TEST_GEOM)
        self.assertEqual(len(patches), 12)
        self.assertEqual(len(boundaries), 36)
        self.assertEqual(len(interfaces), 18)
        self.assertEqual(len(subdomains), 0)

    def test_geo_load_full_structure(self):
        (geometry, boundaries, interfaces, subdomains,
         boundary_interfaces) = mp_geo_load(TEST_GEOM)
        self.assertEqual(len(geometry), 12)
        self.assertEqual(len(boundaries), 36)
        self.assertEqual(len(interfaces), 18)
        self.assertEqual(len(boundary_interfaces), 72)
        p1 = geometry[0]
        self.assertEqual(p1['rdim'], 3)
        self.assertEqual(p1['nurbs'].number, (4, 4, 7))
        self.assertEqual(p1['nurbs'].order, (4, 4, 4))
        self.assertEqual(len(p1['boundary']), 6)
        for key in ('nurbs', 'rdim', 'dnurbs', 'dnurbs2', 'map',
                    'map_der', 'map_der2', 'boundary'):
            self.assertIn(key, p1)

    def test_map_consistency_with_nrbdeval(self):
        (geometry, *_rest) = mp_geo_load(TEST_GEOM)
        p1 = geometry[0]
        nurbs = p1['nurbs']
        tt = [np.array([0.2, 0.5, 0.8])] * 3
        # direct reference
        dnurbs = nrbderiv(nurbs)
        dnurbs2 = []
        for i in range(3):
            dnurbs2.append([])
            for j in range(3):
                dnurbs2[i].append(nrbderiv(dnurbs[i])[j])
        pnt_ref, jac_ref, hess_ref = nrbdeval(nurbs, dnurbs, dnurbs2, tt)
        pnt, jac, hess = p1['map_der2'](tt)
        self.assertTrue(np.allclose(pnt, pnt_ref[:3]))
        for a, b in zip(jac, jac_ref):
            self.assertTrue(np.allclose(a, b[:3]))
        for i in range(3):
            for j in range(3):
                self.assertTrue(np.allclose(hess[i][j], hess_ref[i][j][:3]))

    def test_boundary_entry(self):
        (geometry, *_rest) = mp_geo_load(TEST_GEOM)
        b0 = geometry[0]['boundary'][0]
        for key in ('nurbs', 'rdim', 'dnurbs', 'dnurbs2', 'map', 'map_der', 'map_der2'):
            self.assertIn(key, b0)
        self.assertEqual(b0['nurbs'].ndim, 2)
        # grid evaluation: 2 points per direction -> 2x2 = 4 points
        tt = [np.array([0.3, 0.7]), np.array([0.2, 0.8])]
        pnt, jac, hess = b0['map_der2'](tt)
        self.assertEqual(pnt.shape, (3, 2, 2))
        self.assertEqual(len(jac), 2)
        self.assertEqual(hess[0][0].shape, (3, 2, 2))

    def test_list_of_nurbs_input(self):
        A = make_box(0.0, 1.0)
        B = make_box(1.0, 2.0)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            (geometry, boundaries, interfaces, subdomains,
             boundary_interfaces) = mp_geo_load([A, B])
        self.assertTrue(any('nrbmultipatch' in str(w.message) for w in ws))
        self.assertEqual(len(geometry), 2)
        self.assertEqual(len(interfaces), 1)
        self.assertEqual(len(subdomains), 1)
        self.assertEqual(subdomains[0]['name'], 'SUBDOMAIN 1')

    def test_list_of_curves(self):
        n1 = make_bezier_curve(np.array([[0.0, 1.0, 2.0, 3.0]]), 3)
        n2 = make_bezier_curve(np.array([[3.0, 4.0, 5.0, 6.0]]), 3)
        (geometry, boundaries, interfaces, subdomains,
         boundary_interfaces) = mp_geo_load([n1, n2])
        self.assertEqual(len(geometry), 2)
        # x-only curve -> intrinsic embedding dimension 1 (MATLAB _detect_rdim)
        self.assertEqual(geometry[0]['rdim'], 1)
        # no boundary entries for 1D
        self.assertNotIn('boundary', geometry[0])
        self.assertEqual(len(boundary_interfaces), 0)
        tt = [np.array([0.0, 0.5, 1.0])]
        pnt, jac, hess = geometry[0]['map_der2'](tt)
        # x-only curve -> intrinsic embedding dim 1 -> pnt/hess are (1, N)
        self.assertEqual(pnt.shape, (1, 3))
        self.assertEqual(len(jac), 1)
        self.assertEqual(hess.shape, (1, 3))

    def test_bad_input(self):
        with self.assertRaises(NotImplementedError):
            mp_geo_load("file.mat")
        with self.assertRaises(ValueError):
            mp_geo_load(42)

# ---------------------------------------------------------------------------
# NaN discipline (standing rule, 2026-09-27)
# ---------------------------------------------------------------------------
class TestNoSpuriousNaN(unittest.TestCase):
    """NaN discipline (user standing rule).

    For WELL-FORMED NURBS geometry (finite control points, clamped knot
    vectors, positive weights) the NURBS kernels must NOT spuriously generate
    a NaN or Inf.  A NaN from a 0/0 division, an out-of-range index, or an
    un-elevated degree is a *defect* in the port -- NOT an injected input.
    (Injected NaN control points, as in the STL test above, are a different,
    handled case.)  We therefore drive the full kernel pipeline -- evaluation,
    first and second derivatives, knot insertion, degree elevation and glue --
    for 1D, 2D and 3D and assert every output is finite.

    This exercises findspan / basisfun / bspeval / bspderiv / bspkntins /
    bspdegelev through the public API.  It is the executable form of the
    "verify against MATLAB ground truth + The NURBS Book" requirement: the C
    kernels (basisfun.c, bspdegelev.c, ...) perform these divisions without
    guards, so on *well-formed* input they are guaranteed non-zero and no NaN
    is produced -- exactly what these tests lock in.
    """

    def _assert_finite(self, name, *arrays):
        for a in arrays:
            if a is None:
                continue
            a = np.asarray(a)
            if a.size == 0:
                continue
            self.assertTrue(
                bool(np.all(np.isfinite(a))),
                "SPURIOUS NaN/Inf generated in {0} (shape={1}, "
                "n_bad={2})".format(name, a.shape,
                                   int(np.sum(~np.isfinite(a)))))

    def test_curve_pipeline_no_nan(self):
        P = np.array([[0.0, 1.0, 2.0, 4.0], [0.0, 2.0, 4.0, 6.0]])
        nrb = make_bezier_curve(P, 3)
        us = np.linspace(0.0, 1.0, 13)

        cp, cw = nrbeval(nrb, us)
        self._assert_finite("curve nrbeval", cp, cw)

        dnurbs = nrbderiv(nrb)
        for d in dnurbs:
            dc, dw = nrbeval(d, us)
            self._assert_finite("curve 1st-der nrbeval", dc, dw)

        pnt, jac, _ = nrbdeval(nrb, dnurbs, tt=[us])
        self._assert_finite("curve nrbdeval pnt", pnt)
        for j in jac:
            self._assert_finite("curve nrbdeval jac", j)

        # Second derivatives: differentiate the first-derivative object again.
        d2 = nrbderiv(dnurbs[0])
        for dd in d2:
            dcc, _ = nrbeval(dd, us)
            self._assert_finite("curve 2nd-der nrbeval", dcc)

        ins = nrbkntins(nrb, np.array([0.25, 0.5, 0.75]))
        ic, _ = nrbeval(ins, us)
        self._assert_finite("curve nrbkntins nrbeval", ic)

        el = nrbdegelev(nrb, 1)
        ec, _ = nrbeval(el, us)
        self._assert_finite("curve nrbdegelev nrbeval", ec)

    def test_glue_pipeline_no_nan(self):
        n1 = make_bezier_curve(
            np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 2.0]]), 3)
        n2 = make_bezier_curve(
            np.array([[3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 6.0]]), 3)
        g = nrbglue(n1, n2)
        us = np.linspace(0.0, 2.0, 25)
        gc, gw = nrbeval(g, us)
        self._assert_finite("glued nrbeval", gc, gw)
        gdn = nrbderiv(g)
        for d in gdn:
            dc, _ = nrbeval(d, us)
            self._assert_finite("glued 1st-der nrbeval", dc)

    def test_surface_pipeline_no_nan(self):
        P = np.random.default_rng(0).random((3, 5, 5))
        nrb = make_bicubic_surface(P)
        tt = [np.linspace(0.0, 1.0, 7), np.linspace(0.0, 1.0, 5)]

        cp, cw = nrbeval(nrb, tt)
        self._assert_finite("surface nrbeval", cp, cw)

        dnurbs = nrbderiv(nrb)
        for d in dnurbs:
            dc, _ = nrbeval(d, tt)
            self._assert_finite("surface 1st-der nrbeval", dc)

        pnt, jac, _ = nrbdeval(nrb, dnurbs, tt=tt)
        self._assert_finite("surface nrbdeval pnt", pnt)
        for j in jac:
            self._assert_finite("surface nrbdeval jac", j)

        ins = nrbkntins(nrb, [np.array([0.5]), np.array([0.25, 0.75])])
        ic, _ = nrbeval(ins, tt)
        self._assert_finite("surface nrbkntins nrbeval", ic)

        el = nrbdegelev(nrb, 1)
        ec, _ = nrbeval(el, tt)
        self._assert_finite("surface nrbdegelev nrbeval", ec)

    def test_volume_pipeline_no_nan(self):
        P = np.random.default_rng(1).random((3, 5, 5, 5))
        nrb = make_bicubic_volume(P)
        u = np.linspace(0.0, 1.0, 5)
        tt = [u, u, u]

        cp, cw = nrbeval(nrb, tt)
        self._assert_finite("volume nrbeval", cp, cw)

        dnurbs = nrbderiv(nrb)
        for d in dnurbs:
            dc, _ = nrbeval(d, tt)
            self._assert_finite("volume 1st-der nrbeval", dc)

        pnt, jac, _ = nrbdeval(nrb, dnurbs, tt=tt)
        self._assert_finite("volume nrbdeval pnt", pnt)
        for j in jac:
            self._assert_finite("volume nrbdeval jac", j)

        ins = nrbkntins(nrb, [np.array([0.5]), np.array([0.5]),
                             np.array([0.5])])
        ic, _ = nrbeval(ins, tt)
        self._assert_finite("volume nrbkntins nrbeval", ic)

        el = nrbdegelev(nrb, 1)
        ec, _ = nrbeval(el, tt)
        self._assert_finite("volume nrbdegelev nrbeval", ec)

if __name__ == '__main__':
    unittest.main()
