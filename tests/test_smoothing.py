"""
A4 smoothing tests -- faithful literal port of the MATLAB ground truth
under ``smth_utils/multipatch/`` (read-only).

Covers:

* EDGE / CRNR / EXCEPTION weight formulas (``smthCtrlPtsTopologyAttribs.m``)
  on hand-built :class:`hexiga.smth.cpt.CPT` cases, including the dispatch
  rules (``ptShl > 0`` -> EDGE or CRNR; ``ptShl == 0`` -> EXCEPTION;
  ``isSmth`` points are left untouched).
* Degenerate guards: empty ``nnLbls`` (isolated point, valence 0) must leave
  the point unchanged and must not raise (valence == 0 is a degenerate case
  -- it means a single isolated control point with no lattice).
* ``refineTermKntsMPHexa`` on 1D / 2D / 3D NURBS embeddings (terminal knot
  interval refinement grows the control-point lattice).
* ``mpTurboSmooth`` end-to-end on the 32-patch socket: patch count and
  homogeneous shape preserved, no NaN/Inf, terminal knot spans preserved,
  and ``SKIP`` returns the input unchanged.

Run:
    cd C:\\HexIGA_Python
    & "C:\\...\\Python311\\python.exe" -m unittest tests.test_smoothing -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexiga.nurbs import Nrb
from hexiga.smth.cpt import CPT
from hexiga.smth.smth_cpt_topology_atrbs import (
    _get_edge_weights,
    _get_crnr_weights,
    _get_exception_weights,
    smth_ctrl_pts_topology_atrbs,
)
from hexiga.smth.refine_term_knts_mp import refine_term_knts_mp
from hexiga.smth.mp_turbo_smooth import mp_turbo_smooth
from hexiga.demo.demo_socket import genSocket

# ---------------------------------------------------------------------------
# small builders
# ---------------------------------------------------------------------------
def _cubic_clamped_knots():
    """cubic (order 4), 5 ctrl pts -> 9-knot vector; one interior span [0,0.5]."""
    return np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float)

def make_curve():
    k = _cubic_clamped_knots()
    c = np.zeros((4, 5))
    c[:3, :] = np.array([[0, 1, 2, 3, 4],
                         [0.0, 1.0, 2.0, 3.0, 4.0],
                         [0.0, 1.0, 2.0, 3.0, 4.0]])
    c[3, :] = 1.0
    return Nrb(number=(5,), order=(4,), knots=(k,), coefs=c)

def make_surface():
    k = _cubic_clamped_knots()
    c = np.zeros((4, 5, 5))
    c[3, :, :] = 1.0
    for i in range(5):
        for j in range(5):
            c[0, i, j] = i
            c[1, i, j] = j
            c[2, i, j] = 0.0
    return Nrb(number=(5, 5), order=(4, 4), knots=(k, k.copy()), coefs=c)

def make_volume():
    k = _cubic_clamped_knots()
    c = np.zeros((4, 5, 5, 5))
    c[3, :, :, :] = 1.0
    for i in range(5):
        for j in range(5):
            for m in range(5):
                c[0, i, j, m] = i
                c[1, i, j, m] = j
                c[2, i, j, m] = m
    return Nrb(number=(5, 5, 5), order=(4, 4, 4), knots=(k, k.copy(), k.copy()), coefs=c)

def _cpt(pt3D, nnIDs=(), nnLbls=(), ptShl=1.0, ptFrm=0.0, isSmth=False):
    return CPT(
        pt3D=np.asarray(pt3D, dtype=float),
        nnIDs=np.asarray(nnIDs, dtype=int),
        nnLbls=np.asarray(nnLbls, dtype=bool),
        ptShl=float(ptShl),
        ptFrm=float(ptFrm),
        isSmth=bool(isSmth),
    )

# ---------------------------------------------------------------------------
# weight formulas (MATLAB ground truth: smthCtrlPtsTopologyAttribs.m)
# ---------------------------------------------------------------------------
class TestWeightFormulas(unittest.TestCase):
    def test_edge_weights(self):
        alpha, beta = _get_edge_weights()
        self.assertEqual(alpha, 1 / 2)
        self.assertEqual(beta, 1 / 4)

    def test_crnr_weights(self):
        for v in (2, 3, 4, 6):
            alpha, beta, gamma = _get_crnr_weights(v)
            self.assertAlmostEqual(alpha, (v - 3) / v)
            self.assertAlmostEqual(beta, 2 / v ** 2)
            self.assertAlmostEqual(gamma, 1 / v ** 2)

    def test_exception_weights(self):
        for v in (1, 2, 3):
            alpha, beta = _get_exception_weights(v)
            self.assertAlmostEqual(alpha, 1 / 2)
            self.assertAlmostEqual(beta, 1 / (2 * v))

# ---------------------------------------------------------------------------
# smoothing dispatch + formulas on hand-built CPTs
# ---------------------------------------------------------------------------
class TestSmthCPT(unittest.TestCase):
    # NOTE: nnIDs are 1-BASED control-point ids (MATLAB convention):
    # neighbour k is mpCPT[k-1].  mpCPT[0] is always the point P under test.
    def test_edge_formula(self):
        # EDGE: P with 2 edge-neighbours -> alpha=1/2, beta=1/4
        P = np.array([1.0, 2.0, 3.0])
        N1 = np.array([0.0, 0.0, 0.0])
        N2 = np.array([4.0, 4.0, 4.0])
        mpCPT = [_cpt(P, nnIDs=[2, 3], nnLbls=[False, False], ptShl=1.0),
                 _cpt(N1), _cpt(N2)]
        out = smth_ctrl_pts_topology_atrbs(mpCPT)
        expected = 0.5 * P + 0.25 * (N1 + N2)   # = [1.5, 2.0, 2.5]
        np.testing.assert_allclose(out[0].pt3D, expected, rtol=0, atol=1e-14)
        self.assertFalse(out[0].isSmth)

    def test_edge_is_smth_when_coincident(self):
        # EDGE: neighbours coincide with P -> smoothed == P -> isSmth True
        P = np.array([1.0, 2.0, 3.0])
        mpCPT = [_cpt(P, nnIDs=[2, 3], nnLbls=[False, False], ptShl=1.0),
                 _cpt(P), _cpt(P)]
        out = smth_ctrl_pts_topology_atrbs(mpCPT)
        np.testing.assert_allclose(out[0].pt3D, P, rtol=0, atol=1e-14)
        self.assertTrue(out[0].isSmth)

    def test_crnr_formula(self):
        # CORNER: 6 neighbours, 3 edge (label False) + 3 face (label True)
        # -> valence v=3, alpha=(v-3)/v=0, beta=2/v^2, gamma=1/v^2
        P = np.array([1.0, 2.0, 3.0])
        E = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        F = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0])]
        nnIDs = [2, 3, 4, 5, 6, 7]
        nnLbls = [False, False, False, True, True, True]
        mpCPT = [_cpt(P, nnIDs=nnIDs, nnLbls=nnLbls, ptShl=2.0)]
        mpCPT += [_cpt(x) for x in (E + F)]
        out = smth_ctrl_pts_topology_atrbs(mpCPT)
        alpha, beta, gamma = _get_crnr_weights(3)   # alpha=0
        expected = alpha * P + beta * np.sum(E, axis=0) + gamma * np.sum(F, axis=0)
        np.testing.assert_allclose(out[0].pt3D, expected, rtol=0, atol=1e-14)

    def test_exception_formula(self):
        # EXCEPTION: ptShl==0, 2 edge-neighbours (valence v=2)
        # -> alpha=1/2, beta=1/(2v)=1/4
        P = np.array([2.0, 4.0, 6.0])
        N1 = np.array([0.0, 0.0, 0.0])
        N2 = np.array([1.0, 1.0, 1.0])
        mpCPT = [_cpt(P, nnIDs=[2, 3], nnLbls=[False, False], ptShl=0.0),
                 _cpt(N1), _cpt(N2)]
        out = smth_ctrl_pts_topology_atrbs(mpCPT)
        expected = 0.5 * P + 0.25 * (N1 + N2)   # = [1.25, 2.25, 3.25]
        np.testing.assert_allclose(out[0].pt3D, expected, rtol=0, atol=1e-14)

    def test_is_smth_point_is_skipped(self):
        P = np.array([1.0, 2.0, 3.0])
        mpCPT = [_cpt(P, nnIDs=[1, 2], nnLbls=[False, False], ptShl=1.0, isSmth=True),
                 _cpt(np.zeros(3)), _cpt(np.ones(3))]
        out = smth_ctrl_pts_topology_atrbs(mpCPT)
        np.testing.assert_allclose(out[0].pt3D, P, rtol=0, atol=1e-14)
        self.assertTrue(out[0].isSmth)

    def test_isolated_point_empty_nnlbls_no_raise(self):
        # Degenerate case (valence 0): isolated point, no lattice neighbours.
        # MATLAB guard `~isempty(nnLbls)` skips the formula; point unchanged.
        P = np.array([7.0, 8.0, 9.0])
        for ptShl, ptFrm in ((1.0, 0.0), (0.0, 0.0)):
            mpCPT = [_cpt(P, nnIDs=[], nnLbls=[], ptShl=ptShl, ptFrm=ptFrm)]
            out = smth_ctrl_pts_topology_atrbs(mpCPT)
            np.testing.assert_allclose(out[0].pt3D, P, rtol=0, atol=1e-14)
            self.assertTrue(out[0].isSmth)

    def test_crnr_unbalanced_labels_is_skipped(self):
        # sum(nnLbls) != sum(~nnLbls) -> MATLAB guard fails -> point unchanged
        P = np.array([1.0, 2.0, 3.0])
        nnLbls = [False, False, True]  # 2 edge vs 1 face -> unbalanced
        mpCPT = [_cpt(P, nnIDs=[1, 2, 3], nnLbls=nnLbls, ptShl=2.0),
                 _cpt(np.zeros(3)), _cpt(np.ones(3)), _cpt(np.full(3, 2.0))]
        out = smth_ctrl_pts_topology_atrbs(mpCPT)
        np.testing.assert_allclose(out[0].pt3D, P, rtol=0, atol=1e-14)

# ---------------------------------------------------------------------------
# refineTermKntsMPHexa (1D / 2D / 3D embeddings)
# ---------------------------------------------------------------------------
class TestRefineTermKnts(unittest.TestCase):
    def test_1d_curve_both_sides(self):
        H = make_curve()
        out = refine_term_knts_mp([H])[0]
        # interior span [0,0.5] refined at both ends -> +2 control points
        self.assertEqual(out.number, (7,))
        self.assertEqual(out.knots[0].size, 7 + 4)
        self.assertFalse(np.any(np.isnan(out.coefs)))
        self.assertTrue(np.allclose(out.knots[0][0], 0.0))
        self.assertTrue(np.allclose(out.knots[0][-1], 1.0))
        self.assertEqual(out.coefs.shape, (4, 7))

    def test_1d_curve_uref_only(self):
        # U direction only (USIDE default -1 = both sides) -> +2 control points
        H = make_curve()
        out = refine_term_knts_mp([H], "UREF", True, "VREF", False, "WREF", False)[0]
        self.assertEqual(out.number, (7,))

    def test_2d_surface_both_sides(self):
        H = make_surface()
        out = refine_term_knts_mp([H])[0]
        self.assertEqual(out.number, (7, 7))
        self.assertEqual(len(out.knots), 2)
        self.assertFalse(np.any(np.isnan(out.coefs)))
        self.assertEqual(out.coefs.shape, (4, 7, 7))

    def test_2d_surface_vref_only(self):
        H = make_surface()
        out = refine_term_knts_mp([H], "UREF", False, "VREF", True, "WREF", False)[0]
        self.assertEqual(out.number, (5, 7))

    def test_3d_volume_both_sides(self):
        H = make_volume()
        out = refine_term_knts_mp([H])[0]
        self.assertEqual(out.number, (7, 7, 7))
        self.assertFalse(np.any(np.isnan(out.coefs)))
        self.assertEqual(out.coefs.shape, (4, 7, 7, 7))

    def test_3d_volume_uref_only(self):
        # U direction only (USIDE default -1 = both sides) -> +2 in u, +0 in v,w
        H = make_volume()
        out = refine_term_knts_mp([H], "UREF", True, "VREF", False, "WREF", False)[0]
        self.assertEqual(out.number, (7, 5, 5))

    def test_uside_only_first_span(self):
        # USIDE 0 -> only the [0, minKnt] side refined -> +1 ctrl point
        H = make_curve()
        out = refine_term_knts_mp([H], "UREF", True, "USIDE", 0)[0]
        self.assertEqual(out.number, (6,))

# ---------------------------------------------------------------------------
# mpTurboSmooth end-to-end on the 32-patch socket
# ---------------------------------------------------------------------------
class TestMPTurboSmoothSocket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mpIn = genSocket(3, 4)
        cls.out = mp_turbo_smooth(cls.mpIn, "SHELLS", [3, 4], "MAXITER", 3)

    def test_patch_count_preserved(self):
        self.assertEqual(len(self.mpIn), 32)
        self.assertEqual(len(self.out), 32)

    def test_homogeneous_coefs_no_nan(self):
        for H in self.out:
            self.assertEqual(H.dim, 4)
            self.assertEqual(H.coefs.shape[0], 4)
            self.assertFalse(np.any(np.isnan(H.coefs)), "NaN in smoothed coefs")
            self.assertFalse(np.any(np.isinf(H.coefs)), "Inf in smoothed coefs")

    def test_terminal_knot_spans_preserved(self):
        for H in self.out:
            for i, (k, n, o) in enumerate(zip(H.knots, H.number, H.order)):
                if n > 1:
                    self.assertEqual(k.size, n + o)
                    self.assertTrue(np.all(k[:o] == 0.0))
                    self.assertTrue(np.all(k[-o:] == 1.0))

    def test_skip_returns_input_unchanged(self):
        mp = genSocket(3, 4)
        out = mp_turbo_smooth(mp, "SKIP", True)
        self.assertEqual(len(out), 32)
        for a, b in zip(mp, out):
            np.testing.assert_allclose(a.coefs, b.coefs, rtol=0, atol=0)
            for ka, kb in zip(a.knots, b.knots):
                np.testing.assert_allclose(ka, kb, rtol=0, atol=0)

    def test_default_smooth_runs(self):
        # no SHELLS: full smoothing of every shell -- slower but must converge
        out = mp_turbo_smooth(self.mpIn, "MAXITER", 3)
        self.assertEqual(len(out), 32)
        for H in out:
            self.assertFalse(np.any(np.isnan(H.coefs)))

# ---------------------------------------------------------------------------
# LIGHT E2E smoke tests on small CAD/synthetic demos (keep the suite fast)
# ---------------------------------------------------------------------------
class TestDemosSmoothSmoke(unittest.TestCase):
    """Very small end-to-end E2E smoothing tests that exercise the
    ``mp_turbo_smooth`` wiring inside the demo drivers.

    These are intentionally LIGHT (1-2 demos only) because full CAD
    demos with TURBO smoothing can take minutes each.  The correctness
    bar per the user is: *visibly smooth* and *no NaNs*; the patch
    count must be preserved.
    """

    def test_cross6_smooth(self):
        from hexiga.demo.demo_cross6 import demoCross6
        mp = demoCross6("SMOOTH", True)
        self.assertEqual(len(mp), 7)
        for H in mp:
            self.assertFalse(np.any(np.isnan(H.coefs)))
            self.assertFalse(np.any(np.isinf(H.coefs)))

    def test_cross6_nosmooth_reference(self):
        from hexiga.demo.demo_cross6 import demoCross6
        mp_ns = demoCross6("SMOOTH", False)
        self.assertEqual(len(mp_ns), 7)
        for H in mp_ns:
            self.assertFalse(np.any(np.isnan(H.coefs)))

# ---------------------------------------------------------------------------
# Capped-iteration E2E smoothing smoke tests for every remaining demo.
#
# Per the user directive: "test also the smoothing for the remaining synthetic
# and cad geometries. They do not have to iterate completely for all 100
# maxIteration, few cycles may be enough to check that they work end-2-end with
# the warning that the final smoothing result should be run completely up until
# maxIteration for a valid visual result."
#
# The demo drivers do NOT forward a ``MAXITER`` argument to
# ``mp_turbo_smooth``, so we temporarily monkeypatch each module's
# ``mp_turbo_smooth`` symbol with a wrapper that injects ``MAXITER=2`` (or 3).
# This exercises the REAL driver code path (slicing, merges, tuple returns,
# SHELLS / INOUTLETSSIDES / degelev) while bounding the number of smoothing
# sweeps.  The assertion is a liveness/NaN check: correct patch count + no
# NaN/Inf.  It is NOT a visual-correctness check -- a valid visual result
# requires a full ``MAXITER=100`` run (see the note above).
# ---------------------------------------------------------------------------
class TestDemosSmoothCapped(unittest.TestCase):
    """E2E smoothing smoke test, one per demo driver.

    Each test monkeypatches the demo module's ``mp_turbo_smooth`` to cap the
    number of smoothing iterations, drives the demo with ``("SMOOTH", True)``,
    and asserts the expected patch count and the absence of NaN/Inf in every
    returned patch's homogeneous coefficients.
    """

    MAXITER = 3  # few cycles -- liveness check only, not a visual result

    def _flatten(self, result):
        """Flatten a demo's return value (list or tuple-of-lists) to patches."""
        if isinstance(result, tuple):
            return [h for part in result for h in part]
        return list(result)

    def _capped_run(self, module, driver, expected, maxiter=None):
        """Run ``driver`` on ``module`` with ``mp_turbo_smooth`` capped."""
        real = module.mp_turbo_smooth
        mi = self.MAXITER if maxiter is None else maxiter

        def cap(mpHexa, *pairs):
            return real(mpHexa, *pairs, "MAXITER", mi)

        module.mp_turbo_smooth = cap
        try:
            out = driver()
        finally:
            module.mp_turbo_smooth = real

        flat = self._flatten(out)
        self.assertEqual(len(flat), expected)
        for H in flat:
            self.assertEqual(H.dim, 4)
            self.assertEqual(H.coefs.shape[0], 4)
            self.assertFalse(np.any(np.isnan(H.coefs)), "NaN in smoothed coefs")
            self.assertFalse(np.any(np.isinf(H.coefs)), "Inf in smoothed coefs")
        return out

    def test_torus_smooth(self):
        import hexiga.demo.demo_torus as m
        self._capped_run(m, lambda: m.demoTorus("SMOOTH", True), 4)

    def test_frame_smooth(self):
        import hexiga.demo.demo_frame as m
        self._capped_run(m, lambda: m.demoFrame("SMOOTH", True), 8)

    def test_twist_smooth(self):
        import hexiga.demo.demo_twist as m
        self._capped_run(m, lambda: m.demoTwist("SMOOTH", True), 1)

    def test_quadball_smooth(self):
        import hexiga.demo.demo_quadball as m
        self._capped_run(m, lambda: m.demoQuadball("SMOOTH", True), 1)

    def test_egg_lw_smooth(self):
        import hexiga.demo.demo_egg_lw as m
        out = self._capped_run(m, lambda: m.demoEggLW("SMOOTH", True), 20)
        self.assertEqual(len(out[0]), 4)   # lumen
        self.assertEqual(len(out[1]), 16)  # wall

    def test_socket_smooth(self):
        import hexiga.demo.demo_socket as m
        self._capped_run(m, lambda: m.demoSocketCAD("SMOOTH", True), 32)

    def test_hooks_smooth(self):
        import hexiga.demo.demo_hooks as m
        self._capped_run(m, lambda: m.demoHooksCAD("SMOOTH", True), 14)

    def test_stent_smooth(self):
        # 120 patches -- heaviest demo; cap iterations to keep the suite light.
        import hexiga.demo.demo_stent as m
        self._capped_run(m, lambda: m.demoStentCAD("SMOOTH", True), 120)

    def test_gear_smooth(self):
        import hexiga.demo.demo_gear as m
        self._capped_run(m, lambda: m.demoGearCAD("SMOOTH", True), 54)

    def test_tpipe_smooth(self):
        import hexiga.demo.demo_tpipe as m
        # merge=True default -> 12 merged patches.
        self._capped_run(m, lambda: m.demoTPipeCAD("SMOOTH", True), 12)

    def test_turbine_smooth(self):
        import hexiga.demo.demo_turbine as m
        out = self._capped_run(m, lambda: m.demoTurbineCAD("SMOOTH", True), 84)
        self.assertEqual(len(out[0]), 48)  # blades
        self.assertEqual(len(out[1]), 36)  # air

    def test_pinocchio_smooth(self):
        import hexiga.demo.demo_pinocchio as m
        # version 1 default -> genGraftLarge -> 24 patches.
        self._capped_run(m, lambda: m.demoPinocchio("SMOOTH", True), 24)

    def test_plate_smooth(self):
        import hexiga.demo.demo_plate as m
        self._capped_run(m, lambda: m.demoPlateCAD("SMOOTH", True), 72)

    def test_torus_turbo_smooth(self):
        """Also cover the TURBO cycle path (initial + TURBOCYCLES sweeps), capped."""
        import hexiga.demo.demo_torus as m
        self._capped_run(m, lambda: m.demoTorus("SMOOTH", True, "TURBOSMOOTH", True), 4)

class TestTorusCPTTopologyParity(unittest.TestCase):
    """Regression guard for the torus CPT-extraction topology.

    The expected values below are **byte-for-byte ground truth produced by
    MATLAB** (``extractCtrlPtsTopologyFromMultiPatchHexa`` on the raw 8-patch
    torus, via ``probe_scp.m``). A faithful port must reproduce them exactly.
    This is the invariant that was violated when ``_get_simplex6`` /
    ``_get_simplex12`` were defined as 6x3 / 12x3 (rows = offsets) instead of
    the MATLAB 3x6 / 3x12 (rows = axes du/dv/dw), which silently corrupted the
    EDGE / CRNR neighbor lists and hence the whole smoothed geometry.
    """

    @classmethod
    def setUpClass(cls):
        import hexiga.demo.demo_torus as m
        from hexiga.smth.extract_cpt_topology_mp import (
            extract_ctrl_pts_topology_from_multipatch_hexa,
        )
        from hexiga.smth.append_smth_atrbs_new import append_smth_atrbs_new
        mpHexa = m.genTorus()
        mpCPT, mpCPTatrb, _ = extract_ctrl_pts_topology_from_multipatch_hexa(
            mpHexa, "INOUTLETSSIDES", [], "EXEPTINOUTLETSPATCHES", [],
            "EXEPTPATCHSIDEPAIRS", [])
        cls.mpCPT = append_smth_atrbs_new(
            mpCPT, mpCPTatrb, "SHELLS", [], "FREEZESHELLBOUNDARY", False)

    def test_cpt_count(self):
        # 4+4+4 = 12 ctrl pts / patch * 8 patches = 96, doubled by the 48
        # interior CPTs and further split -> 384 total (MATLAB-verified).
        self.assertEqual(len(self.mpCPT), 384)

    def test_neighbor_count_distribution(self):
        # Histogram of per-point EDGE/CRNR neighbor counts. MATLAB (via
        # probe_scp.m) == Python after the simplex fix: {0:128, 2:128,
        # 6:96, 8:32} -- 128 non-smoothable isolated points, the rest with
        # 2 / 6 / 8 neighbors.
        from collections import Counter
        hist = Counter(len(c.nnIDs) for c in self.mpCPT)
        self.assertEqual(dict(sorted(hist.items())), {0: 128, 2: 128, 6: 96, 8: 32})

    def test_corner_point_neighbor_ids(self):
        # CPT #1 is a cube corner (valence 8). MATLAB ground-truth neighbor
        # IDs (probe_scp.m): 2, 5, 17, 18, 21, 369, 370, 373; labels
        # 0, 0, 0, 1, 1, 0, 1, 1. The exact ID set is the fingerprint of
        # the correct 3x6 (axes-row) simplex; with the transposition bug
        # the EDGE list collapsed (e.g. point 1 -> [17] only).
        p1 = self.mpCPT[0]
        self.assertEqual(len(p1.nnIDs), 8)
        self.assertEqual(set(int(x) for x in p1.nnIDs), {2, 5, 17, 18, 21, 369, 370, 373})

    def test_no_point_has_degenerate_edge_list(self):
        # With the transposition bug, some points collapsed to a single EDGE
        # neighbor (e.g. point 1 had only nn=[17] instead of 8 neighbors).
        # Assert every non-isolated point has a well-formed neighbor list of
        # valid 1-based point indices into the 384.
        for c in self.mpCPT:
            if len(c.nnIDs) > 0:
                self.assertGreater(len(c.nnIDs), 1)
                self.assertTrue(all(1 <= int(x) <= 384 for x in c.nnIDs))

if __name__ == "__main__":
    unittest.main()
