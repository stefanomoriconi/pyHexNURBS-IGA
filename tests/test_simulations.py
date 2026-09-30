#!/usr/bin/env python3
"""
tests/test_simulations.py -- M4: stdlib ``unittest`` suite for
``hexiga.iga.simulations`` (the three renamed run*Sim wrappers).

Covers the literal MATLAB->Python port of:

  * runFluidSim_GeoPDEsAPI.m      -> run_fluid_simulation
  * runElastSim_GeoPDEsAPI.m      -> run_elastic_simulation
  * runMaxwellEigSim_GeoPDEsAPI.m -> run_maxwell_simulation

Oracle conventions:
  * MATLAB is ground truth (the ``iga_utils/run*Sim_GeoPDEsAPI.m`` files).
  * The external FEM backend is isolated behind ``MockGeoPDEsBackend``;
    the tests exercise the wrapper's bookkeeping (BC selection,
    method_data assembly, field propagation, failure banner).
  * The interactive ``questdlg`` in the MATLAB Maxwell wrapper is
    abstracted behind the ``force_all_prompt`` callable; the default
    is auto-'Yes' (headless).
"""

from __future__ import annotations
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexiga.iga import (  # noqa: E402
    BoundaryConditions,
    SideAttribsRow,
    VolumetricSource,
    FluidData,
    ElasticData,
    MaxwellData,
    MockGeoPDEsBackend,
    run_fluid_simulation,
    run_elastic_simulation,
    run_maxwell_simulation,
    mp_stats,
)
from hexiga.iga.simulations import (  # noqa: E402
    _make_f,
    _build_bc_closures,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_fluid_data() -> FluidData:
    """side 1,2 Dirichlet (zero); side 3 leading m=2.5 with versor (1,0,0)."""
    return FluidData(
        GeomFileName="fluid_geometry",
        BoundaryConditions=BoundaryConditions(
            dirichlet=SideAttribsRow(
                IDs=np.array([1, 2]),
                m=np.zeros(2), mx=np.zeros(2), my=np.zeros(2), mz=np.zeros(2),
            ),
            leading=SideAttribsRow(
                IDs=np.array([3]),
                m=np.array([2.5]),
                mx=np.array([1.0]), my=np.array([0.0]), mz=np.array([0.0]),
            ),
        ),
        VolumetricSource=VolumetricSource(f=1.0, fx=1.0, fy=0.0, fz=0.0),
    )

def make_elastic_data() -> ElasticData:
    return ElasticData(
        GeomFileName="elastic_geometry",
        BoundaryConditions=BoundaryConditions(
            dirichlet=SideAttribsRow(
                IDs=np.array([1, 2]),
                m=np.zeros(2), mx=np.zeros(2), my=np.zeros(2), mz=np.zeros(2),
            ),
            leading=SideAttribsRow(
                IDs=np.array([3]),
                m=np.array([2.5]),
                mx=np.array([1.0]), my=np.array([0.0]), mz=np.array([0.0]),
            ),
        ),
        VolumetricSource=VolumetricSource(f=1.0, fx=1.0, fy=0.0, fz=0.0),
    )

def make_elastic_data_no_dirichlet() -> ElasticData:
    """MATLAB quirk: empty dirichlet.IDs -> drchlt falls back to leading."""
    return ElasticData(
        GeomFileName="elastic_geometry",
        BoundaryConditions=BoundaryConditions(
            dirichlet=SideAttribsRow(),  # empty
            leading=SideAttribsRow(
                IDs=np.array([3]),
                m=np.array([2.5]),
                mx=np.array([1.0]), my=np.array([0.0]), mz=np.array([0.0]),
            ),
        ),
        VolumetricSource=VolumetricSource(f=1.0, fx=1.0, fy=0.0, fz=0.0),
    )

def make_maxwell_data() -> MaxwellData:
    return MaxwellData(
        GeomFileName="maxwell_geometry",
        BoundaryConditions=BoundaryConditions(
            dirichlet=SideAttribsRow(
                IDs=np.array([1]),
                m=np.zeros(1), mx=np.zeros(1), my=np.zeros(1), mz=np.zeros(1),
            ),
            leading=SideAttribsRow(
                IDs=np.array([2]),
                m=np.array([2.5]),
                mx=np.array([1.0]), my=np.array([0.0]), mz=np.array([0.0]),
            ),
            floating=SideAttribsRow(
                IDs=np.array([3]),
                m=np.array([np.nan]), mx=np.array([np.nan]),
                my=np.array([np.nan]), mz=np.array([np.nan]),
            ),
        ),
        VolumetricSource=VolumetricSource(),
    )

class _BrokenBackend:
    """Backend whose ``load_geometry`` raises, to exercise the except branch."""

    def load_geometry(self, geo_name):
        raise RuntimeError("broken backend")

# ---------------------------------------------------------------------------
# Fluid (Stokes) wrapper
# ---------------------------------------------------------------------------
class TestFluidSim(unittest.TestCase):
    def test_all_dirichlet_populates_fluid_data(self):
        fd = make_fluid_data()
        backend = MockGeoPDEsBackend()

        buf = io.StringIO()
        with redirect_stdout(buf):
            out = run_fluid_simulation(fd, backend=backend)

        # returned object is the same instance (MATLAB: in-place struct edit)
        self.assertIs(out, fd)
        # geometry / spaces / fields are all populated
        self.assertIsNotNone(fd.geometry)
        self.assertIsNotNone(fd.space_v)
        self.assertIsNotNone(fd.space_p)
        self.assertIsNotNone(fd.vel)
        self.assertIsNotNone(fd.press)
        # shapes match the mock backend (ndof_v=10, ndof_p=5)
        self.assertEqual(np.asarray(fd.vel).shape, (10, 1))
        self.assertEqual(np.asarray(fd.press).shape, (5, 1))
        # solver ran (no failure banner)
        self.assertNotIn("FAILED!", buf.getvalue())
        self.assertIn("Stokes Fluid Simulation: Completed!", buf.getvalue())

    def test_failure_banner(self):
        fd = make_fluid_data()
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = run_fluid_simulation(fd, backend=_BrokenBackend())
        # dataclass unchanged (the except branch does not modify fields)
        self.assertIs(out, fd)
        self.assertIsNone(fd.vel)
        self.assertIn("FAILED!", buf.getvalue())

    def test_leading_side_h_nonzero(self):
        """h on a leading side must be m .* (mx, my, mz)."""
        B = make_fluid_data().BoundaryConditions
        drchlt = np.array([1, 2, 3])  # dirichlet + leading
        h, g = _build_bc_closures(B, drchlt, np.array([3]))
        # leading side 3 has m=2.5, versor (1,0,0) -> h(3) = (2.5, 0, 0)
        out_h3 = np.asarray(h(0.0, 0.0, 0.0, iside=3)).ravel()
        np.testing.assert_allclose(out_h3, [2.5, 0.0, 0.0])
        # g on the same side is also (2.5, 0, 0)
        out_g3 = np.asarray(g(0.0, 0.0, 0.0, iside=3)).ravel()
        np.testing.assert_allclose(out_g3, [2.5, 0.0, 0.0])
        # Dirichlet side 1 is zero
        out_h1 = np.asarray(h(0.0, 0.0, 0.0, iside=1)).ravel()
        np.testing.assert_allclose(out_h1, [0.0, 0.0, 0.0])

# ---------------------------------------------------------------------------
# Elasticity wrapper
# ---------------------------------------------------------------------------
class TestElasticSim(unittest.TestCase):
    def test_elastic_all_dirichlet(self):
        ed = make_elastic_data()
        backend = MockGeoPDEsBackend()

        buf = io.StringIO()
        with redirect_stdout(buf):
            out = run_elastic_simulation(ed, backend=backend)

        self.assertIs(out, ed)
        self.assertIsNotNone(ed.u)
        self.assertIsNotNone(ed.space)
        self.assertIsNotNone(ed.lambda_lame)
        self.assertIsNotNone(ed.mu_lame)
        # u shape (ndof_e=12, 1)
        self.assertEqual(np.asarray(ed.u).shape, (12, 1))
        self.assertNotIn("FAILED!", buf.getvalue())
        self.assertIn("Linear Elasticity Simulation: Completed!", buf.getvalue())

    def test_elastic_no_dirichlet_fallback(self):
        """MATLAB quirk: empty dirichlet.IDs -> drchlt = leading.IDs."""
        ed = make_elastic_data_no_dirichlet()
        backend = MockGeoPDEsBackend()

        buf = io.StringIO()
        with redirect_stdout(buf):
            out = run_elastic_simulation(ed, backend=backend)

        self.assertIs(out, ed)
        self.assertIsNotNone(ed.u)
        self.assertIsNotNone(ed.space)
        self.assertEqual(np.asarray(ed.u).shape, (12, 1))
        self.assertNotIn("FAILED!", buf.getvalue())

    def test_failure_banner(self):
        ed = make_elastic_data()
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = run_elastic_simulation(ed, backend=_BrokenBackend())
        self.assertIs(out, ed)
        self.assertIsNone(ed.u)
        self.assertIn("FAILED!", buf.getvalue())

# ---------------------------------------------------------------------------
# Maxwell wrapper
# ---------------------------------------------------------------------------
class TestMaxwellSim(unittest.TestCase):
    def test_maxwell_eigs_branch(self):
        md = make_maxwell_data()
        backend = MockGeoPDEsBackend()

        buf = io.StringIO()
        with redirect_stdout(buf):
            out = run_maxwell_simulation(md, backend=backend)

        self.assertIs(out, md)
        self.assertIsNotNone(md.eigv)
        self.assertIsNotNone(md.eigf)
        self.assertIsNotNone(md.space)
        self.assertEqual(np.asarray(md.eigf).shape, (12, 4))
        # mock: eigv = [7, 8, 9, 10] (eigs branch, eigenvalues closest to 0.1)
        np.testing.assert_allclose(np.sort(np.asarray(md.eigv)), [7.0, 8.0, 9.0, 10.0])
        self.assertNotIn("FAILED!", buf.getvalue())
        self.assertIn("Maxwell Eigenfunction Simulation: Completed!", buf.getvalue())

    def test_maxwell_force_all_prompt_yes(self):
        """force_all_prompt returning 'Yes' -> forced re-solve (eig branch)."""
        md = make_maxwell_data()
        backend = MockGeoPDEsBackend()

        # To hit the prompt branch, we need a geometry where the eigs
        # branch finds no non-zero eigenvalues.  With the mock backend the
        # eigs branch always finds 4 non-zero eigenvalues, so the prompt
        # branch is NOT triggered in this fixture.  We exercise the
        # prompt code path by patching the solver to return all-zero
        # eigenvalues the first time, then normal eigenvalues the second.
        calls = {"n": 0}

        real_solve = __import__("hexiga.iga.maxwell", fromlist=[
            "mp_solve_maxwell_eig_complete"
        ]).mp_solve_maxwell_eig_complete

        def _patched(problem_data, method_data, backend=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # Force the prompt branch: eigv = [0, 0, 0, 0]
                eigv = np.array([0.0, 0.0, 0.0, 0.0])
                eigf = np.zeros((12, 4))
                geom = object()
                return geom, object(), object(), eigv, eigf, 0.0, 0.0
            return real_solve(problem_data, method_data, backend=backend)

        import hexiga.iga.simulations as sim
        orig = sim.mp_solve_maxwell_eig_complete
        sim.mp_solve_maxwell_eig_complete = _patched
        try:
            buf = io.StringIO()
            prompted = []

            def _prompt(msg: str) -> str:
                prompted.append(msg)
                return "Yes"

            with redirect_stdout(buf):
                out = run_maxwell_simulation(md, backend=backend,
                                             force_all_prompt=_prompt)
        finally:
            sim.mp_solve_maxwell_eig_complete = orig

        self.assertIs(out, md)
        self.assertEqual(len(prompted), 1)
        self.assertIn("Force All", prompted[0])
        # after the forced re-solve, eigv comes from the real solver.
        # The forced re-solve uses the forceAll/eig branch -> all 6 interior
        # eigenvalues [7,8,9,10,11,12] (not the 4 from the eigs branch).
        self.assertEqual(calls["n"], 2)
        np.testing.assert_allclose(
            np.sort(np.asarray(md.eigv)), [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    def test_maxwell_force_all_prompt_no(self):
        """force_all_prompt returning 'No' -> no re-solve, eigv stays zero."""
        md = make_maxwell_data()
        backend = MockGeoPDEsBackend()

        calls = {"n": 0}

        real_solve = __import__("hexiga.iga.maxwell", fromlist=[
            "mp_solve_maxwell_eig_complete"
        ]).mp_solve_maxwell_eig_complete

        def _patched(problem_data, method_data, backend=None):
            calls["n"] += 1
            eigv = np.array([0.0, 0.0, 0.0, 0.0])
            eigf = np.zeros((12, 4))
            return object(), object(), object(), eigv, eigf, 0.0, 0.0

        import hexiga.iga.simulations as sim
        orig = sim.mp_solve_maxwell_eig_complete
        sim.mp_solve_maxwell_eig_complete = _patched
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                out = run_maxwell_simulation(md, backend=backend,
                                             force_all_prompt=lambda m: "No")
        finally:
            sim.mp_solve_maxwell_eig_complete = orig

        self.assertIs(out, md)
        self.assertEqual(calls["n"], 1)  # no re-solve
        np.testing.assert_allclose(np.asarray(md.eigv),
                                   np.zeros(4))

    def test_failure_banner(self):
        md = make_maxwell_data()
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = run_maxwell_simulation(md, backend=_BrokenBackend())
        self.assertIs(out, md)
        self.assertIsNone(md.eigv)
        self.assertIn("FAILED!", buf.getvalue())

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
class TestHelpers(unittest.TestCase):
    def test_mp_stats_none(self):
        numH, numP = mp_stats(None)
        self.assertEqual(numH, 0)
        self.assertEqual(numP, 0)

    def test_make_f_zero(self):
        f = _make_f(None)
        out = np.asarray(f(0.0, 0.0, 0.0)).ravel()
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0])

    def test_make_f_scalar(self):
        f = _make_f(VolumetricSource(f=2.0, fx=1.0, fy=0.0, fz=-1.0))
        # x = a single point -> shape () -> out shape (3,)
        out = np.asarray(f(0.5)).ravel()
        np.testing.assert_allclose(out, [2.0, 0.0, -2.0])
        # x = a 2x3 grid -> out shape (3, 2, 3)
        x = np.zeros((2, 3))
        out = np.asarray(f(x))
        self.assertEqual(out.shape, (3, 2, 3))
        np.testing.assert_allclose(out[0], np.full((2, 3), 2.0))
        np.testing.assert_allclose(out[1], np.zeros((2, 3)))
        np.testing.assert_allclose(out[2], np.full((2, 3), -2.0))

if __name__ == "__main__":
    unittest.main(verbosity=2)
