#!/usr/bin/env python3
"""
tests/test_iga.py — M2: stdlib ``unittest`` suite for ``hexiga.iga``.

Covers the literal MATLAB->Python port of the IGA (GeoPDEs-agnostic) layer:

  * converters      : convert_geom_data_2_fluid_data / _elastic_data
  * boundary_map    : map_boundaries_h_to_h_sbdv
  * stokes          : mp_solve_stokes_complete (saddle point + BCs)
  * elasticity      : mp_solve_linear_elasticity_complete

Oracle conventions:
  * MATLAB is ground truth (the ``iga_utils/*.m`` files).
  * All DOF indices and side IDs are 1-based in the MATLAB / mock world;
    the solvers normalise to 0-based internally.
  * The external GeoPDEs backend is isolated behind :class:`MockGeoPDEsBackend`,
    so the tests exercise the ported assembly / BC / solve logic, not GeoPDEs.
"""

from __future__ import annotations
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexiga.iga import (  # noqa: E402
    SideAttribs,
    SideHandle,
    VolumetricSource,
    MockGeoPDEsBackend,
    convert_geom_data_2_fluid_data,
    convert_geom_data_2_elastic_data,
    map_boundaries_h_to_h_sbdv,
    mp_solve_stokes_complete,
    mp_solve_linear_elasticity_complete,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_side(is_dirichlet: bool, mag: float = 0.0,
              n3d=(0.0, 0.0, 0.0)) -> SideAttribs:
    return SideAttribs(
        Srf=None,
        pt3D=np.array([0.0, 0.0, 0.0]),
        n3D=np.asarray(n3d, dtype=float),
        mag=mag,
        isDirichlet=is_dirichlet,
    )

def make_geom_data(n_sides=3):
    """A minimal GeomDATA mixing dirichlet / leading / floating sides."""
    # side 1: Dirichlet
    # side 2: leading (handle isPersistent=True) with a versor + magnitude
    # side 3: floating (handle isPersistent=False)
    bndrs_attrbs = [
        make_side(True, mag=0.0),
        make_side(False, mag=2.5, n3d=(1.0, 2.0, 3.0)),
        make_side(False, mag=0.0),
    ]
    bndrs_handles = [
        SideHandle(isPersistent=False),
        SideHandle(isPersistent=True),
        SideHandle(isPersistent=False),
    ]
    vol_source = VolumetricSource(f=1.0, fx=0.0, fy=0.0, fz=0.0)
    geom = types.SimpleNamespace(
        mpHexa="some_geometry",
        bndrsAttrbs=bndrs_attrbs,
        bndrsHandles=bndrs_handles,
        volSource=vol_source,
    )
    return geom

def make_method_data_stokes() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        element_name="TH",
        degree=[2, 2, 2],
        regularity=[1, 1, 1],
        nsub=[1, 1, 1],
        nquad=[4, 4, 4],
    )

def make_problem_data_stokes(nmnn_sides=()) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        geo_name="g",
        drchlt_sides=[1, 2],
        nmnn_sides=list(nmnn_sides),
        f=1,
        g=None,
        h=1,
        viscosity=1.0,
    )

def make_method_data_elas() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        degree=[3, 3, 3],
        regularity=[2, 2, 2],
        nsub=[1, 1, 1],
        nquad=[4, 4, 4],
    )

def make_problem_data_elas(nmnn_sides=()) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        geo_name="g",
        drchlt_sides=[1, 2],
        nmnn_sides=list(nmnn_sides),
        lambda_lame=0.5,
        mu_lame=1.0,
        f=1,
        g=None,
        h=1,
    )

# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------
class TestConverters(unittest.TestCase):
    def test_fluid_bc_split(self):
        geom = make_geom_data()
        out = convert_geom_data_2_fluid_data(geom)

        # Dirichlet: side 1 only, zero magnitude/normal.
        self.assertEqual(out.BoundaryConditions.dirichlet.IDs.tolist(), [1])
        self.assertEqual(out.BoundaryConditions.dirichlet.m.tolist(), [0.0])
        self.assertEqual(out.BoundaryConditions.dirichlet.mx.tolist(), [0.0])
        self.assertEqual(out.BoundaryConditions.dirichlet.my.tolist(), [0.0])
        self.assertEqual(out.BoundaryConditions.dirichlet.mz.tolist(), [0.0])

        # Leading: side 2, magnitude + versor components.
        self.assertEqual(out.BoundaryConditions.leading.IDs.tolist(), [2])
        self.assertEqual(out.BoundaryConditions.leading.m.tolist(), [2.5])
        self.assertEqual(out.BoundaryConditions.leading.mx.tolist(), [1.0])
        self.assertEqual(out.BoundaryConditions.leading.my.tolist(), [2.0])
        self.assertEqual(out.BoundaryConditions.leading.mz.tolist(), [3.0])

        # Floating: side 3, NaNs.
        self.assertEqual(out.BoundaryConditions.floating.IDs.tolist(), [3])
        for arr in (out.BoundaryConditions.floating.m,
                    out.BoundaryConditions.floating.mx,
                    out.BoundaryConditions.floating.my,
                    out.BoundaryConditions.floating.mz):
            self.assertTrue(np.all(np.isnan(arr)))

        # Pass-through fields.
        self.assertEqual(out.GeomFileName, "some_geometry")
        self.assertEqual(out.VolumetricSource.f, 1.0)

    def test_elastic_bc_identical_logic(self):
        geom = make_geom_data()
        outf = convert_geom_data_2_fluid_data(geom)
        oute = convert_geom_data_2_elastic_data(geom)

        self.assertEqual(oute.BoundaryConditions.dirichlet.IDs.tolist(), [1])
        self.assertEqual(oute.BoundaryConditions.leading.IDs.tolist(), [2])
        self.assertEqual(oute.BoundaryConditions.floating.IDs.tolist(), [3])
        # Same BC content as fluid (identical MATLAB logic).
        np.testing.assert_array_equal(
            oute.BoundaryConditions.leading.m, outf.BoundaryConditions.leading.m
        )

    def test_leading_bad_n3d_raises(self):
        geom = make_geom_data()
        geom.bndrsAttrbs[1] = make_side(
            False, mag=1.0, n3d=(1.0, 2.0)
        )
        with self.assertRaises(ValueError):
            convert_geom_data_2_fluid_data(geom)

    def test_all_dirichlet(self):
        geom = make_geom_data()
        geom.bndrsAttrbs = [make_side(True) for _ in range(4)]
        geom.bndrsHandles = [SideHandle(isPersistent=False) for _ in range(4)]
        out = convert_geom_data_2_fluid_data(geom)
        self.assertEqual(out.BoundaryConditions.dirichlet.IDs.tolist(), [1, 2, 3, 4])
        self.assertEqual(len(out.BoundaryConditions.leading.IDs), 0)
        self.assertEqual(len(out.BoundaryConditions.floating.IDs), 0)

# ---------------------------------------------------------------------------
# Boundary map
# ---------------------------------------------------------------------------
class TestBoundaryMap(unittest.TestCase):
    def test_mapping_and_divisor(self):
        parent = [
            types.SimpleNamespace(patches=1, faces=1),
            types.SimpleNamespace(patches=1, faces=3),
            types.SimpleNamespace(patches=2, faces=1),
        ]
        subdiv = [
            types.SimpleNamespace(patches=1, faces=1),
            types.SimpleNamespace(patches=1, faces=1),
            types.SimpleNamespace(patches=1, faces=3),
            types.SimpleNamespace(patches=2, faces=1),
            types.SimpleNamespace(patches=2, faces=1),
        ]
        idxs, dvsr = map_boundaries_h_to_h_sbdv(parent, subdiv)
        np.testing.assert_array_equal(idxs, np.array([1, 1, 2, 3, 3]))
        np.testing.assert_array_equal(dvsr, np.array([2, 2, 1, 2, 2]))

    def test_empty_input(self):
        idxs, dvsr = map_boundaries_h_to_h_sbdv([], [])
        self.assertEqual(idxs.shape, (0,))
        self.assertEqual(dvsr.shape, (0,))
        self.assertEqual(idxs.dtype, np.int64)

# ---------------------------------------------------------------------------
# Stokes
# ---------------------------------------------------------------------------
class TestStokes(unittest.TestCase):
    def test_all_dirichlet_solution(self):
        backend = MockGeoPDEsBackend()
        r = mp_solve_stokes_complete(
            make_problem_data_stokes(nmnn_sides=()),
            make_method_data_stokes(),
            backend,
        )
        # 8-tuple: geometry, msh, space_v, vel, space_p, press, elpsTime, memFtprt
        self.assertEqual(len(r), 8)
        vel, press = np.asarray(r[3], dtype=float), np.asarray(r[5], dtype=float)
        self.assertEqual(vel.shape, (10, 1))
        self.assertEqual(press.shape, (5, 1))

        # Mock: A=I, B=0, F=1, E=ones(1,np).
        #   Dirichlet dofs (1-based [1..5] -> 0-based [0..4]) set to vel_drchlt=1.
        #   Int system: A(int,int)-N(int,int) = I - I = 0 (singular; B=0, E=ones).
        #   lstsq min-norm with constraint sum(press)=0 -> press=0, int vel free->0.
        np.testing.assert_allclose(vel[:5, 0], np.ones(5), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(vel[5:, 0], np.zeros(5), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(press[:, 0], np.zeros(5), rtol=1e-8, atol=1e-8)
        self.assertGreaterEqual(float(r[6]), 0.0)  # elpsTime

    def test_neumann_solution_finite(self):
        backend = MockGeoPDEsBackend()
        r = mp_solve_stokes_complete(
            make_problem_data_stokes(nmnn_sides=[1]),
            make_method_data_stokes(),
            backend,
        )
        vel, press = np.asarray(r[3], dtype=float), np.asarray(r[5], dtype=float)
        self.assertEqual(vel.shape, (10, 1))
        self.assertEqual(press.shape, (5, 1))
        self.assertTrue(np.all(np.isfinite(vel)))
        self.assertTrue(np.all(np.isfinite(press)))

    def test_rt_element_rejected(self):
        md = make_method_data_stokes()
        md.element_name = "RT"
        with self.assertRaises(ValueError) as ctx:
            mp_solve_stokes_complete(
                make_problem_data_stokes(), md, MockGeoPDEsBackend()
            )
        self.assertIn("For RT and NDL spaces", str(ctx.exception))

    def test_ndl_element_rejected(self):
        md = make_method_data_stokes()
        md.element_name = "NDL"
        with self.assertRaises(ValueError) as ctx:
            mp_solve_stokes_complete(
                make_problem_data_stokes(), md, MockGeoPDEsBackend()
            )
        self.assertIn("use mp_solve_stokes_div_conforming", str(ctx.exception))

# ---------------------------------------------------------------------------
# Elasticity
# ---------------------------------------------------------------------------
class TestElasticity(unittest.TestCase):
    def test_all_dirichlet_solution(self):
        backend = MockGeoPDEsBackend()
        r = mp_solve_linear_elasticity_complete(
            make_problem_data_elas(nmnn_sides=()),
            make_method_data_elas(),
            backend,
        )
        # 6-tuple: geometry, msh, space, u, elpsTime, memFtprt
        self.assertEqual(len(r), 6)
        u = np.asarray(r[3], dtype=float)
        self.assertEqual(u.shape, (12, 1))

        # Mock: mat=I, rhs=1.  Dirichlet dofs [0..5]=1.
        #   rhs(int) = rhs - mat(int,drchlt)*u_drchlt = 1 - I*1 = 0? No:
        #   mat is 12x12 identity, mat(int,drchlt) is zero off-diagonal, so
        #   rhs(int) = 1 - 0 = 1;  u(int) = I \ 1 = 1.  -> all 1s (matches MATLAB).
        np.testing.assert_allclose(u[:, 0], np.ones(12), rtol=1e-8, atol=1e-8)
        self.assertGreaterEqual(float(r[4]), 0.0)  # elpsTime

    def test_neumann_solution_finite(self):
        backend = MockGeoPDEsBackend()
        r = mp_solve_linear_elasticity_complete(
            make_problem_data_elas(nmnn_sides=[1]),
            make_method_data_elas(),
            backend,
        )
        u = np.asarray(r[3], dtype=float)
        self.assertEqual(u.shape, (12, 1))
        self.assertTrue(np.all(np.isfinite(u)))

if __name__ == "__main__":
    unittest.main(verbosity=2)
