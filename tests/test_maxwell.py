#!/usr/bin/env python3
"""
tests/test_maxwell.py -- M3: stdlib ``unittest`` suite for the Maxwell
eigenvalue port in ``hexiga.iga``.

Covers the literal MATLAB->Python port of:

  * converter : convert_geom_data_2_maxwell_data  <- convertGeomDATA2MaxwellDATA.m
  * solver    : mp_solve_maxwell_eig_complete     <- mp_solve_maxwell_eig_complete.m

Oracle conventions:
  * MATLAB is ground truth (the ``iga_utils/*.m`` files).
  * All DOF indices and side IDs are 1-based in the MATLAB / mock world; the
    solver normalises to 0-based internally.
  * The external GeoPDEs backend is isolated behind :class:`MockGeoPDEsBackend`.

Mock Maxwell oracle (``MockGeoPDEsBackend`` with default ``ndof_e = 12``):

  * ``stiffness_maxwell`` -> diag(1, 2, ..., 12)
  * ``mass_maxwell``      -> identity (12 x 12)
  * ``maxwell_dirichlet_dofs`` -> 1-based [1..6]  (-> 0-based [0..5])
  * interior dofs -> 0-based [6..11]

  Hence the interior sub-matrix is ``A_sub = diag(7,8,9,10,11,12)``,
  ``M_sub = I``, and the (hand-computable) generalized eigenvalues are exactly
  ``[7, 8, 9, 10, 11, 12]`` with standard-basis eigenvectors.
"""

from __future__ import annotations
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexiga.iga import (  # noqa: E402
    MaxwellData,
    MockGeoPDEsBackend,
    convert_geom_data_2_maxwell_data,
    mp_solve_maxwell_eig_complete,
)
from hexiga.iga.dataclasses import SideAttribs, SideHandle, VolumetricSource

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

def make_geom_data():
    """A minimal GeomDATA mixing dirichlet / leading / floating sides."""
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
    return types.SimpleNamespace(
        mpHexa="maxwell_geometry",
        bndrsAttrbs=bndrs_attrbs,
        bndrsHandles=bndrs_handles,
        volSource=vol_source,
    )

def make_method_data_maxwell() -> types.SimpleNamespace:
    # NOTE: no ``element_name`` -- the Maxwell solver does not consume one
    # (MATLAB: element_name is commented out in the wrapper).
    return types.SimpleNamespace(
        degree=[2, 2, 2],
        regularity=[1, 1, 1],
        nsub=[1, 1, 1],
        nquad=[3, 3, 3],
    )

def make_problem_data_maxwell(nmnn_sides=(), forceAll=False) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        geo_name="maxwell_geometry",
        drchlt_sides=[1, 2, 3],
        nmnn_sides=list(nmnn_sides),
        c_elec_perm=1.0,
        c_magn_perm=1.0,
        forceAll=forceAll,
    )

# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------
class TestMaxwellConverter(unittest.TestCase):
    def test_bc_split_and_passthrough(self):
        geom = make_geom_data()
        out = convert_geom_data_2_maxwell_data(geom)

        self.assertIsInstance(out, MaxwellData)

        # Dirichlet: side 1 only, zero magnitude/normal.
        self.assertEqual(out.BoundaryConditions.dirichlet.IDs.tolist(), [1])
        self.assertEqual(out.BoundaryConditions.dirichlet.m.tolist(), [0.0])

        # Leading: side 2, magnitude + versor.
        self.assertEqual(out.BoundaryConditions.leading.IDs.tolist(), [2])
        self.assertEqual(out.BoundaryConditions.leading.m.tolist(), [2.5])
        np.testing.assert_allclose(
            np.array([out.BoundaryConditions.leading.mx.tolist()[0],
                      out.BoundaryConditions.leading.my.tolist()[0],
                      out.BoundaryConditions.leading.mz.tolist()[0]]),
            np.array([1.0, 2.0, 3.0]),
        )

        # Floating: side 3, NaNs.
        self.assertEqual(out.BoundaryConditions.floating.IDs.tolist(), [3])
        for arr in (out.BoundaryConditions.floating.m,
                    out.BoundaryConditions.floating.mx,
                    out.BoundaryConditions.floating.my,
                    out.BoundaryConditions.floating.mz):
            self.assertTrue(np.all(np.isnan(arr)))

        # Pass-through fields.
        self.assertEqual(out.GeomFileName, "maxwell_geometry")
        self.assertEqual(out.VolumetricSource.f, 1.0)
        # Result fields default to unset.
        self.assertIsNone(out.geometry)
        self.assertIsNone(out.space)
        self.assertIsNone(out.eigv)
        self.assertIsNone(out.eigf)

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class TestMaxwellSolver(unittest.TestCase):
    def test_forceAll_eigenvalues(self):
        backend = MockGeoPDEsBackend()
        r = mp_solve_maxwell_eig_complete(
            make_problem_data_maxwell(forceAll=True),
            make_method_data_maxwell(),
            backend,
        )
        # 7-tuple: geometry, msh, space, eigv, eigf, elpsTime, memFtprt
        self.assertEqual(len(r), 7)
        geometry, msh, space, eigv, eigf = r[0], r[1], r[2], r[3], r[4]
        eigv = np.asarray(eigv, dtype=float).ravel()
        eigf = np.asarray(eigf, dtype=float)

        # forceAll branch: all interior eigenvalues -> [7, 8, 9, 10, 11, 12].
        np.testing.assert_allclose(eigv, np.array([7., 8., 9., 10., 11., 12.]),
                                  rtol=1e-8, atol=1e-8)
        # Sorted ascending (MATLAB: [eigv, perm] = sort (eigv)).
        self.assertTrue(np.all(np.diff(eigv) >= -1e-12))
        # eigf: (ndof=12, k=6); Dirichlet rows [0..5] zero, interior block is
        # a set of standard-basis columns (unit 2-norm, exactly one 1 each).
        self.assertEqual(eigf.shape, (12, 6))
        np.testing.assert_allclose(eigf[:6, :], np.zeros((6, 6)), atol=1e-10)
        int_block = eigf[6:, :]
        self.assertTrue(np.all(np.allclose(np.abs(int_block).sum(axis=0), 1.0,
                                           atol=1e-8)))
        self.assertGreaterEqual(float(r[5]), 0.0)  # elpsTime

    def test_eigs_limited_branch(self):
        backend = MockGeoPDEsBackend()
        r = mp_solve_maxwell_eig_complete(
            make_problem_data_maxwell(nmnn_sides=[1], forceAll=False),
            make_method_data_maxwell(),
            backend,
        )
        self.assertEqual(len(r), 7)
        eigv = np.asarray(r[3], dtype=float).ravel()
        eigf = np.asarray(r[4], dtype=float)

        # maxNumEigs (100) is clamped to int_dofs.size - 2 = 6 - 2 = 4.
        # eigs(A', M', k=4, sigma=0.1) picks the 4 eigenvalues closest to 0.1
        # -> [7, 8, 9, 10], then sorted ascending.
        np.testing.assert_allclose(eigv, np.array([7., 8., 9., 10.]),
                                  rtol=1e-6, atol=1e-6)
        self.assertEqual(eigf.shape, (12, 4))
        # Dirichlet rows zero; interior block columns unit-norm.
        np.testing.assert_allclose(eigf[:6, :], np.zeros((6, 4)), atol=1e-8)
        self.assertTrue(np.allclose(np.linalg.norm(eigf[6:, :], axis=0), 1.0,
                                    atol=1e-6))
        self.assertGreaterEqual(float(r[5]), 0.0)

    def test_rdim_guard(self):
        # rdim==3 mock -> invmu uses c_magn_perm; solver must not crash.
        backend = MockGeoPDEsBackend(rdim=3)
        r = mp_solve_maxwell_eig_complete(
            make_problem_data_maxwell(forceAll=True),
            make_method_data_maxwell(),
            backend,
        )
        self.assertEqual(len(r), 7)
        self.assertTrue(np.all(np.isfinite(np.asarray(r[3], dtype=float))))

if __name__ == "__main__":
    unittest.main(verbosity=2)
