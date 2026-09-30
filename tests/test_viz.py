"""B3 - Headless smoke tests for the hexiga.viz rendering layer (B1).

These tests are LIGHT: they only assert structural invariants (non-empty
meshes, finite coordinates, actor counts > 0) and never call Plotter.show()
so they run headless and in a few seconds.

The ground-truth MATLAB equivalent is MAIN_gui/DemoSynthCAD.m, which drives
the same five plot_solid_domain / plot_ctrl_pts / plot_exploded_param /
plot_reflection_lines / plot_trabecular modes via its "Demo" dropdown.
"""

import numpy as np
import pyvista as pv
import unittest

from hexiga.nurbs import Nrb
from hexiga.viz.mesh import (
    SIDE_COLORS,
    all_side_meshes,
    control_points,
    curve_mesh,
    exterior_boundary_meshes,
    surface_mesh,
)
from hexiga.viz.plots import (
    get_explosion_transforms,
    plot_ctrl_pts,
    plot_exploded_param,
    plot_reflection_lines,
    plot_solid_domain,
    plot_trabecular,
)

def _cube(x0, y0, z0, x1, y1, z1):
    """Build a 3-D bicubic NURBS hexahedron with identity weight row.

    Knots are [0,0,1,1] (bilinear degree-1 per axis); the control-point
    grid maps the 8 corners exactly.  This is the canonical cube used
    throughout the test-suite and is the ground-truth shape that the
    multipatch interface-detection relies on (nrbmultipatch is verified
    against it in test_multipatch.py).
    """
    knots = [np.array([0.0, 0.0, 1.0, 1.0]) for _ in range(3)]
    coefs = np.zeros((4, 2, 2, 2))
    coefs[0, 0, :, :] = x0
    coefs[0, 1, :, :] = x1
    coefs[1, :, 0, :] = y0
    coefs[1, :, 1, :] = y1
    coefs[2, :, :, 0] = z0
    coefs[2, :, :, 1] = z1
    coefs[3] = 1.0
    return Nrb(number=(2, 2, 2), order=(2, 2, 2), knots=knots, coefs=coefs)

def _surface():
    """A 2-D bicubic NURBS surface for surface_mesh tests."""
    knots = [np.array([0.0, 0.0, 1.0, 1.0]) for _ in range(2)]
    coefs = np.zeros((4, 2, 2))
    coefs[0, 0, :] = 0.0
    coefs[0, 1, :] = 1.0
    coefs[1, :, 0] = 0.0
    coefs[1, :, 1] = 1.0
    coefs[2] = 0.5
    coefs[3] = 1.0
    return Nrb(number=(2, 2), order=(2, 2), knots=knots, coefs=coefs)

def _curve():
    """A 1-D cubic NURBS curve for curve_mesh tests."""
    knots = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    coefs = np.zeros((4, 4))
    coefs[0] = [0.0, 0.25, 0.75, 1.0]
    coefs[1] = [0.0, 0.5, 0.5, 0.0]
    coefs[2] = 0.0
    coefs[3] = 1.0
    return Nrb(number=(4,), order=(4,), knots=(knots,), coefs=coefs)

def _2x2_grid():
    return [
        _cube(0, 0, 0, 1, 1, 1),
        _cube(1, 0, 0, 2, 1, 1),
        _cube(0, 1, 0, 1, 2, 1),
        _cube(1, 1, 0, 2, 2, 1),
    ]

class TestMeshPrimitives(unittest.TestCase):
    """Structural invariants of the primitive mesh builders."""

    def test_side_colors_shape_and_range(self):
        self.assertEqual(len(SIDE_COLORS), 6)
        for c in SIDE_COLORS:
            self.assertEqual(len(c), 3)
            self.assertTrue(np.all((np.asarray(c) >= 0.0) & (np.asarray(c) <= 1.0)))

    def test_surface_mesh_quads_finite(self):
        surf = _surface()
        mesh = surface_mesh(surf, nsub=8)
        self.assertTrue(mesh.n_points >= 64)
        self.assertTrue(mesh.n_cells >= 49)
        self.assertTrue(np.all(np.isfinite(mesh.points)))

    def test_surface_mesh_rejects_3d(self):
        with self.assertRaises(ValueError):
            surface_mesh(_cube(0, 0, 0, 1, 1, 1), nsub=4)

    def test_surface_mesh_with_color_alpha(self):
        surf = _surface()
        mesh = surface_mesh(surf, nsub=4, color=(0.1, 0.2, 0.3), alpha=0.5)
        self.assertIn("rgba", mesh.point_data.keys())

    def test_curve_mesh_line_segments(self):
        cv = _curve()
        mesh = curve_mesh(cv, nsub=20)
        self.assertEqual(mesh.n_points, 20)
        self.assertEqual(mesh.n_cells, 1)
        self.assertTrue(np.all(np.isfinite(mesh.points)))

    def test_control_points_count(self):
        pts = control_points(_cube(0, 0, 0, 1, 1, 1))
        self.assertEqual(pts.n_points, 8)
        self.assertTrue(np.all(np.isfinite(pts.points)))

    def test_all_side_meshes_are_coloured(self):
        faces = all_side_meshes(_cube(0, 0, 0, 1, 1, 1), nsub=4)
        for f in faces:
            self.assertIn("rgba", f.point_data.keys())

    def test_all_side_meshes_count(self):
        faces = all_side_meshes(_cube(0, 0, 0, 1, 1, 1), nsub=4)
        self.assertEqual(len(faces), 6)
        for f in faces:
            self.assertTrue(f.n_points > 0)

    def test_exterior_boundary_single_cube(self):
        faces = exterior_boundary_meshes([_cube(0, 0, 0, 1, 1, 1)], nsub=4)
        self.assertEqual(len(faces), 6)

    def test_exterior_boundary_2x2_grid(self):
        faces = exterior_boundary_meshes(_2x2_grid(), nsub=4)
        # 4 cubes * 6 faces = 24; 4 interior shared faces = 24 - 8 = 16
        self.assertEqual(len(faces), 16)

    def test_exterior_boundary_2x2x2_grid(self):
        cubes = []
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    cubes.append(_cube(i, j, k, i + 1, j + 1, k + 1))
        faces = exterior_boundary_meshes(cubes, nsub=4)
        # 8 cubes * 6 faces = 48; 12 interior shared faces = 48 - 24 = 24
        self.assertEqual(len(faces), 24)

class TestPlotFunctionsHeadless(unittest.TestCase):
    """Smoke test each plot_* function headless (no .show()).

    We build the plotter, assert it has actors, and close it.
    """

    def test_plot_solid_domain_ghostflag_mix(self):
        p = plot_solid_domain(_2x2_grid(), ghostflag=[0, 1, 0, 1], nsub=4)
        try:
            self.assertGreater(len(p.actors), 0)
        finally:
            p.close()

    def test_plot_solid_domain_default(self):
        p = plot_solid_domain(_2x2_grid(), nsub=4)
        try:
            self.assertGreater(len(p.actors), 0)
        finally:
            p.close()

    def test_plot_ctrl_pts(self):
        p = plot_ctrl_pts(_2x2_grid(), nsub=4)
        try:
            self.assertGreater(len(p.actors), 0)
        finally:
            p.close()

    def test_plot_exploded_param(self):
        p = plot_exploded_param(_2x2_grid(), nsub=4)
        try:
            self.assertGreater(len(p.actors), 0)
        finally:
            p.close()

    def test_plot_reflection_lines(self):
        p = plot_reflection_lines(_2x2_grid(), nsub=4)
        try:
            self.assertGreater(len(p.actors), 0)
        finally:
            p.close()

    def test_plot_trabecular_default(self):
        p = plot_trabecular(_2x2_grid(), nsub=4)
        try:
            self.assertGreater(len(p.actors), 0)
        finally:
            p.close()

class TestExplosionTransforms(unittest.TestCase):
    def test_shape_per_patch(self):
        mp = _2x2_grid()
        M1s, M2s = get_explosion_transforms(mp)
        self.assertEqual(M1s.shape, (4, 4, len(mp)))
        self.assertEqual(M2s.shape, (4, 4, len(mp)))
        self.assertTrue(np.all(np.isfinite(M1s)))
        self.assertTrue(np.all(np.isfinite(M2s)))

    def test_rotation_is_proper(self):
        mp = [_cube(0, 0, 0, 1, 1, 1), _cube(1, 0, 0, 2, 1, 1)]
        M1s, _ = get_explosion_transforms(mp)
        for k in range(len(mp)):
            R = M1s[:, :, k][:3, :3]
            # Proper rotation: det = +1.
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=8)

    def test_identity_patch_is_translation_only(self):
        mp = [_cube(0, 0, 0, 1, 1, 1)]
        M1s, _ = get_explosion_transforms(mp)
        R = M1s[:, :, 0][:3, :3]
        self.assertTrue(np.allclose(R, np.eye(3), atol=1e-10))

if __name__ == "__main__":
    unittest.main(verbosity=2)
