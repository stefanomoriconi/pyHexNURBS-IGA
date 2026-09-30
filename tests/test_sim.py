"""
Light test-suite for ``hexiga.gui.app_sim`` (faithful port of
``MAIN_gui/HexIGASim.m``).

Kept LIGHT per project directives:
  * no ``tkinter`` GUI event loop (headless-safe) -- only the headless core
    pipeline functions and the ``GeomDATA`` container are exercised,
  * the geometry-dependent tests are skipped when the reference geometry
    file (``test_geometry/Junct3.txt`` in the MATLAB ground-truth tree) is
    not present,
  * no heavy solver run -- the BC-validity gate (which fires *before* any
    solver call) is what is asserted for the "invalid BC" paths.

Run with:
    python -m unittest tests.test_sim
"""

from __future__ import annotations

import os
import unittest
import warnings

import numpy as np

from hexiga.gui.app_sim import (
    GeomDATA,
    chk_valid_bc_vol_source,
    get_uv_w_dirichlet,
    get_void_attribs,
    get_void_boundary_handles,
    get_void_geom_data,
    get_void_vol_source,
    gen_arrow_boundary_handles,
    gen_boundaries_attributes,
    import_geometry,
    run_sim,
)

# Reference geometry shipped with the MATLAB ground-truth tree (dev-machine
# only; not distributed with this repo). Skipped when absent.
_TEST_GEOM = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "HexIGA_Master_Orig",
    "test_geometry", "Junct3.txt"))

def _geom_path() -> str:
    if os.path.isfile(_TEST_GEOM):
        return _TEST_GEOM
    raise unittest.SkipTest("reference geometry (Junct3.txt) not found")

# ---------------------------------------------------------------------------
# container + void helpers
# ---------------------------------------------------------------------------
class TestGeomData(unittest.TestCase):
    def test_void_defaults(self) -> None:
        g = get_void_geom_data()
        self.assertIsInstance(g, GeomDATA)
        self.assertEqual(g.FileName, "")
        self.assertEqual(g.FilePath, "")
        self.assertIsNone(g.mpHexa)
        self.assertIsNone(g.bndrs)
        self.assertEqual(len(g.bndrsAttrbs), 0)
        self.assertEqual(len(g.bndrsHandles), 0)
        self.assertIsNone(g.volSource)
        self.assertFalse(g.is_loaded())

    def test_void_helpers(self) -> None:
        a = get_void_attribs()
        self.assertEqual(a.mag, 0.0)
        self.assertFalse(a.isDirichlet)
        h = get_void_boundary_handles()
        self.assertFalse(h.isSelected)
        self.assertFalse(h.isPersistent)
        v = get_void_vol_source()
        self.assertEqual(v.f, 0.0)

class TestUVWDirichlet(unittest.TestCase):
    def test_duplicates_each_axis(self) -> None:
        # MATLAB (ground truth) duplicates each axis into a 6-vector:
        #   [U U V V W W]
        self.assertEqual(get_uv_w_dirichlet(True, False, True),
                         [True, True, False, False, True, True])
        self.assertEqual(get_uv_w_dirichlet(False, False, False),
                         [False] * 6)

# ---------------------------------------------------------------------------
# BC / volume-source validity
# ---------------------------------------------------------------------------
def _bc(dirichlet_ids, leading_ids, leading_m):
    from hexiga.iga import BoundaryConditions, SideAttribsRow
    drch = SideAttribsRow(IDs=np.asarray(dirichlet_ids, dtype=int),
                          m=np.zeros(max(len(dirichlet_ids), 1),
                                     dtype=float))
    lead = SideAttribsRow(IDs=np.asarray(leading_ids, dtype=int),
                          m=np.asarray(leading_m, dtype=float))
    return BoundaryConditions(dirichlet=drch, leading=lead)

def _sim(BC, f=0.0, fx=0.0, fy=0.0, fz=0.0):
    import types
    d = types.SimpleNamespace()
    d.BoundaryConditions = BC
    d.VolumetricSource = types.SimpleNamespace(
        f=f, fx=fx, fy=fy, fz=fz)
    return d

class TestChkValidBCVolSource(unittest.TestCase):
    def test_leading_plus_dirichlet_valid(self) -> None:
        bc = _bc([1], [2], [1.0])
        self.assertTrue(chk_valid_bc_vol_source(_sim(bc)))

    def test_volsource_plus_dirichlet_valid(self) -> None:
        bc = _bc([1], [], [])
        self.assertTrue(chk_valid_bc_vol_source(_sim(bc, f=1.0, fx=1.0)))

    def test_dirichlet_only_invalid(self) -> None:
        bc = _bc([1], [], [])
        self.assertFalse(chk_valid_bc_vol_source(_sim(bc)))

    def test_no_dirichlet_invalid(self) -> None:
        bc = _bc([], [2], [1.0])
        self.assertFalse(chk_valid_bc_vol_source(_sim(bc)))

# ---------------------------------------------------------------------------
# geometry import + boundary attributes (skip if no reference file)
# ---------------------------------------------------------------------------
class TestImportGeometry(unittest.TestCase):
    def test_import_and_attributes(self) -> None:
        path = _geom_path()
        geom, ok = import_geometry(path)
        self.assertTrue(ok)
        self.assertTrue(geom.is_loaded())
        self.assertGreater(len(geom.mpHexa), 0)
        self.assertGreater(len(geom.bndrsAttrbs), 0)
        self.assertEqual(len(geom.bndrsHandles), len(geom.bndrsAttrbs))
        # Every side attribute must carry finite, non-emptied vectors.
        for a in geom.bndrsAttrbs:
            self.assertTrue(np.all(np.isfinite(a.pt3D)))
            self.assertTrue(np.all(np.isfinite(a.n3D)))
            # unit normal (MATLAB ``uvect``)
            self.assertAlmostEqual(float(np.linalg.norm(a.n3D)), 1.0,
                                   places=6)

    def test_dirichlet_mask_applied(self) -> None:
        path = _geom_path()
        geom, ok = import_geometry(path, u_dirichlet=False,
                                   v_dirichlet=True, w_dirichlet=False)
        self.assertTrue(ok)
        # Default UVW is [F F T F F F]; count the sides flagged Dirichlet.
        n_dir = sum(1 for a in geom.bndrsAttrbs if a.isDirichlet)
        self.assertGreater(n_dir, 0)

class TestRunSimInvalidBC(unittest.TestCase):
    """The BC-validity gate fires before any solver run (light)."""

    def test_fluid_missing_bc_rejected(self) -> None:
        path = _geom_path()
        geom, ok = import_geometry(path)
        self.assertTrue(ok)
        # Force an invalid BC state: no leading magnitude, no vol source.
        from hexiga.iga import SideAttribsRow
        geom.bndrsAttrbs = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, ok2 = run_sim(geom, kind="fluid")
        self.assertFalse(ok2)

@unittest.skipUnless(
    __import__("hexiga.gui.app_sim", fromlist=["_TK_AVAILABLE"])._TK_AVAILABLE,
    "tkinter is not available (headless build smoke test skipped)")
class TestGuiBuild(unittest.TestCase):
    """Guard against the GUI class failing to import/build at all.

    This is intentionally the *only* Tk-touching test: it constructs the
    ``HexIGASim`` widget tree (``build()``) and immediately tears it down,
    never starting the event loop.  It catches regressions such as a broken
    ``tkinter`` import that silently disables the whole GUI.
    """

    def test_build_headless(self) -> None:
        from hexiga.gui.app_sim import HexIGASim
        app = HexIGASim()
        try:
            app.build()
            self.assertTrue(app.root.winfo_exists())
        finally:
            app.root.destroy()

if __name__ == "__main__":
    unittest.main()
