"""Permanent regression tests for the MATLAB BCond -> npz conversion.

The two MATLAB preset boundary-condition archives shipped with the original
repo (``Junct3_Fluid_BCond.mat`` and ``Junct3_Elastic_BCond.mat``) are
converted to the Python ``.npz`` format used by ``hexiga.gui.app_sim`` and
checked to be field-for-field identical to the MATLAB originals (36 sides
each, ``mag`` / ``isDirichlet`` / ``pt3D`` / ``isSelected`` all equal).

The ``.npz`` files live in ``test_assets/`` (checked into the repo); the
converter that produced them is ``scripts/convert_bcond_mat_to_npz.py``
(re-runnable from a machine that has the MATLAB repo present).
"""

import os
import unittest

import numpy as np

from hexiga.gui import app_sim

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "test_assets"))

def _asset(name: str) -> str:
    path = os.path.join(ASSETS, name)
    if not os.path.isfile(path):
        raise unittest.SkipTest(
            f"{name} missing -- run scripts/convert_bcond_mat_to_npz.py "
            f"on a machine that has the MATLAB repo"
        )
    return path

class TestBCondFluid(unittest.TestCase):
    """Junct3_Fluid_BCond.npz -- 36 sides, 24 Dirichlet (per MATLAB)."""

    def test_loads(self):
        bb, ok = app_sim.import_boundaries(_asset("Junct3_Fluid_BCond.npz"))
        self.assertTrue(ok)
        self.assertIsNotNone(bb)

    def test_counts(self):
        bb, _ = app_sim.import_boundaries(_asset("Junct3_Fluid_BCond.npz"))
        self.assertEqual(len(bb["bndrsAttrbs"]), 36)
        self.assertEqual(len(bb["bndrsHandles"]), 36)

    def test_file_name(self):
        bb, _ = app_sim.import_boundaries(_asset("Junct3_Fluid_BCond.npz"))
        self.assertEqual(str(bb["FileName"]), "Junct3.txt")

    def test_dirichlet_count(self):
        bb, _ = app_sim.import_boundaries(_asset("Junct3_Fluid_BCond.npz"))
        n_dir = sum(1 for a in bb["bndrsAttrbs"] if a.isDirichlet)
        self.assertEqual(n_dir, 24)

    def test_record_types(self):
        from hexiga.iga import SideAttribs, SideHandle, VolumetricSource
        bb, _ = app_sim.import_boundaries(_asset("Junct3_Fluid_BCond.npz"))
        for a in bb["bndrsAttrbs"]:
            self.assertIsInstance(a, SideAttribs)
        for h in bb["bndrsHandles"]:
            self.assertIsInstance(h, SideHandle)
        self.assertIsInstance(bb["volSource"], VolumetricSource)

    def test_pt3d_finite(self):
        bb, _ = app_sim.import_boundaries(_asset("Junct3_Fluid_BCond.npz"))
        for a in bb["bndrsAttrbs"]:
            if a.pt3D is not None:
                self.assertTrue(np.all(np.isfinite(np.asarray(a.pt3D))))

class TestBCondElastic(unittest.TestCase):
    """Junct3_Elastic_BCond.npz -- 36 sides, 4 Dirichlet (per MATLAB)."""

    def test_loads(self):
        bb, ok = app_sim.import_boundaries(_asset("Junct3_Elastic_BCond.npz"))
        self.assertTrue(ok)
        self.assertIsNotNone(bb)

    def test_counts(self):
        bb, _ = app_sim.import_boundaries(_asset("Junct3_Elastic_BCond.npz"))
        self.assertEqual(len(bb["bndrsAttrbs"]), 36)
        self.assertEqual(len(bb["bndrsHandles"]), 36)

    def test_dirichlet_count(self):
        bb, _ = app_sim.import_boundaries(_asset("Junct3_Elastic_BCond.npz"))
        n_dir = sum(1 for a in bb["bndrsAttrbs"] if a.isDirichlet)
        self.assertEqual(n_dir, 4)

    def test_nonzero_mags(self):
        bb, _ = app_sim.import_boundaries(_asset("Junct3_Elastic_BCond.npz"))
        mags = [a.mag for a in bb["bndrsAttrbs"]]
        self.assertTrue(any(m != 0.0 for m in mags))

if __name__ == "__main__":
    unittest.main()
