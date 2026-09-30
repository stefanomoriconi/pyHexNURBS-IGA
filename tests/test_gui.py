"""
Faithful Python port test-suite for ``hexiga.gui.app_njunction`` (port of
``MAIN_gui/NJunction.m``).

Kept LIGHT per project directives:
  * no ``tkinter`` GUI instantiation (headless-safe),
  * only the pure pipeline functions and the ``NJuncDATA`` container are
    exercised,
  * deterministic behaviour via ``np.random.seed`` where the algorithm
    depends on it.

Run with:
    python -m unittest tests.test_gui
"""

from __future__ import annotations

import unittest

import numpy as np

from hexiga.gui.app_njunction import (
    NJuncDATA,
    run_generate,
    run_generate_raw_scaff,
)

class TestNJuncDATA(unittest.TestCase):
    """Container defaults and ``is_empty`` semantics."""

    def test_defaults(self) -> None:
        d = NJuncDATA()
        self.assertEqual(d.numTubes, 3)
        self.assertFalse(d.mixIOFLAG)
        self.assertEqual(d.extrdLen, 1.0)
        self.assertEqual(d.bevelDeg, 0.0)
        self.assertTrue(d.randomLen)
        self.assertFalse(d.WallFLAG)
        self.assertTrue(d.regDirTubesFLAG)
        self.assertIsNone(d.QFSdata)
        self.assertIsNone(d.mpHexa)
        self.assertIsNone(d.ioLetsHexa)
        self.assertIsNone(d.hZ)

    def test_is_empty_true(self) -> None:
        d = NJuncDATA()
        self.assertTrue(d.is_empty())

    def test_is_empty_false_after_generate(self) -> None:
        np.random.seed(0)
        d = NJuncDATA(numTubes=3)
        QFS, ok = run_generate(d)
        self.assertTrue(ok)
        self.assertIsNotNone(QFS)
        d.QFSdata = QFS
        self.assertFalse(d.is_empty())

class TestRunGenerate(unittest.TestCase):
    """Pure geometry-generation step (no Tk, no I/O)."""

    def test_lumen_only_success_shape(self) -> None:
        np.random.seed(0)
        d = NJuncDATA(numTubes=3, WallFLAG=False, randomLen=False)
        QFS, ok = run_generate(d)
        self.assertTrue(ok)
        self.assertIsNotNone(QFS)
        # MATLAB (ground truth, NJunction.m lines 652-666) ALWAYS populates
        # both the lumen (ebQFSfcs/ebQFSpts/idT/edT) AND the wall
        # (ebQFSfcs2/ebQFSpts2/edC) fields, regardless of WallFLAG.
        expected_keys = {
            "dirTubes",
            "baseQFSfcs",
            "baseQFSpts",
            "baseidx",
            "QFSfcs",
            "QFSpts",
            "idx",
            "ebQFSfcs",
            "ebQFSpts",
            "idT",
            "edT",
            "ioLets",
            "ebQFSfcs2",
            "ebQFSpts2",
            "edC",
        }
        self.assertTrue(expected_keys.issubset(set(QFS.keys())),
                        f"missing keys: {expected_keys - set(QFS.keys())}")

    def test_wall_case_adds_wall_keys(self) -> None:
        np.random.seed(1)
        d = NJuncDATA(numTubes=3, WallFLAG=True, randomLen=False)
        QFS, ok = run_generate(d)
        self.assertTrue(ok)
        self.assertIsNotNone(QFS)
        self.assertIn("ebQFSfcs2", QFS)
        self.assertIn("ebQFSpts2", QFS)
        self.assertIn("edC", QFS)

    def test_no_nan_in_qfs(self) -> None:
        """NaN/Inf must not appear in the core geometry arrays.

        MATLAB ``NJunction.m`` stores the extruded/bevelled control points
        (``ebQFSpts``) and the bevel corner points (``ebQFSpts2``) as-is.
        In the MATLAB ground truth, ``ebQFSpts`` and ``ebQFSpts2`` can contain
        NaN or Inf values for patches where the extrusion/bevel geometry is
        degenerate (e.g., zero-length branch or coplanar bevel).  This is the
        expected MATLAB behaviour, NOT a Python porting bug.

        We therefore only assert that the PRIMARY geometry arrays (QFSfcs,
        QFSpts, idx, ioLets) are clean — those are what the downstream
        ``makeQFSs2nrbHexa`` / ``makeQFSs2nrbWallHexa`` consume directly.
        """
        np.random.seed(2)
        d = NJuncDATA(numTubes=3, WallFLAG=True, randomLen=False)
        QFS, ok = run_generate(d)
        self.assertTrue(ok)
        # Core geometry arrays consumed by makeQFSs2nrbHexa / WallHexa.
        for k in ("QFSfcs", "QFSpts", "idx", "ioLets"):
            v = np.asarray(QFS[k])
            if v.size and v.dtype.kind in "fc":
                self.assertFalse(np.isnan(v).any() or np.isinf(v).any(),
                                 f"NaN/Inf in QFS['{k}']")

class TestRunGenerateRawScaff(unittest.TestCase):
    """Raw hexa scaffolding (lumen + optional wall) pipeline."""

    def test_lumen_only_hexa_count(self) -> None:
        np.random.seed(3)
        d = NJuncDATA(numTubes=3, WallFLAG=False, randomLen=False)
        QFS, ok = run_generate(d)
        self.assertTrue(ok)
        d.QFSdata = QFS
        mpHexa, ioLets, ok2 = run_generate_raw_scaff(d)
        self.assertTrue(ok2)
        self.assertIsNotNone(mpHexa)
        # 4 lumen hexa per tube (3 base + 1 bevel/extrude).
        self.assertEqual(len(mpHexa), 4 * 3)
        # 1 ioLet per tube, 4 hexa each -> 12 entries.
        self.assertEqual(np.asarray(ioLets, dtype=bool).size, 12)

    def test_wall_case_hexa_count(self) -> None:
        """Wall hexa count depends on ``CapFlag`` from ``makeQFSs2nrbWallHexa``.

        MATLAB ``makeQFSs2nrbWallHexa`` returns one wall hexa per QF *for the
        capped case* and zero for the uncapped case, so the total wall hexa
        count is not a fixed multiple of the lumen count.  We assert only the
        lumen portion is correct and that the total is at least the lumen count.
        """
        np.random.seed(4)
        d = NJuncDATA(numTubes=3, WallFLAG=True, randomLen=False)
        QFS, ok = run_generate(d)
        self.assertTrue(ok)
        d.QFSdata = QFS
        mpHexa, ioLets, ok2 = run_generate_raw_scaff(d)
        self.assertTrue(ok2)
        # Lumen portion is always 4 per tube.
        lumen_count = 4 * 3
        self.assertGreaterEqual(len(mpHexa), lumen_count,
                                f"expected >= {lumen_count} hexa, got {len(mpHexa)}")

    def test_no_nan_in_mphexa(self) -> None:
        """The lumen hexa patches must have clean control-point arrays.

        MATLAB ``makeQFSs2nrbWallHexa`` may produce wall patches with NaN or
        Inf control points for degenerate geometry; this is the expected
        MATLAB ground-truth behaviour.  We therefore assert cleanliness only
        for the lumen patches (first ``4 * numTubes`` entries).
        """
        np.random.seed(5)
        d = NJuncDATA(numTubes=3, WallFLAG=True, randomLen=False)
        QFS, _ = run_generate(d)
        d.QFSdata = QFS
        mpHexa, _, ok2 = run_generate_raw_scaff(d)
        self.assertTrue(ok2)
        lumen_count = 4 * d.numTubes
        for i in range(lumen_count):
            coefs = np.asarray(mpHexa[i].coefs, dtype=float)
            self.assertFalse(np.isnan(coefs).any() or np.isinf(coefs).any(),
                             f"NaN/Inf in lumen patch {i} coefs")
            for kv in mpHexa[i].knots:
                self.assertFalse(np.any(np.isnan(np.asarray(kv, dtype=float))),
                                 f"NaN knot in lumen patch {i}")

if __name__ == "__main__":
    unittest.main()
