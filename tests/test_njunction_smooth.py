"""Smoke tests for the smooth / turbo-smooth path in the njunction GUI."""
import unittest

import numpy as np

from hexiga.gui.app_njunction import (
    NJuncDATA,
    run_generate,
    run_generate_raw_scaff,
    run_smooth_raw_scaff,
    run_turbo_smooth_raw_scaff,
)

def _build(nj: NJuncDATA) -> bool:
    QFSdata, ok1 = run_generate(nj)
    if not ok1 or QFSdata is None:
        return False
    nj.QFSdata = QFSdata
    mpHexa, ioLetsHexa, ok2 = run_generate_raw_scaff(nj)
    if not ok2 or mpHexa is None:
        return False
    nj.mpHexa = mpHexa
    nj.ioLetsHexa = ioLetsHexa
    return True

def _bad(mpHexa) -> int:
    """Count hexa with non-finite control points."""
    bad = 0
    for h in mpHexa:
        cp = getattr(h, "coefs", None)
        if cp is None:
            continue
        arr = np.asarray(cp, dtype=float)
        if not np.all(np.isfinite(arr)):
            bad += 1
    return bad

class TestNJunctionSmooth(unittest.TestCase):
    def test_smooth_N3(self):
        nj = NJuncDATA(numTubes=3, mixIOFLAG=False, extrdLen=1.0,
                       bevelDeg=0.0, randomLen=True, WallFLAG=False,
                       regDirTubesFLAG=True)
        np.random.seed(1)
        self.assertTrue(_build(nj))
        mpS, ok = run_smooth_raw_scaff(nj)
        self.assertTrue(ok)
        self.assertIsNotNone(mpS)
        self.assertGreater(len(mpS), 0)
        self.assertEqual(_bad(mpS), 0)

    def test_turbo_smooth_N3(self):
        nj = NJuncDATA(numTubes=3, mixIOFLAG=False, extrdLen=1.0,
                       bevelDeg=0.0, randomLen=True, WallFLAG=False,
                       regDirTubesFLAG=True)
        np.random.seed(1)
        self.assertTrue(_build(nj))
        mpT, ok = run_turbo_smooth_raw_scaff(nj)
        self.assertTrue(ok)
        self.assertIsNotNone(mpT)
        self.assertGreater(len(mpT), 0)
        self.assertEqual(_bad(mpT), 0)

    def test_smooth_N4(self):
        nj = NJuncDATA(numTubes=4, mixIOFLAG=False, extrdLen=1.0,
                       bevelDeg=0.0, randomLen=True, WallFLAG=False,
                       regDirTubesFLAG=True)
        np.random.seed(2)
        self.assertTrue(_build(nj))
        mpS, ok = run_smooth_raw_scaff(nj)
        self.assertTrue(ok)
        self.assertIsNotNone(mpS)
        self.assertGreater(len(mpS), 0)
        self.assertEqual(_bad(mpS), 0)

if __name__ == "__main__":
    unittest.main()
