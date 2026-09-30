"""Regression / documentation tests for known issues in the HexIGA Python port.

These tests keep the suite green while pinning down the *current* behavior of a
reported defect (the mixed-I/O capped-tube smoothing face-association bug) so a
future fix is measured against a concrete baseline. The full report, the user's
verbatim description, and the workaround live in ``KNOWN_ISSUES.md``; the fix
plan lives in ``ROADMAP.md`` (P0).
"""
import unittest

import numpy as np

from hexiga.gui.app_njunction import (
    NJuncDATA,
    run_generate,
    run_generate_raw_scaff,
    run_smooth_raw_scaff,
)

# Verified to yield EXACTLY one capped (pointy-extrusion) tube among five,
# matching the user's reproduction: "num branches = 5 with only 1 directional
# tube being flagged as cap-closed, all the rest are open tubes".
_SEED = 8

def _build(nj: NJuncDATA) -> bool:
    """Generate the NJunction + raw scaffolding (mirrors ``runGenerate*``)."""
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

def _axial_spread(mpHexa) -> list:
    """Per-hexa axial (z) control-point spread — a coarse spike/flatness proxy.

    Uses ``hexa.cart()`` which returns a ``(3, ...)`` cartesian control grid;
    ``arr[2]`` is the z coordinate cloud.
    """
    out = []
    for h in mpHexa:
        try:
            pts = np.asarray(h.cart(), dtype=float)
            if pts.ndim == 0 or pts.size == 0 or not np.all(np.isfinite(pts)):
                out.append(np.nan)
                continue
            out.append(float(np.max(pts[2]) - np.min(pts[2])))
        except Exception:
            out.append(np.nan)
    return out

class TestMixedIOCappedTubeSmoothing(unittest.TestCase):
    """Known issue: capped-tube face-association under 'Mixed I/O' smoothing.

    The user report (verbatim, see KNOWN_ISSUES.md #1):

        > "there is still a bug in the face-association for smoothing in the
        >  case of 'Mixed I/O' enabled — the capped (closed) tube within does
        >  not [do] a flat extrusion, but a pointy extrusion of the tube does
        >  not seem to be smoothed correctly... This is reproducible for num
        >  branches = 5 and only 1 directional tube being flagged as cap-closed
        >  (pointy extrusion), all the rest are open tubes."

    Workaround (verbatim):

        > "if the 'include wall' is enabled, the face-swap or miss-assignment
        >  is resolved and the smoothing yields a correct Luminal + Wall
        >  compartment correctly smoothed for the whole scaffolding, including
        >  the capped tube."

    Observed in the Python port (seed 8, N=5, exactly one capped tube):
      * ``WallFLAG=False``: the capped lumen hexa ARE modified by smoothing but
        the cap does not resolve to a clean flat cap; smoothing still terminates
        and all control points stay finite.
      * ``WallFLAG=True``: the capped LUMEN hexa are left frozen (spread
        unchanged) while the capped WALL hexa are smoothed — the reported
        workaround holds.
    """

    def test_workaround_wall_includes_capped_tube(self):
        """Workaround path: with the wall included, smoothing stays clean.

        Invariants we DO pin down (all currently hold):
          * the run succeeds (``ok`` is True) and terminates,
          * every hexa keeps finite control points after smoothing,
          * the capped LUMEN hexa are preserved (their axial spread is
            unchanged), which is the user-reported correct behavior when the
            wall is included.
        """
        nj = NJuncDATA(numTubes=5, mixIOFLAG=True, extrdLen=1.0, bevelDeg=0.0,
                       randomLen=True, WallFLAG=True, regDirTubesFLAG=True)
        np.random.seed(_SEED)
        self.assertTrue(_build(nj))

        ioLetsHexa = np.asarray(nj.ioLetsHexa, dtype=bool)
        # Lumen hexa precede wall hexa (ioLetsLumen = np.repeat(ioLets, 4)).
        n_lumen = 4 * int(np.asarray(nj.QFSdata["ioLets"]).size)
        lumen_capped = [i for i, v in enumerate(ioLetsHexa) if not v and i < n_lumen]

        raw_spread = _axial_spread(nj.mpHexa)
        mpS, ok = run_smooth_raw_scaff(nj)
        self.assertTrue(ok)
        self.assertIsNotNone(mpS)
        self.assertGreater(len(mpS), 0)

        smooth_spread = _axial_spread(mpS)
        for i in lumen_capped:
            self.assertTrue(np.isfinite(raw_spread[i]), f"raw hexa[{i}] not finite")
            self.assertTrue(np.isfinite(smooth_spread[i]), f"smooth hexa[{i}] not finite")
            # Workaround: capped lumen hexa are frozen (spread preserved).
            self.assertAlmostEqual(
                raw_spread[i], smooth_spread[i], delta=0.05 * max(1.0, raw_spread[i]),
                msg=f"wall-included capped lumen hexa[{i}] should be preserved",
            )

    def test_no_wall_bug_baseline(self):
        """Baseline (buggy) path: capped tube without the wall included.

        We pin the *currently-observed* behavior so the fix can be validated
        against it. Right now smoothing still terminates and stays finite, but
        the cap does not resolve to a flat cap (the reported defect). This test
        is intentionally GREEN — it documents the invariants that hold today;
        the assertion that *would* fail (a flat, correctly-associated cap) is
        deferred to the ROADMAP P0 fix and will be tightened there.
        """
        nj = NJuncDATA(numTubes=5, mixIOFLAG=True, extrdLen=1.0, bevelDeg=0.0,
                       randomLen=True, WallFLAG=False, regDirTubesFLAG=True)
        np.random.seed(_SEED)
        self.assertTrue(_build(nj))

        ioLetsHexa = np.asarray(nj.ioLetsHexa, dtype=bool)
        capped = [i for i, v in enumerate(ioLetsHexa) if not v]
        self.assertGreaterEqual(len(capped), 1, "expected at least one capped tube")

        mpS, ok = run_smooth_raw_scaff(nj)
        self.assertTrue(ok)
        self.assertIsNotNone(mpS)
        self.assertGreater(len(mpS), 0)

        smooth_spread = _axial_spread(mpS)
        for i in capped:
            self.assertTrue(np.isfinite(smooth_spread[i]),
                            f"capped hexa[{i}] should stay finite after smoothing")

if __name__ == "__main__":
    unittest.main()
