"""
A3 CAD-demo regression tests.

Covers the seven CAD demos under ``hexiga/demo/``:

    demoSocketCAD    --  32 patches
    demoHooksCAD     --  14 patches
    demoStentCAD     --  120 patches
    demoGearCAD      --  54 patches
    demoTurbineCAD   -- (48, 36) tuples
    demoPlateCAD     --  72 patches (32 + 8 + 32)
    demoTPipeCAD     -- 24 non-merged / 12 merged

Invariants (per patch):
    * ``dim == 4`` (homogeneous coordinates -- standing directive).
    * ``tuple(order) == (4, 4, 3)`` or ``(4,4,4)`` (cubic NURBS hexa).
    * Coefs contain no NaN / Inf (guards against the T-Pipe NaN regression).

Faithful literal port of the MATLAB ground truth under
``demo_utils/CAD/`` (read-only):
    demoSocketCAD.m   -- preserves RIN/ROUT parse-typo (writes to
                         ``opts.replicates`` / ``opts.zLevels`` instead of
                         ``opts.rin`` / ``opts.rout``).
    demoTPipeCAD.m    -- preserves 'SMOOTH' parse-typo printing
                         ``' * demoAsymSocketCAD: Unrecognised Parsed
                         Parameter: ...'``.
    demoStentCAD.m / demoGearCAD.m -- preserve 'gearCAD' typo in parse
                         messages.
    Smooth paths raise NotImplementedError (mpTurboSmooth not yet ported,
     milestone A4).

Run:
    cd C:\\HexIGA_Python
    & "C:\\...\\Python311\\python.exe" -m unittest tests.test_cad -v
"""

import contextlib
import io
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexiga.nurbs import Nrb
from hexiga.demo import (
    demoSocketCAD,
    demoHooksCAD,
    demoStentCAD,
    demoGearCAD,
    demoTurbineCAD,
    demoPlateCAD,
    demoTPipeCAD,
)

# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------
def _check_patches(patches, label: str) -> None:
    for i, p in enumerate(patches):
        assert isinstance(p, Nrb), f"{label}[{i}] not an Nrb: {type(p)}"
        assert p.dim == 4, f"{label}[{i}] dim={p.dim} (expected 4)"
        assert p.coefs.shape[0] == 4, \
            f"{label}[{i}] coefs.shape[0]={p.coefs.shape[0]} (expected 4)"
        w = np.asarray(p.coefs[3], dtype=float)
        assert np.all(np.isfinite(w)), f"{label}[{i}] non-finite weight row"
        assert np.all(np.isfinite(p.coefs)), \
            f"{label}[{i}] NaN/Inf in coefs"
        # Order check: each axis must be order 4 (cubic: degree 3, order 4).
        o = tuple(int(x) for x in np.asarray(p.order).ravel())
        assert o == (4, 4, 4), f"{label}[{i}] order={o} (expected (4,4,4))"

class _BaseCAD(unittest.TestCase):
    """Common helpers: non-smooth invocation + invariant assertions.

    DEMO must be wrapped in ``staticmethod(...)`` so that ``self.DEMO``
    returns the raw function (not a bound method with ``self`` injected
    as the first argument -- which would shift the ``*varargs`` pairs).
    """

    DEMO = None  # overridden per test class (staticmethod-wrapped)
    EXPECTED_COUNTS = None  # int or tuple of ints

    def _call(self):
        # Non-smooth, non-merge path -- the only branch currently ported.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = self.DEMO("smooth", False, "merge", False)
        return result

    def _expected_count(self):
        e = self.EXPECTED_COUNTS
        if isinstance(e, int):
            return [e]
        return list(e)

# ---------------------------------------------------------------------------
# 1. demoSocketCAD
# ---------------------------------------------------------------------------
class TestDemoSocketCAD(_BaseCAD):
    DEMO = staticmethod(demoSocketCAD)
    EXPECTED_COUNTS = 32

    def test_patch_count(self):
        r = self._call()
        self.assertEqual(len(r), self._expected_count()[0])

    def test_invariants(self):
        _check_patches(self._call(), "socket")

    def test_smooth_runs(self):
        # A4: the smoothing path is now implemented and must succeed on
        # the socket demo, preserving the 32-patch count and returning
        # NaN-free control-point data.
        r = demoSocketCAD("smooth", True)
        self.assertEqual(len(r), 32)
        for p in r:
            self.assertFalse(np.isnan(p.coefs).any(),
                             f"patch {p.number} has NaN coefs after smoothing")

    def test_unknown_param_prints_typo(self):
        # MATLAB prints 'demoSocketCAD: Unrecognised Parsed Parameter: ...'
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demoSocketCAD("SMOOTH", False, "MERGE", False, "RIN", 1.234)
        msg = buf.getvalue()
        self.assertIn("demoSocketCAD", msg)
        self.assertIn("Unrecognised Parsed Parameter", msg)

    def test_rin_rout_override_bug(self):
        # MATLAB typo: 'RIN' writes to opts.replicates, 'ROUT' to
        # opts.zLevels.  The actual genSocket call still uses the
        # defaults (rin=3, rout=4) -- so overriding RIN/ROUT has NO
        # effect on the output.  Ported faithfully.
        a = demoSocketCAD("smooth", False, "merge", False)
        b = demoSocketCAD("smooth", False, "merge", False, "RIN", 0.5, "ROUT", 2.0)
        self.assertEqual(len(a), len(b))
        for pa, pb in zip(a, b):
            np.testing.assert_allclose(pa.coefs, pb.coefs, rtol=0, atol=1e-12)

# ---------------------------------------------------------------------------
# 2. demoHooksCAD
# ---------------------------------------------------------------------------
class TestDemoHooksCAD(_BaseCAD):
    DEMO = staticmethod(demoHooksCAD)
    EXPECTED_COUNTS = 14

    def test_patch_count(self):
        r = self._call()
        self.assertEqual(len(r), self._expected_count()[0])

    def test_invariants(self):
        _check_patches(self._call(), "hooks")

# ---------------------------------------------------------------------------
# 3. demoStentCAD
# ---------------------------------------------------------------------------
class TestDemoStentCAD(_BaseCAD):
    DEMO = staticmethod(demoStentCAD)
    EXPECTED_COUNTS = 120

    def test_patch_count(self):
        r = self._call()
        self.assertEqual(len(r), self._expected_count()[0])

    def test_invariants(self):
        _check_patches(self._call(), "stent")

# ---------------------------------------------------------------------------
# 4. demoGearCAD
# ---------------------------------------------------------------------------
class TestDemoGearCAD(_BaseCAD):
    DEMO = staticmethod(demoGearCAD)
    EXPECTED_COUNTS = 54

    def test_patch_count(self):
        r = self._call()
        self.assertEqual(len(r), self._expected_count()[0])

    def test_invariants(self):
        _check_patches(self._call(), "gear")

# ---------------------------------------------------------------------------
# 5. demoTurbineCAD (returns (turbine, fan) tuple)
# ---------------------------------------------------------------------------
class TestDemoTurbineCAD(unittest.TestCase):
    def _call(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return demoTurbineCAD("smooth", False, "merge", False)

    def test_returns_two_lists(self):
        T, F = self._call()
        self.assertEqual(len(T), 48)
        self.assertEqual(len(F), 36)

    def test_invariants_t(self):
        T, _ = self._call()
        _check_patches(T, "turbine-T")

    def test_invariants_f(self):
        _, F = self._call()
        _check_patches(F, "turbine-F")

# ---------------------------------------------------------------------------
# 6. demoPlateCAD
# ---------------------------------------------------------------------------
class TestDemoPlateCAD(_BaseCAD):
    DEMO = staticmethod(demoPlateCAD)
    EXPECTED_COUNTS = 72

    def test_patch_count(self):
        r = self._call()
        self.assertEqual(len(r), self._expected_count()[0])

    def test_invariants(self):
        _check_patches(self._call(), "plate")

# ---------------------------------------------------------------------------
# 7. demoTPipeCAD
# ---------------------------------------------------------------------------
class TestDemoTPipeCAD(_BaseCAD):
    DEMO = staticmethod(demoTPipeCAD)
    EXPECTED_COUNTS = 24

    def test_patch_count_non_merged(self):
        r = self._call()
        self.assertEqual(len(r), 24)

    def test_patch_count_merged(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = demoTPipeCAD("smooth", False, "merge", True)
        self.assertEqual(len(r), 12)

    def test_invariants_non_merged(self):
        _check_patches(self._call(), "tpipe-nm")

    def test_invariants_merged(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = demoTPipeCAD("smooth", False, "merge", True)
        _check_patches(r, "tpipe-m")

    def test_unknown_param_prints_asymsocket_typo(self):
        # MATLAB typo: 'SMOOTH' parse prints 'demoAsymSocketCAD: ...'
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demoTPipeCAD("foo", 1.234, "smooth", False, "merge", False)
        msg = buf.getvalue()
        self.assertIn("demoAsymSocketCAD", msg)
        self.assertIn("Unrecognised Parsed Parameter", msg)
        self.assertIn("1.234", msg)

if __name__ == "__main__":
    unittest.main(verbosity=2)
