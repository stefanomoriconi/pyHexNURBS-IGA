#!/usr/bin/env python3
"""
tests/test_nurbs_c_backend.py — cross-checks the optional C accelerator
(``hexiga.nurbs._c``) against the pure-NumPy kernel reference
(``hexiga.nurbs._kernels``).

The C backend is optional: it must be built locally with
``bash hexiga/nurbs/_c/build.sh`` before it can be imported. If the
compiled shared library is not present, every test here is skipped.
"""

from __future__ import annotations
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexiga.nurbs import _kernels  # noqa: E402

try:
    from hexiga.nurbs import _c as _cbackend  # noqa: E402
    _C_AVAILABLE = True
except Exception:
    _cbackend = None
    _C_AVAILABLE = False

def _make_clamped_knots(rng, degree, n_internal):
    internal = np.sort(rng.uniform(0.0, 1.0, n_internal))
    return np.concatenate([np.zeros(degree + 1), internal, np.ones(degree + 1)])

@unittest.skipUnless(_C_AVAILABLE, "hexiga.nurbs._c is not built (run build.sh)")
class TestNurbsCBackend(unittest.TestCase):
    """Randomized cross-checks of every C kernel against the NumPy oracle."""

    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_findspan_basisfun(self):
        rng = self.rng
        for _ in range(200):
            degree = int(rng.integers(1, 5))
            n_internal = int(rng.integers(0, 10))
            n_ctrl = degree + 1 + n_internal
            k = _make_clamped_knots(rng, degree, n_internal)
            u = float(rng.uniform(0.0, 1.0))

            s1 = _kernels.findspan(n_ctrl, degree, u, k)
            s2 = _cbackend.findspan(n_ctrl, degree, u, k)
            self.assertEqual(s1, s2)

            b1 = _kernels.basisfun(s1, u, degree, k)
            b2 = _cbackend.basisfun(s2, u, degree, k)
            np.testing.assert_allclose(b1, b2, atol=1e-12)

    def test_bspeval(self):
        rng = self.rng
        for _ in range(100):
            degree = int(rng.integers(1, 5))
            n_internal = int(rng.integers(0, 10))
            n_ctrl = degree + 1 + n_internal
            k = _make_clamped_knots(rng, degree, n_internal)
            c = rng.uniform(-1.0, 1.0, (4, n_ctrl))
            u = np.sort(rng.uniform(0.0, 1.0, int(rng.integers(1, 50))))

            p1 = _kernels.bspeval(degree, c, k, u)
            p2 = _cbackend.bspeval(degree, c, k, u)
            np.testing.assert_allclose(p1, p2, atol=1e-9)

    def test_bspderiv(self):
        rng = self.rng
        for _ in range(100):
            degree = int(rng.integers(1, 5))
            n_internal = int(rng.integers(0, 10))
            n_ctrl = degree + 1 + n_internal
            k = _make_clamped_knots(rng, degree, n_internal)
            c = rng.uniform(-1.0, 1.0, (4, n_ctrl))

            dc1, dk1 = _kernels.bspderiv(degree, c, k)
            dc2, dk2 = _cbackend.bspderiv(degree, c, k)
            np.testing.assert_allclose(dc1, dc2, atol=1e-9)
            np.testing.assert_allclose(dk1, dk2, atol=1e-9)

    def test_bspkntins(self):
        rng = self.rng
        for _ in range(100):
            degree = int(rng.integers(1, 5))
            n_internal = int(rng.integers(0, 10))
            n_ctrl = degree + 1 + n_internal
            k = _make_clamped_knots(rng, degree, n_internal)
            c = rng.uniform(-1.0, 1.0, (4, n_ctrl))
            u_ins = np.sort(rng.uniform(0.0, 1.0, int(rng.integers(1, 5))))

            ic1, ik1 = _kernels.bspkntins(degree, c, k, u_ins)
            ic2, ik2 = _cbackend.bspkntins(degree, c, k, u_ins)
            np.testing.assert_allclose(ic1, ic2, atol=1e-9)
            np.testing.assert_allclose(ik1, ik2, atol=1e-9)

    def test_bspdegelev(self):
        rng = self.rng
        for _ in range(300):
            degree = int(rng.integers(1, 4))
            n_internal = int(rng.integers(0, 8))
            n_ctrl = degree + 1 + n_internal
            k = _make_clamped_knots(rng, degree, n_internal)
            c = rng.uniform(-1.0, 1.0, (4, n_ctrl))
            t = int(rng.integers(1, 4))

            ic1, ik1 = _kernels.bspdegelev(degree, c, k, t)
            ic2, ik2 = _cbackend.bspdegelev(degree, c, k, t)
            self.assertEqual(ic1.shape, ic2.shape)
            np.testing.assert_allclose(ic1, ic2, atol=1e-9)
            np.testing.assert_allclose(ik1, ik2, atol=1e-9)

    def test_end_to_end_backend_switch(self):
        """Full nrbeval round-trip matches with HEXIGA_BACKEND=nurbs_c.

        ``_backend`` resolves the kernel module once at import time, so the
        two backends are compared across separate subprocesses rather than
        by toggling the environment variable mid-process.
        """
        import subprocess
        import textwrap

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = textwrap.dedent(
            """
            import numpy as np
            from hexiga.nurbs import nrbmak, nrbeval

            rng = np.random.default_rng(12345)
            degree, n_internal = 3, 4
            n_ctrl = degree + 1 + n_internal
            internal = np.sort(rng.uniform(0.0, 1.0, n_internal))
            k = np.concatenate([np.zeros(degree + 1), internal, np.ones(degree + 1)])
            c = rng.uniform(-1.0, 1.0, (4, n_ctrl))
            c[3, :] = 1.0
            crv = nrbmak(c, (k,))
            u = np.linspace(0.0, 1.0, 25)
            cp, cw = nrbeval(crv, (u,))
            np.save("/tmp/_hexiga_c_backend_test_cp.npy", cp)
            np.save("/tmp/_hexiga_c_backend_test_cw.npy", cw)
            """
        )

        env_numpy = dict(os.environ)
        env_numpy.pop("HEXIGA_BACKEND", None)
        subprocess.run(
            [sys.executable, "-c", script], cwd=repo_root, env=env_numpy, check=True
        )
        cp_numpy = np.load("/tmp/_hexiga_c_backend_test_cp.npy")
        cw_numpy = np.load("/tmp/_hexiga_c_backend_test_cw.npy")

        env_c = dict(os.environ)
        env_c["HEXIGA_BACKEND"] = "nurbs_c"
        subprocess.run(
            [sys.executable, "-c", script], cwd=repo_root, env=env_c, check=True
        )
        cp_c = np.load("/tmp/_hexiga_c_backend_test_cp.npy")
        cw_c = np.load("/tmp/_hexiga_c_backend_test_cw.npy")

        np.testing.assert_allclose(cp_numpy, cp_c, atol=1e-9)
        np.testing.assert_allclose(cw_numpy, cw_c, atol=1e-9)

if __name__ == "__main__":
    unittest.main()
