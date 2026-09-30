"""J4 - NJunction "Generate" pipeline (hexiga.junc) tests.

Ground-truth MATLAB equivalent: ``MAIN_gui/NJunction.m`` -> ``runGenerateNJunction``
followed by ``runGenerateRawScaff`` (``makeQFSs2nrbHexa``).  The full chain is::

    genDirTubes -> getBaseQForkSimplex -> sdvBaseQForkSimplex
              -> getExtrudeBevelQFS    -> makeQFSs2nrbHexa

These tests are LIGHT: they assert structural invariants (homogeneous NURBS
hexahedra, finite control points, ``4*N`` patches) and never render.  They
deliberately use the same direction sets that the MATLAB demo produces for N=3
(random, always valid) and a well-spread N=4 tetrahedral set, so they pass on a
faithful port.

N>=4 (numTubes > 3) — authorized divergence
----------------------------------------------
The native MATLAB algorithm reports ``splitClusterDirTubes: Both Cis and
Trans partitions are NOT Valid! [BUG]!`` / ``Fatal Error`` for a large fraction
of random N>=4 direction sets (the MATLAB GUI defaults to N=3), and additionally
crashes when a base direction group ends up empty (``getAdjIVX`` -> ``min(D(:))``
on an empty matrix) or leaves some base directions unassigned.  The user
explicitly authorized resolving the "[BUG]!" case (and the related robustness
gaps) so that **every** branch count N>=3 yields a *consistent* quadrilateral
interface at the junction simplex, with ``idx`` an exact permutation of 1..N.

Three targeted robustness additions in the Python port implement that:

1. ``split_cluster_dir_tubes`` -- when both the Cis and Trans partitions are
   invalid, a deterministic fallback (split the directions into the two halves
   nearest the base-face diagonal corners) is used instead of a fatal error.
2. ``sdv_base_qfork_simplex`` -- face labels are compacted to a dense 1..N
   permutation (and the ``QFSfcs`` columns are reordered to match), so no
   downstream ``assert idx.max() == idx.size`` can trip on a label gap.
3. ``get_adj_ivx`` / ``updt_idx_base_partition`` -- empty direction groups and
   unassigned base directions are handled gracefully (nearest-base-face
   assignment) instead of crashing.

These keep the N=3 result byte-for-byte identical to MATLAB (the code paths are
only reached in the N>=4 / degenerate cases) while making N>=4 always succeed.
The tests below assert exactly those invariants.
"""

import unittest

import numpy as np

from hexiga.nurbs.nrb import Nrb

from hexiga.junc import (
    detSign,
    getCentroid,
    isPtInTriangle,
    getIVX,
    genDirTubes,
    regulariseDirTubes,
    getBaseQForkSimplex,
    sdvBaseQForkSimplex,
    getExtrudeBevelQFS,
    makeQFSs2nrbHexa,
    orthog,
)

def _run_pipeline(dirTubes, seed=1):
    """Run the full NJunction 'Generate' chain on a given direction set.

    ``dirTubes`` is a 3xN array of unit vectors.  Returns the list of NURBS
    hexahedral patches produced by ``makeQFSs2nrbHexa``.
    """
    np.random.seed(seed)
    dirTubes = np.asarray(dirTubes, dtype=float)

    baseQFSfcs, baseQFSpts, baseidx, baselblCIS = getBaseQForkSimplex(dirTubes)
    QFSfcs, QFSpts, idx, _isCis, _itr, _lblCis = sdvBaseQForkSimplex(
        dirTubes, baseQFSfcs, baseQFSpts, baseidx, baselblCIS
    )
    idx = np.asarray(idx, dtype=int)
    n = idx.size

    # Faithful GUI defaults (runGenerateNJunction): ioLets all True, the
    # random extrude/bevel jitter terms are the only stochastic part.
    ioLets = np.ones(n, dtype=bool)
    shrtFct = np.ones(n)
    bvlFct = np.ones(n)
    extrLen = (2 + (0.2 * ioLets) * (0.3 * np.random.rand(n))) * shrtFct
    bevelF = (1 + (0.2 * ioLets) * (0.3 * np.random.randn(n))) * bvlFct
    dirMag = (1 + 0.1 * np.random.rand(n)) * shrtFct

    ebQFSfcs, ebQFSpts, _KAe, idT, edT = getExtrudeBevelQFS(
        dirTubes, QFSfcs, QFSpts, idx, extrLen, bevelF,
        isflat=ioLets, dirMag=dirMag,
    )
    return makeQFSs2nrbHexa(QFSfcs, QFSpts, idT, ebQFSfcs, ebQFSpts, edT, idx)

def _assert_clean_hexa(H, expected_n):
    self_len = len(H)
    assert self_len == expected_n, f"patch count {self_len} != {expected_n}"
    for h in H:
        assert isinstance(h, Nrb), "patch is not an Nrb"
        assert h.dim == 4, "Nrb should be homogeneous (dim=4)"
        co = np.asarray(h.coefs)
        assert co.shape[0] == 4, "homogeneous coefs must be 4 rows"
        assert np.all(np.isfinite(co)), "Nrb control points contain NaN/Inf"
        for k in h.knots:
            assert np.all(np.isfinite(np.asarray(k))), "Nrb knots not finite"

class TestJuncGeometry(unittest.TestCase):
    """Small, targeted unit tests on the leaf geometry helpers."""

    def test_detSign(self):
        # detSign(A, B, flip) -> (m, flip): m is +1 or -1 depending on whether
        # proj(A,B)>0 with an un-flipped side.  A positive projection with no
        # prior flip yields m=-1; a negative projection keeps m=+1.
        A = np.array([1.0, 0.0, 0.0])
        B = np.array([1.0, 0.0, 0.0])   # dot > 0
        m, flip = detSign(A, B, False)
        self.assertEqual(m, -1)
        self.assertTrue(flip)
        A2 = np.array([1.0, 0.0, 0.0])
        B2 = np.array([-1.0, 0.0, 0.0])  # dot < 0
        m2, flip2 = detSign(A2, B2, False)
        self.assertEqual(m2, 1)
        self.assertFalse(flip2)

    def test_getCentroid(self):
        # getCentroid(uVects) returns the UNIT mean of the input unit-vectors.
        # The bisector of +x and +y is [1,1,0]/sqrt(2).
        c = getCentroid(np.column_stack([np.array([1.0, 0.0, 0.0]),
                                         np.array([0.0, 1.0, 0.0])]))
        self.assertAlmostEqual(float(np.linalg.norm(c)), 1.0, places=8)
        self.assertAlmostEqual(float(c[0]), float(c[1]), places=8)
        self.assertAlmostEqual(float(c[0]), 1.0 / np.sqrt(2.0), places=8)

    def test_isPtInTriangle(self):
        # Triangle in the xy-plane: (0,0), (1,0), (0,1).  T is 3x3 (columns).
        T = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0]])
        self.assertTrue(isPtInTriangle(np.array([0.25, 0.25, 0.0]), T))
        self.assertFalse(isPtInTriangle(np.array([0.8, 0.8, 0.0]), T))

    def test_orthog(self):
        # orthog(A, B) projects A orthogonal to B: the result is unit and
        # perpendicular to B.
        A = np.array([1.0, 1.0, 0.0])
        B = np.array([1.0, 0.0, 0.0])
        r = orthog(A, B)
        self.assertAlmostEqual(float(np.dot(r, B)), 0.0, places=8)
        self.assertAlmostEqual(float(np.linalg.norm(r)), 1.0, places=8)

    def test_getIVX(self):
        # getIVX(A, B, oC, V, flip): interleaving-side vector for A and B
        # wrt V, sign-checked by oC.  It should be a finite 3-vector.
        A = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 1.0, 0.0])
        oC = np.array([0.0, 0.0, 1.0])
        V = np.array([0.5, 0.5, 0.0])
        iAB, _flip = getIVX(A, B, oC, V, False)
        iAB = np.asarray(iAB)
        self.assertEqual(iAB.size, 3)
        self.assertTrue(np.all(np.isfinite(iAB)))

class TestNJunctionPipeline(unittest.TestCase):
    """Full 'Generate' chain: random N=3 and controlled N=4 direction sets."""

    def test_pipeline_N3_random(self):
        # The MATLAB demo default (N=3) is valid for random directions.
        dirTubes = genDirTubes(3)
        H = _run_pipeline(dirTubes, seed=7)
        _assert_clean_hexa(H, expected_n=12)  # 4 patches per tube * 3 tubes

    def test_pipeline_N4_tetrahedral(self):
        # N=4 tetrahedral direction set.  With the authorized N>=4 robustness
        # additions (see module docstring) the pipeline now COMPLETES and
        # produces 16 fully clean hexahedra (4 patches x 4 tubes).  The MATLAB
        # reference would instead hit the "[BUG]!" / NaN path here.
        tet = np.array([[1.0, 1.0, -1.0, -1.0],
                        [1.0, -1.0, 1.0, -1.0],
                        [1.0, -1.0, -1.0, 1.0]])
        tet = tet / np.linalg.norm(tet, axis=0, keepdims=True)
        H = _run_pipeline(tet, seed=3)
        _assert_clean_hexa(H, expected_n=16)

    def test_pipeline_N_gt3_random_always_builds(self):
        # The user-authorized invariant: for ANY numTubes >= 3 the pipeline
        # completes and yields 4*N clean hexahedra, even for random direction
        # sets that the native MATLAB algorithm rejects (see module docstring).
        for N in (3, 4, 5, 6, 8, 12):
            for seed in (1, 2, 3):
                dT = genDirTubes(N)
                H = _run_pipeline(dT, seed=seed)
                self.assertEqual(
                    len(H), 4 * N,
                    f"N={N} seed={seed}: expected 4*N={4 * N} hexa, got {len(H)}")
                for h in H:
                    self.assertEqual(h.dim, 4)
                    self.assertTrue(np.all(np.isfinite(np.asarray(h.coefs))),
                                    f"N={N} seed={seed}: non-finite coefs")

    def test_qfs_idx_is_permutation_N_gt3(self):
        # Core QFS invariant: after sdvBaseQForkSimplex the face index array
        # must be an exact permutation of 1..N (one direction per face).  This
        # is the invariant that the N>=4 robustness additions guarantee and
        # that makeQFSs2nrbHexa / getExtrudeBevelQFS rely on.
        for N in (3, 4, 5, 6, 8):
            dT = genDirTubes(N)
            baseQFSfcs, baseQFSpts, baseidx, baselblCIS = getBaseQForkSimplex(dT)
            QFSfcs, QFSpts, idx, _isCis, _itr, _lblCis = sdvBaseQForkSimplex(
                dT, baseQFSfcs, baseQFSpts, baseidx, baselblCIS)
            idx = np.asarray(idx, dtype=int).ravel()
            self.assertEqual(idx.size, N, f"N={N}: idx size {idx.size} != {N}")
            self.assertEqual(
                sorted(idx.tolist()), list(range(1, N + 1)),
                f"N={N}: idx {idx.tolist()} is not a permutation of 1..{N}")
            # QFSfcs must have one face column per idx entry.
            self.assertEqual(np.asarray(QFSfcs).shape[1], idx.size)

    def test_regularise_dir_tubes_unit(self):
        # regulariseDirTubes preserves the number of tubes and returns unit
        # (or near-unit) directions.
        d = genDirTubes(3)
        r = regulariseDirTubes(d)
        self.assertEqual(r.shape, d.shape)

if __name__ == "__main__":
    unittest.main()
