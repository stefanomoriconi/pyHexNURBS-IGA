"""
hexiga.junc.clstr_dirs
======================

Cluster tubular directions
into ``numClusters`` groups using k-medoids.

MATLAB's ``kmedoids`` (Statistics Toolbox) is not available in numpy/scipy, so
a pure-numpy PAM (Partitioning Around Medoids) implementation is provided as a
faithful substitute: it selects *actual data points* as medoids and minimises
the total euclidean distance to their assigned medoids, exactly matching the
``kmedoids(..., 'Distance','euclidean')`` objective. The medoids (the selected
points themselves) are returned, matching MATLAB's ``[~, C]`` output.

Signature::

    C = clstrDirs(dirTubes, numClusters)   % dirTubes: 3xN, numClusters: n<=N, C: 3xn
"""

from __future__ import annotations

import numpy as np

from .sort_dir3_tubes import sortDir3Tubes

__all__ = ["clstrDirs"]

def _pam(X: np.ndarray, k: int) -> np.ndarray:
    """Pure-numpy k-medoids (PAM). ``X`` is (n x d); returns (n x k) medoids.

    1. Pick the first ``k`` points as initial medoids.
    2. Assign each point to its nearest medoid.
    3. Swap: for each non-medoid candidate, evaluate the total-distance change
       of swapping it in for each medoid; keep the best improving swap.
    4. Repeat steps 2-3 until no improving swap exists.
    """
    n, d = X.shape
    X = np.atleast_2d(X)
    n = X.shape[0]

    # distance matrix D[i,j]
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    medoids = list(range(k))

    def total_cost(meds):
        meds = np.asarray(meds, dtype=int)
        nearest = np.argmin(D[:, meds], axis=1)
        return float(np.sum(D[np.arange(n), meds[nearest]]))

    best = total_cost(medoids)
    improved = True
    while improved:
        improved = False
        best_swap = None
        best_swap_cost = best

        # candidates = non-medoid points
        nonmed = [i for i in range(n) if i not in medoids]
        for cand in nonmed:
            for r, m in enumerate(medoids):
                new_meds = medoids[:]
                new_meds[r] = cand
                cost = total_cost(new_meds)
                if cost < best_swap_cost - 1e-12:
                    best_swap_cost = cost
                    best_swap = (new_meds, cost)

        if best_swap is not None:
            medoids, best = best_swap
            improved = True

    return X[medoids, :]  # (k x d) medoids as rows; caller transposes to (d x k)

def clstrDirs(dirTubes: np.ndarray, numClusters: int) -> np.ndarray:
    """Cluster ``dirTubes`` (3 x N) into ``numClusters`` groups (3 x numClusters)."""
    dirTubes = np.asarray(dirTubes, dtype=float)
    # MATLAB jitters the input to break exact-tie degeneracies:
    jittered = dirTubes.T + 0.01 * np.random.randn(dirTubes.shape[1], 3)
    medoids = _pam(jittered, numClusters)  # (n x numClusters) medoid points
    C = medoids.T  # (numClusters x n) -> (3 x numClusters) when numClusters==3

    C = sortDir3Tubes(C)
    return C
