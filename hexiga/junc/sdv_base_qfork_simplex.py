"""
hexiga.junc.sdv_base_qfork_simplex
==================================

Iteratively subdivide (split) every cluster that contains more than one direction into
two Cis/Trans sub-clusters, growing the QFS face/point arrays until every
cluster holds a single direction.

Signature::

    [QFSfcs, QFSpts, idx, isCis, itrQFS, lblCis] = \
        sdvBaseQForkSimplex(dirTubes, QFSfcs, QFSpts, idx, lblCis)

    % dirTubes: 3xN, QFSfcs: 4xf, QFSpts: 3xn, idx: 1xN, lblCis: 1xN logical

Notes
-----
* ``itrQFS`` is a MATLAB struct-array; here it is returned as a Python list
  of ``dict``s with keys ``'QFSfcs'``, ``'QFSpts'``, ``'QFSidx'`` (snapshots).
* ``QFSfcs`` / ``idx`` values are **1-based** (MATLAB convention).
* A safety ``max_iterations`` guard is added (the MATLAB while-loop is
  unbounded); it never triggers for valid N-junction input.
"""

from __future__ import annotations

import numpy as np

from .get_lbls2split import getLbls2split
from .map_cis_trans_qfs import mapCisTransQFSs
from .map_split_indices import mapSplitIndices
from .split_cluster_dir_tubes import splitClusterDirTubes

__all__ = ["sdvBaseQForkSimplex"]

def sdvBaseQForkSimplex(
    dirTubes: np.ndarray,
    QFSfcs: np.ndarray,
    QFSpts: np.ndarray,
    idx: np.ndarray,
    lblCis: np.ndarray,
):
    """Iteratively split multi-direction clusters into Cis/Trans sub-clusters.

    Returns ``(QFSfcs, QFSpts, idx, isCis, itrQFS, lblCis)`` where
    ``itrQFS`` is a list of dicts ``{'QFSfcs','QFSpts','QFSidx'}``.
    """
    dirTubes = np.asarray(dirTubes, dtype=float)
    QFSfcs = np.asarray(QFSfcs, dtype=int)
    QFSpts = np.asarray(QFSpts, dtype=float)
    idx = np.asarray(idx, dtype=int).copy()
    lblCis = np.asarray(lblCis, dtype=bool).copy()

    converged = False
    isCis = True

    # new: history of the QFSfcs labels iteratively assigned
    itrQFS = [{"QFSfcs": QFSfcs.copy(), "QFSpts": QFSpts.copy(), "QFSidx": idx.copy()}]
    itr = 1

    max_iterations = 100  # safety net (MATLAB while-loop is unbounded)

    while (not converged) and itr < max_iterations:
        lbls2split = getLbls2split(idx)

        if lbls2split.size > 0:
            for ll in range(lbls2split.size):
                itr += 1
                lbl = int(lbls2split[ll])
                idx2split = idx == lbl
                dTs = dirTubes[:, idx2split]
                QFSfc = QFSfcs[:, lbl - 1]
                QFSpt = QFSpts[:, QFSfc - 1]

                M, FlagCIS, idxSPLIT, fatalErr = splitClusterDirTubes(dTs, QFSpt)

                if fatalErr:
                    converged = True
                    print(" *** Fatal Error Occurred!  - Skipping ***")
                else:
                    # Concatenate, Arrange & Integrate
                    QFSpts = np.column_stack([QFSpts, M])
                    Mlbl = QFSpts.shape[1]  # 1-based index of the new column

                    QFSfcsM = mapCisTransQFSs(QFSfc, Mlbl, FlagCIS)  # 4x2

                    lblNew = int(np.max(idx)) + 1  # == K+1
                    K = QFSfcs.shape[1]
                    QFSfcs_new = np.empty((4, K + 1), dtype=int)
                    QFSfcs_new[:, :K] = QFSfcs
                    QFSfcs_new[:, lbl - 1] = QFSfcsM[:, 0]
                    QFSfcs_new[:, K] = QFSfcsM[:, 1]
                    QFSfcs = QFSfcs_new

                    idxSPLITmap = mapSplitIndices(idxSPLIT, lbl, lblNew)
                    idx[idx2split] = idxSPLITmap
                    lblCis[idx2split] = FlagCIS

                    isCis = isCis and FlagCIS

                    itrQFS.append(
                        {"QFSfcs": QFSfcs.copy(), "QFSpts": QFSpts.copy(), "QFSidx": idx.copy()}
                    )

        else:
            converged = True

    # -- Compact face labels to a dense 1..N permutation (resolve label gaps). --
    #
    # Convergence of the splitting loop (no face holds >1 direction, no
    # direction is unassigned) guarantees that the ``N`` directions occupy
    # ``N`` *distinct* faces -- but the numeric face labels may have gaps or
    # be out of order (e.g. directions on faces {1, 3, 4, 5}).  Downstream
    # consumers (``makeQFSs2nrbHexa`` / ``getExtrudeBevelQFS``) require
    # ``idx`` to be an exact permutation of 1..N with ``QFSfcs`` having one
    # column per direction, so we relabel the faces densely and reorder the
    # ``QFSfcs`` columns to match.  (This is the N>3 generalisation of the
    # MATLAB algorithm, which only produces a clean 1..N assignment for the
    # 3-tube case; for >3 the original ``splitClusterDirTubes`` raises a
    # ``[BUG]`` gate.  See ``split_cluster_dir_tubes.py``.)
    idx = np.asarray(idx, dtype=int)
    if (
        idx.size
        and np.all(idx > 0)
        and len(set(idx.tolist())) == idx.size
    ):
        # Select EXACTLY the ``N`` faces referenced by ``idx`` (one per
        # direction) and reorder so that face column ``j-1`` is the face that
        # holds direction ``j``.  This both remaps the labels to a dense 1..N
        # AND drops any leftover empty face (the split recursion can leave an
        # extra face with no direction assigned, e.g. ``nfaces = N + 1``).
        # After this, ``QFSfcs`` has exactly ``N`` columns and
        # ``makeQFSs2nrbHexa``'s ``size(QFSfcs,2) == length(idx)`` invariant
        # holds for every N (not just 3).
        if QFSfcs.shape[1] != idx.size or not np.array_equal(
            np.sort(idx), np.arange(1, idx.size + 1)
        ):
            keep = [int(x) - 1 for x in idx]  # 0-based column for each direction
            QFSfcs = QFSfcs[:, keep]
            idx = np.arange(1, idx.size + 1, dtype=int)

    return QFSfcs, QFSpts, idx, isCis, itrQFS, lblCis
