"""
hexiga.iga.boundary_map
=======================

[idxs, dvsr] = mapBoundariesHtoHsbdv(HsbdvPrnt, HsbdvBndrs)

Given a parent boundary list (``HsbdvPrnt``) and a subdivided boundary list
(``HsbdvBndrs``), map each subdivided boundary side to the *parent* side it
belongs to, plus a divisor (how many subdivided sides share that parent side)
so the parent's magnitude can be redistributed.

Input contract:

* ``HsbdvPrnt``  -- list (or array-like) of parent boundary entries; each
  exposes ``.patches`` and ``.faces`` (scalar ints).
* ``HsbdvBndrs`` -- list (or array-like) of subdivided boundary entries; each
  exposes ``.patches`` and ``.faces`` (scalar ints).

Returns a tuple ``(idxs, dvsr)`` of 1-D ``np.ndarray`` (``int64``) of
length ``len(HsbdvBndrs)``:

* ``idxs[k]``  -- 1-based index into ``HsbdvPrnt`` for the k-th subdivided side.
* ``dvsr[k]``  -- divisor (count of subdivided sides that share the same
  parent side as ``idxs[k]``).

The MATLAB code uses ``unique(.., 'rows')`` which returns unique rows in
lexicographic order; we reproduce that with ``np.unique(axis=0, return_inverse=True)``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["map_boundaries_h_to_h_sbdv"]

def _pf_array(entries: Sequence) -> np.ndarray:
    """Build the ``[patches, faces]`` (n, 2) int array, mirroring
    ``pfIDs = [patchIDs, faceIDs]`` in the MATLAB."""
    if len(entries) == 0:
        return np.empty((0, 2), dtype=int)
    arr = np.empty((len(entries), 2), dtype=int)
    for i, e in enumerate(entries):
        arr[i, 0] = int(e.patches)
        arr[i, 1] = int(e.faces)
    return arr

def map_boundaries_h_to_h_sbdv(HsbdvPrnt: Sequence,
                               HsbdvBndrs: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``mapBoundariesHtoHsbdv.m``.

    Returns ``(idxs, dvsr)`` as 1-D int arrays of length ``len(HsbdvBndrs)``.
    """
    n_bndrs = len(HsbdvBndrs)
    if n_bndrs == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    pf_ids = _pf_array(HsbdvBndrs)
    # unique(pfIDs, 'rows') + inverse indices (MATLAB 1-based).
    unique_pf, idxs_0based = np.unique(pf_ids, axis=0, return_inverse=True)

    # acc(jj) = number of rows in pfIDs equal to unique_pf[jj]
    acc = np.bincount(idxs_0based, minlength=unique_pf.shape[0]).astype(np.int64)

    # dvsr(k) = acc(idxs(k))   -- MATLAB 1-based indexing.
    idxs_1based = (idxs_0based + 1).astype(np.int64)
    dvsr = acc[idxs_0based].astype(np.int64)

    return idxs_1based, dvsr
