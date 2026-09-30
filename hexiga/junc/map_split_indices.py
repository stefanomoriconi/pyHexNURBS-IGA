"""
hexiga.junc.map_split_indices
=============================

Remap the local
split labels (1 / 2) to the new global labels ``(Albl, Blbl)``.

Signature::

    idxSPLITmap = mapSplitIndices(idxSPLIT, Albl, Blbl)   % idxSPLIT: 1xN
"""

from __future__ import annotations

import numpy as np

__all__ = ["mapSplitIndices"]

def mapSplitIndices(idxSPLIT: np.ndarray, Albl: int, Blbl: int) -> np.ndarray:
    """Remap local split labels (1, 2) in ``idxSPLIT`` to global ``(Albl, Blbl)``.

    Both masks must be taken from the *original* ``idxSPLIT`` (as in MATLAB's
    ``idxSPLITmap(idxSPLIT == 1) = Albl`` / ``... (idxSPLIT == 2) = Blbl``).
    Masking on the evolving ``idxSPLITmap`` is wrong when ``Albl`` collides
    with a value that already appears in the array after the first assignment
    (e.g. ``[1, 2]`` with ``Albl=2`` would cascade ``[1, 2] -> [2, 2] ->
    [Blbl, Blbl]`` and collapse both local labels onto one global label).
    """
    src = np.asarray(idxSPLIT, dtype=int)
    idxSPLITmap = np.empty_like(src)
    idxSPLITmap[src == 1] = Albl
    idxSPLITmap[src == 2] = Blbl
    return idxSPLITmap
