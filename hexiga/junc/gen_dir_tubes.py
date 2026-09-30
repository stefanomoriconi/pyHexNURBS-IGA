"""
hexiga.junc.gen_dir_tubes
=========================

Generate ``numTubes``
random unit direction vectors.

Signature::

    dirTubes = genDirTubes(numTubes)   % 3 x numTubes
"""

from __future__ import annotations

import numpy as np

from .uvect import uvect

__all__ = ["genDirTubes"]

def genDirTubes(numTubes: int) -> np.ndarray:
    """Generate ``numTubes`` random unit direction vectors (3 x numTubes)."""
    dirTubes = np.random.randn(3, numTubes)
    for jj in range(numTubes):
        dirTubes[:, jj] = uvect(dirTubes[:, jj])
    return dirTubes
