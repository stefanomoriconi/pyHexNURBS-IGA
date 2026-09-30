"""
hexiga.junc.det_sign
====================

Determine the correct sign
for ONE SIDE of the QFS (simplex).

``flip`` stands as a memory of previous flips: at most ONE flip is allowed per
QFS.

Signature::

    [m, flip] = detSign(A, B, flip)   % A: 3x1, B: 3x1, flip: 1x1 logical
"""

from __future__ import annotations

import numpy as np

from .proj import proj

__all__ = ["detSign"]

def detSign(A: np.ndarray, B: np.ndarray, flip: bool):
    """Determine the correct sign for ONE SIDE of the QFS (simplex)."""
    m = 1
    if (proj(A, B) > 0) and (not flip):
        m = -1
        flip = True
    return m, flip
