"""
hexiga.junc.get_lbls2split
==========================

Retrieve the labels
of the sub-groups that must be split (those with more than one member).

Signature::

    lbls2split = getLbls2split(idx)   % idx: 1xN, lbls2split: 1xm (m <= N)
"""

from __future__ import annotations

import numpy as np

__all__ = ["getLbls2split"]

def getLbls2split(idx: np.ndarray) -> np.ndarray:
    """Return the labels in ``idx`` (1 x N) that have more than one member (1 x m)."""
    idx = np.asarray(idx, dtype=int).ravel()

    lbls = np.unique(idx)
    lbls2split = []
    for lbl in lbls:
        if np.sum(idx == lbl) > 1:
            lbls2split.append(lbl)

    return np.asarray(lbls2split, dtype=int)
