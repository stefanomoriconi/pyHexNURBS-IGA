"""Port of ``smthCtrlPtsTopologyAttribs.m``.

Per-point Laplacian-style smoothing using the neighbour lists attached by
``append_smth_atrbs_new``.  Three regimes:

* ``ptShl > 0``      -> shell point; EDGE (2 neighbours) or CRNR (more).
* ``ptShl == 0``     -> medial-locus / exception point.
* ``ptFrm < 0``      -> exception point.

Weights (MATLAB ground truth):

* EDGE:      a = 1/2,  b = 1/4
* CRNR:      a = (v-3)/v, b = 2/v^2, g = 1/v^2   (v = # edge-neighbours)
* EXCEPTION: a = 1/2,  b = 1/(2v)
"""
from __future__ import annotations

import copy
from typing import List

import numpy as np

from .cpt import CPT

def _get_edge_weights():
    return 1 / 2, 1 / 4

def _get_crnr_weights(valence: int):
    return (valence - 3) / valence, 2 / (valence ** 2), 1 / (valence ** 2)

def _get_exception_weights(valence: int):
    return 1 / 2, 1 / (2 * valence)

def _smth_edge(mpCPT, ptID):
    """Port of ``smthEDGE``."""
    L2errtol = 1e-6
    c = mpCPT[ptID - 1]
    pt3D = c.pt3D
    nnIDs = c.nnIDs
    if nnIDs.size:
        pt3DE = np.column_stack([mpCPT[i - 1].pt3D for i in nnIDs])
    else:
        pt3DE = np.zeros((3, 0))
    if pt3DE.size and nnIDs.size == 2:
        alpha, beta = _get_edge_weights()
        pt3Dsmth = alpha * pt3D + beta * np.sum(pt3DE, axis=1)
    else:
        pt3Dsmth = pt3D
    isSmth = bool(np.linalg.norm(pt3D - pt3Dsmth) < L2errtol)
    return pt3Dsmth, isSmth

def _smth_crnr(mpCPT, ptID):
    """Port of ``smthCRNR``."""
    L2errtol = 1e-6
    c = mpCPT[ptID - 1]
    pt3D = c.pt3D
    nnIDs = c.nnIDs
    nnLbls = np.asarray(c.nnLbls, dtype=bool)
    pt3DE = np.column_stack([mpCPT[i - 1].pt3D for i in nnIDs[~nnLbls]]) if np.any(~nnLbls) else np.zeros((3, 0))
    pt3DF = np.column_stack([mpCPT[i - 1].pt3D for i in nnIDs[nnLbls]]) if np.any(nnLbls) else np.zeros((3, 0))
    valence = int(np.sum(~nnLbls))
    if nnLbls.size and int(np.sum(nnLbls)) == int(np.sum(~nnLbls)) and valence > 0:
        alpha, beta, gamma = _get_crnr_weights(valence)
        pt3Dsmth = alpha * pt3D + beta * np.sum(pt3DE, axis=1) + gamma * np.sum(pt3DF, axis=1)
    else:
        pt3Dsmth = pt3D
    isSmth = bool(np.linalg.norm(pt3D - pt3Dsmth) < L2errtol)
    return pt3Dsmth, isSmth

def _smth_exception(mpCPT, ptID):
    """Port of ``smthEXCEPTION``."""
    L2errtol = 1e-6
    c = mpCPT[ptID - 1]
    pt3D = c.pt3D
    nnIDs = c.nnIDs
    nnLbls = np.asarray(c.nnLbls, dtype=bool)
    pt3DE = np.column_stack([mpCPT[i - 1].pt3D for i in nnIDs[~nnLbls]]) if np.any(~nnLbls) else np.zeros((3, 0))
    valence = int(np.sum(~nnLbls))
    if nnLbls.size and bool(np.all(~nnLbls)) and valence > 0:
        alpha, beta = _get_exception_weights(valence)
        pt3Dsmth = alpha * pt3D + beta * np.sum(pt3DE, axis=1)
    else:
        pt3Dsmth = pt3D
    isSmth = bool(np.linalg.norm(pt3D - pt3Dsmth) < L2errtol)
    return pt3Dsmth, isSmth

def smth_ctrl_pts_topology_atrbs(mpCPT: List[CPT]) -> List[CPT]:
    """Port of ``smthCtrlPtsTopologyAttribs``.

    Reads from the *original* ``mpCPT`` and writes into a copied list
    ``mpCPTsmth`` (MATLAB semantics: ``mpCPTsmth = mpCPT`` copies the struct
    array, then each smoothed field is assigned per-point)."""
    mpCPTsmth = copy.deepcopy(mpCPT)
    for pt in range(1, len(mpCPTsmth) + 1):
        c = mpCPTsmth[pt - 1]
        if c.isSmth:
            continue
        if c.ptShl > 0:  # Any Shell-Grid Point
            if bool(np.all(~np.asarray(c.nnLbls, dtype=bool))):  # EDGE-smoothing
                pt3Dsmth, isSmth = _smth_edge(mpCPT, pt)
            else:  # CORNER-smoothing
                pt3Dsmth, isSmth = _smth_crnr(mpCPT, pt)
            mpCPTsmth[pt - 1].pt3D = pt3Dsmth
            mpCPTsmth[pt - 1].isSmth = isSmth
        elif c.ptShl == 0 or c.ptFrm < 0:  # Medial Locus Shell-Grid Point
            pt3Dsmth, isSmth = _smth_exception(mpCPT, pt)
            mpCPTsmth[pt - 1].pt3D = pt3Dsmth
            mpCPTsmth[pt - 1].isSmth = isSmth
    return mpCPTsmth
