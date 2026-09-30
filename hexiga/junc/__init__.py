"""
hexiga.junc
===========

Python port of the MATLAB ``junc_utils/`` package: the quad-face-split (QFS)
machinery that builds the NURBS hexahedral patches around multi-tube
junctions (junctions, sockets, stents, T-pipes, ...).

Pipeline (top-level)
--------------------

* :func:`genDirTubes` -- generate N random unit tube directions.
* :func:`regulariseDirTubes` -- spread directions away from each other.
* :func:`getBaseQForkSimplex` -- find the initial (Cis) quad-face-split
  candidate and base IVX for a given direction set.
* :func:`splitClusterDirTubes` -- split the Cis cluster into Cis + Trans.
* :func:`updtIdxBasePartition` -- refine the Cis/Trans partition (base).
* :func:`updtIdxSbdvdPartition` -- refine the partition after sub-divisions.
* :func:`sdvBaseQForkSimplex` -- iteratively sub-divide the Cis quad until a
  valid partition is found (the main loop).
* :func:`getExtrudeBevelQFS` -- extrude/bevel the QFS quad faces along
  per-tube directions.
* :func:`makeQFSs2nrbHexa` -- build the internal NURBS hexahedral patches.
* :func:`makeQFSs2nrbWallHexa` -- build the wall (+cap) hexahedral patches.

Lower-level geometry helpers
----------------------------

* :func:`padQFSpt`, :func:`splitQFSpt` -- cyclic padding / midpoint splits.
* :func:`proj`, :func:`uvect`, :func:`orthog` -- vector primitives.
* :func:`getCentroid` -- centroid of a polygon.
* :func:`getConcaVe` -- concavity sign of a quad face.
* :func:`detSign` -- determinant sign of a 3x3 matrix.
* :func:`getIVX`, :func:`getAdjIVX` -- index vectors & adjacency.
* :func:`isPtInTriangle`, :func:`isPtIntersectingSubSpace` -- point tests.
* :func:`intersectLinePlane` -- line/plane intersection.
* :func:`getIntersectingPointFor2Segments` -- 3D segment intersection.
* :func:`getLbls2split`, :func:`mapSplitIndices` -- sub-divide index maps.
* :func:`getCisTransQFSs`, :func:`getCisTransDiagonals`,
  :func:`mapCisTransQFSs` -- Cis/Trans bookkeeping.
* :func:`sbdvPoly2Tris` -- sub-divide a polygon into triangles.
* :func:`sortDir3Tubes` -- order tubes by pairwise angles.
* :func:`clstrDirs` -- cluster directions by distance threshold.
"""

from __future__ import annotations

from .pad_qfspt import padQFSpt
from .split_qfspt import splitQFSpt
from .proj import proj
from .uvect import uvect
from .orthog import orthog
from .get_intersecting_point_for2_segments import getIntersectingPointFor2Segments
from .get_extrude_bevel_qfs import getExtrudeBevelQFS
from .make_qfs2nrb_hexa import makeQFSs2nrbHexa
from .make_qfs2nrb_wall_hexa import makeQFSs2nrbWallHexa

# --- J1 (leaf / mid) ---
from .det_sign import detSign
from .get_centroid import getCentroid
from .get_concave import getConcaVe
from .get_ivx import getIVX
from .is_pt_in_triangle import isPtInTriangle
from .intersect_line_plane import intersectLinePlane
from .sbdv_poly2_tris import sbdvPoly2Tris
from .is_pt_intersecting_subspace import isPtIntersectingSubSpace
from .sort_dir3_tubes import sortDir3Tubes
from .gen_dir_tubes import genDirTubes
from .regularise_dir_tubes import regulariseDirTubes
from .get_adj_ivx import getAdjIVX
from .get_lbls2split import getLbls2split
from .map_split_indices import mapSplitIndices
from .get_cis_trans_qfs import getCisTransQFSs
from .get_cis_trans_diagonals import getCisTransDiagonals
from .map_cis_trans_qfs import mapCisTransQFSs
from .clstr_dirs import clstrDirs

# --- J2 (partition refinement) ---
from .updt_idx_base_partition import updtIdxBasePartition
from .updt_idx_sbdvd_partition import updtIdxSbdvdPartition
from .split_cluster_dir_tubes import splitClusterDirTubes

# --- J3 (subdivision loop) ---
from .get_base_qfork_simplex import getBaseQForkSimplex
from .sdv_base_qfork_simplex import sdvBaseQForkSimplex

__all__ = [
    # Core (pre-existing)
    "padQFSpt",
    "splitQFSpt",
    "proj",
    "uvect",
    "orthog",
    "getIntersectingPointFor2Segments",
    "getExtrudeBevelQFS",
    "makeQFSs2nrbHexa",
    "makeQFSs2nrbWallHexa",
    # J1
    "detSign",
    "getCentroid",
    "getConcaVe",
    "getIVX",
    "isPtInTriangle",
    "intersectLinePlane",
    "sbdvPoly2Tris",
    "isPtIntersectingSubSpace",
    "sortDir3Tubes",
    "genDirTubes",
    "regulariseDirTubes",
    "getAdjIVX",
    "getLbls2split",
    "mapSplitIndices",
    "getCisTransQFSs",
    "getCisTransDiagonals",
    "mapCisTransQFSs",
    "clstrDirs",
    # J2
    "updtIdxBasePartition",
    "updtIdxSbdvdPartition",
    "splitClusterDirTubes",
    # J3
    "getBaseQForkSimplex",
    "sdvBaseQForkSimplex",
]
