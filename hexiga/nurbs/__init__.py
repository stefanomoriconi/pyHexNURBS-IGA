"""
hexiga.nurbs
============

Core NURBS / B-spline kernel providing the ``nrb*`` API used by HexIGA.

All entities are stored in *homogeneous* form: a ``(4, *number)`` control
point array whose last row holds the weights.  ``nrbeval`` returns
homogeneous ``(cp, cw)`` by default.
"""

from .nrb import Nrb
from .make import nrbmak
from .eval import nrbeval
from .deriv import nrbderiv
from .deval import nrbdeval
from .degelev import nrbdegelev
from .kntins import nrbkntins
from .extract import nrbextract
from .reverse import nrbreverse
from .transp import nrbtransp
from .multipatch import nrbmultipatch
from .io import nrblead, nrbsave
from .tform import nrbtform
from .permute import nrbpermute
from .glue import nrbglue
from .surf4 import nrb4surf
from .interp import bspinterpcrv

__all__ = [
    "Nrb",
    "nrbmak",
    "nrbeval",
    "nrbderiv",
    "nrbdeval",
    "nrbdegelev",
    "nrbkntins",
    "nrbextract",
    "nrbreverse",
    "nrbtransp",
    "nrbmultipatch",
    "nrblead",
    "nrbsave",
    "nrbtform",
    "nrbpermute",
    "nrbglue",
    "nrb4surf",
    "bspinterpcrv",
]
