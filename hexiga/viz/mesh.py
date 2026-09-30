"""
hexiga.viz.mesh
===============

Shared mesh-building helpers for the HexIGA Python visualisation module.

These routines turn a :class:`hexiga.nurbs.Nrb` (or a list of them) into
``pyvista`` meshes by sampling the NURBS with :func:`hexiga.nurbs.nrbeval`
and, where only the *exterior* boundary is wanted, selecting the free
boundary faces with :func:`hexiga.nurbs.nrbmultipatch` +
:func:`hexiga.nurbs.nrbextract`.

This mirrors the MATLAB ``plot_utils`` helpers (``nrbelmplotMP``,
``nrbghostplotMP``, ``getMPnrbExtBoundary``, ``nrbsidesplot`` ...) without
porting each MATLAB plotting function verbatim -- the Python port builds the
meshes directly on the (already ported) NURBS kernel and renders them with
pyVista.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv

from ..nurbs import Nrb, nrbextract, nrbeval, nrbmultipatch

__all__ = [
    "surface_mesh",
    "curve_mesh",
    "control_points",
    "exterior_boundary_meshes",
    "all_side_meshes",
    "SIDE_COLORS",
]

# Side colours taken from MATLAB ``nrbsidesplot.m`` (each row is a side;
# the 4th channel is an alpha of 225/255).  Sides are 1-based (MATLAB order):
#   1: U=0   2: U=1   3: V=0   4: V=1   5: W=0   6: W=1
_SIDE_RGBA = np.array(
    [
        [83, 0, 0, 225],
        [255, 45, 45, 255],
        [0, 92, 0, 255],
        [25, 255, 45, 255],
        [0, 0, 56, 255],
        [0, 45, 255, 255],
    ],
    dtype=float,
) / 255.0

# Public side colours (RGB 0-1).  Index ``i`` corresponds to side ``i+1``.
SIDE_COLORS = [_SIDE_RGBA[i, 0:3].tolist() for i in range(6)]

def _param_axis(knots: np.ndarray, nsub: int) -> np.ndarray:
    """Sample the parametric range of one axis.

    Uses the knot span endpoints plus a uniform refinement so that the
    sampled grid follows the geometric feature density of the knot vector.
    """
    k = np.asarray(knots, dtype=float)
    lo, hi = k[0], k[-1]
    if hi <= lo:
        return np.array([lo])
    return np.linspace(lo, hi, int(nsub))

def _eval_grid(nrb: Nrb, nsub: int) -> List[np.ndarray]:
    """Return a list of 1-D parametric samples, one per axis."""
    axes = []
    for a in range(nrb.ndim):
        k = nrb.knots[a]
        axes.append(_param_axis(k, nsub))
    return axes

def _colour_mesh(mesh: pv.PolyData, rgb: Sequence[float], alpha: float = 1.0):
    """Attach a uniform per-point RGB colour to ``mesh`` (in place)."""
    n = mesh.n_points
    mesh["rgba"] = np.tile(np.array(rgb, dtype=float) + [alpha], (n, 1))
    return mesh

def surface_mesh(
    nrb: Nrb,
    nsub: int = 48,
    color: Optional[Sequence[float]] = None,
    alpha: float = 1.0,
) -> pv.PolyData:
    """Build a pyVista surface mesh for a 2-D NURBS surface.

    Parameters
    ----------
    nrb
        A NURBS surface (``ndim == 2``).
    nsub
        Number of samples along *each* parametric axis.
    color
        Optional uniform RGB colour (0-1 floats) applied to every point.
    alpha
        Alpha (0-1) applied together with ``color``.
    """
    if nrb.ndim != 2:
        raise ValueError("surface_mesh expects a 2-D NURBS surface")
    xyz = nrbeval(nrb, _eval_grid(nrb, nsub), homogeneous=False)  # (3, nu, nv)
    pts = np.ascontiguousarray(xyz.transpose(1, 2, 0))           # (nu, nv, 3)
    mesh = _build_quad_grid(pts)
    if color is not None:
        _colour_mesh(mesh, color, alpha)
    return mesh

def _build_quad_grid(pts: np.ndarray) -> pv.PolyData:
    """Build a ``pv.PolyData`` of quads from a 2-D grid of 3-D points.

    ``pts`` has shape ``(nu, nv, 3)``; the returned mesh has
    ``(nu-1) * (nv-1)`` quad cells (degenerate to triangles if the
    grid is 1-D in any direction).
    """
    nu, nv = pts.shape[0], pts.shape[1]
    verts = pts.reshape(-1, 3)
    if nu == 1 or nv == 1:
        # Degenerate: emit line cells instead.
        n = max(nu, nv)
        cells = [[2, i, i + 1] for i in range(n - 1)]
    else:
        cells = []
        for i in range(nu - 1):
            for j in range(nv - 1):
                a = i * nv + j
                b = (i + 1) * nv + j
                c = (i + 1) * nv + j + 1
                d = i * nv + j + 1
                cells.append([4, a, b, c, d])
    return pv.PolyData(verts, cells)

def curve_mesh(
    nrb: Nrb,
    nsub: int = 200,
    color: Optional[Sequence[float]] = None,
) -> pv.PolyData:
    """Build a pyVista line mesh for a 1-D NURBS curve."""
    if nrb.ndim != 1:
        raise ValueError("curve_mesh expects a 1-D NURBS curve")
    xyz = nrbeval(nrb, [_param_axis(nrb.knots[0], nsub)], homogeneous=False)  # (3, N)
    pts = np.ascontiguousarray(xyz.T)  # (N, 3)
    n = len(pts)
    lines = np.concatenate([[n], np.arange(n)])
    mesh = pv.PolyData(pts, lines=lines)
    if color is not None:
        _colour_mesh(mesh, color)
    return mesh

def control_points(nrb: Nrb) -> pv.PolyData:
    """Return the (de-homogenised) Cartesian control points of ``nrb``."""
    c = np.asarray(nrb.coefs, dtype=float)  # (4, *number)
    cart = c[0:3, ...] / c[3:4, ...]
    pts = cart.reshape(3, -1).T  # (N, 3)
    return pv.PolyData(pts)

def exterior_boundary_meshes(
    mpHexa: Sequence[Nrb],
    nsub: int = 48,
) -> List[pv.PolyData]:
    """Return the exterior boundary surface meshes of a multi-patch solid.

    Mirrors MATLAB ``getMPnrbExtBoundary``: uses :func:`nrbmultipatch` to
    detect the free (unshared) faces, then :func:`nrbextract` to obtain each
    face as a NURBS surface, and finally samples each with
    :func:`surface_mesh`.  Internal (shared) faces are *not* returned.
    """
    mpHexa = list(mpHexa)
    if len(mpHexa) == 0:
        return []

    # Single patch: just return its boundary faces directly.
    if len(mpHexa) == 1:
        if mpHexa[0].ndim == 2:
            return [surface_mesh(mpHexa[0], nsub)]
        faces = nrbextract(mpHexa[0])
        if isinstance(faces, Nrb):
            faces = [faces]
        return [surface_mesh(f, nsub) for f in faces]

    _, boundary = nrbmultipatch(mpHexa)
    meshes: List[pv.PolyData] = []
    for entry in boundary:
        patch_i = int(entry["patches"]) - 1
        face_i = int(entry["faces"]) - 1
        patch = mpHexa[patch_i]
        if patch.ndim == 2:
            meshes.append(surface_mesh(patch, nsub))
        else:
            face = nrbextract(patch, sides=[face_i + 1])[0]
            meshes.append(surface_mesh(face, nsub))
    return meshes

def all_side_meshes(
    nrb_vol: Nrb,
    nsub: int = 48,
    color: Optional[Sequence[float]] = None,
    alpha: float = 1.0,
) -> List[pv.PolyData]:
    """Return the (colourable) side surface meshes of one patch.

    For a volume (``ndim == 3``) returns all 6 boundary faces, each uniformly
    coloured (side colour table if ``color`` is None).  For a surface returns
    the single surface.  Used by the "ghost" plotting paths.
    """
    if nrb_vol.ndim == 2:
        col = color if color is not None else (0.85, 0.85, 0.85)
        return [surface_mesh(nrb_vol, nsub, color=col, alpha=alpha)]
    faces = nrbextract(nrb_vol)
    if isinstance(faces, Nrb):
        faces = [faces]
    out: List[pv.PolyData] = []
    for i, f in enumerate(faces):
        col = color if color is not None else SIDE_COLORS[i % 6]
        out.append(surface_mesh(f, nsub, color=col, alpha=alpha))
    return out
