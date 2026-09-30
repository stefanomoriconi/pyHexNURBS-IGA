"""
hexiga.viz.plots
================

High-level pyVista rendering routines for HexIGA multi-patch NURBS solids.

Each public function mirrors one of the five visualisation modes of the
MATLAB ``MAIN_gui/DemoSynthCAD.m`` GUIDE app (see that file, lines 624-727):

* :func:`plot_solid_domain`   <- ``plotSolidDomain``   (``nrbelmplotMP`` /
  ``nrbghostplotMP`` + GhostFLAG logic)
* :func:`plot_ctrl_pts`       <- ``plotSolidCtrlPts``  (``nrbghostplotMP``
  PLOTALL + ``nrbctrlcageplotMP``)
* :func:`plot_exploded_param` <- ``plotExplodedParam`` (``nrbXplot`` SIDESPLOT)
* :func:`plot_reflection_lines` <- ``plotReflectionLines`` (``nrbZplot``)
* :func:`plot_trabecular`     <- ``plotTrabecular``    (``nrbcellplotMP``)

Every function returns a populated ``pyvista.Plotter`` so it can be shown
interactively (``plotter.show()``) or rendered off-screen
(``plotter.screenshot(...)`` / ``plotter.write_png(...)``).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pyvista as pv

from ..nurbs import Nrb, nrbextract, nrbkntins, nrbmak, nrbeval, nrbtform
from .mesh import (
    SIDE_COLORS,
    all_side_meshes,
    control_points,
    exterior_boundary_meshes,
    surface_mesh,
)

__all__ = [
    "get_explosion_transforms",
    "plot_solid_domain",
    "plot_ctrl_pts",
    "plot_exploded_param",
    "plot_reflection_lines",
    "plot_trabecular",
]

# Ghost (semi-transparent) fill, mirroring MATLAB nrbghostplotMP greys.
_GHOST_RGB = (0.85, 0.85, 0.85)
_GHOST_ALPHA = 0.15

def get_explosion_transforms(
    mpHexa: Sequence[Nrb], scl: float = 0.75, rad: float = 1.1
) -> tuple[np.ndarray, np.ndarray]:
    """Port of MATLAB ``getExplosionMatrixTransform_new.m``.

    Returns per-patch homogeneous 4x4 transforms ``(M1s, M2s)`` (each
    shape ``(4, 4, n)``) such that ``nrbtform(nrbtform(nrb, M1), M2)``
    explodes patch ``n`` about the global centroid by scale ``scl`` along its
    centroid direction with radial factor ``rad``.
    """
    mpHexa = list(mpHexa)
    n = len(mpHexa)
    Cs = np.stack([np.mean(np.asarray(p.coefs)[0:3].reshape(3, -1), axis=1)
                   for p in mpHexa], axis=1)  # (3, n)
    C = Cs.mean(axis=1, keepdims=True)  # (3, 1)

    M1s = np.stack([
        np.array([
            [1, 0, 0, -Cs[0, c]],
            [0, 1, 0, -Cs[1, c]],
            [0, 0, 1, -Cs[2, c]],
            [0, 0, 0, 1.0],
        ])
        for c in range(n)
    ], axis=2)

    M2s = np.stack([
        np.array([
            [scl, 0,   0,   C[0, 0] + rad * (Cs[0, c] - C[0, 0])],
            [0,   scl, 0,   C[1, 0] + rad * (Cs[1, c] - C[1, 0])],
            [0,   0,   scl, C[2, 0] + rad * (Cs[2, c] - C[2, 0])],
            [0,   0,   0,   1.0],
        ])
        for c in range(n)
    ], axis=2)
    return M1s, M2s

def _base_plotter(off_screen: bool = True) -> pv.Plotter:
    p = pv.Plotter(off_screen=off_screen)
    p.background_color = "white"
    return p

def _remove_scalar_bar(plotter: pv.Plotter) -> None:
    """Best-effort removal of any scalar/color bar on ``plotter``.

    The side-coloured exploded view carries a uniform per-point ``rgba``
    array that some pyvista versions interpret as a scalar field and draw a
    legend for.  The colours are purely decorative (side identity), so any
    scalar bar is redundant.  This is a no-op when no scalar bar exists.
    """
    for meth in ("remove_scalar_bar",):
        fn = getattr(plotter, meth, None)
        if fn is not None:
            try:
                fn()
            except Exception:
                pass
            break

def _finalize(plotter: pv.Plotter) -> pv.Plotter:
    # pyvista >=0.48 renamed enable_lighting -> enable_lightkit.
    for meth in ("enable_lighting", "enable_lightkit"):
        fn = getattr(plotter, meth, None)
        if fn is not None:
            try:
                fn()
            except Exception:
                pass
            break
    try:
        plotter.link_axes()
    except Exception:
        pass
    return plotter

def _add_exterior(plotter: pv.Plotter, mpHexa: Sequence[Nrb], nsub: int = 48):
    for mesh in exterior_boundary_meshes(mpHexa, nsub):
        plotter.add_mesh(
            mesh,
            color=_GHOST_RGB,
            opacity=0.9,
            show_edges=True,
            edge_color="k",
            edge_opacity=0.15,
            smooth_shading=True,
            specular=0.6,
            ambient=0.3,
            diffuse=0.7,
        )

def _add_ghost_all(plotter: pv.Plotter, mpHexa: Sequence[Nrb], nsub: int = 48):
    for p in mpHexa:
        for mesh in all_side_meshes(p, nsub, color=_GHOST_RGB, alpha=_GHOST_ALPHA):
            plotter.add_mesh(
                mesh, color=_GHOST_RGB, opacity=_GHOST_ALPHA,
                show_edges=True, edge_color="k", edge_opacity=0.25,
                smooth_shading=True,
            )

def _indices(indices: Sequence[int]) -> List[int]:
    """Convert a 1-based MATLAB index set (e.g. ``1:2:N``) to 0-based list."""
    return [int(i) - 1 for i in indices]

def plot_solid_domain(
    mpHexa: Sequence[Nrb],
    ghostflag: Optional[Sequence[int]] = None,
    idx_composite: Optional[int] = None,
    nsub: int = 48,
    off_screen: bool = True,
) -> pv.Plotter:
    """Render the solid domain (interior removed, exterior shaded).

    Mirrors ``plotSolidDomain`` in ``DemoSynthCAD.m``:

    * ``ghostflag is None`` (or empty) -> plain exterior shading of the whole
      multi-patch solid.
    * ``ghostflag == [0, ...]``        -> solid part(s) plain, remaining
      part(s) semi-transparent ghost.
    * ``ghostflag == [1, ...]``        -> solid part(s) plain, remaining
      part(s) ghost (all faces).
    * ``ghostflag == [-1, ...]``       -> within a part, odd patches plain and
      even patches ghost (alternating), giving a lumen/wall look.
    """
    mpHexa = list(mpHexa)
    plotter = _base_plotter(off_screen)

    if ghostflag is None or len(list(ghostflag)) == 0:
        _add_exterior(plotter, mpHexa, nsub)
        return _finalize(plotter)

    ghostflag = list(ghostflag)
    idxC = len(mpHexa) if idx_composite is None else int(idx_composite)

    for gg, g in enumerate(ghostflag):
        if gg == 0:
            sel = slice(0, idxC)
        else:
            sel = slice(idxC, len(mpHexa))
        part = [mpHexa[i] for i in range(*sel.indices(len(mpHexa)))]
        part_ids = list(range(*sel.indices(len(mpHexa))))

        if g == 0:
            _add_exterior(plotter, part, nsub)
        elif g == 1:
            _add_ghost_all(plotter, part, nsub)
        else:  # -1: alternate plain / ghost
            plain_ids = part_ids[0::2]
            ghost_ids = part_ids[1::2]
            _add_exterior(plotter, [mpHexa[i] for i in plain_ids], nsub)
            _add_ghost_all(plotter, [mpHexa[i] for i in ghost_ids], nsub)
    return _finalize(plotter)

def plot_ctrl_pts(
    mpHexa: Sequence[Nrb],
    nsub: int = 48,
    off_screen: bool = True,
) -> pv.Plotter:
    """Render ghost fill + control-point cages (mirrors ``plotSolidCtrlPts``)."""
    mpHexa = list(mpHexa)
    plotter = _base_plotter(off_screen)
    _add_ghost_all(plotter, mpHexa, nsub)

    # MATLAB plots cages on the odd HexaIDs (1:2:N) to avoid clutter.
    for i in range(0, len(mpHexa), 2):
        pts = control_points(mpHexa[i])
        plotter.add_mesh(
            pts, color="r", point_size=5.0, render_points_as_spheres=True,
        )
    return _finalize(plotter)

def plot_exploded_param(
    mpHexa: Sequence[Nrb],
    scl: float = 0.75,
    rad: float = 1.1,
    nsub: int = 48,
    off_screen: bool = True,
) -> pv.Plotter:
    """Render the exploded parametric view (mirrors ``nrbXplot SIDESPLOT``).

    Each patch is transformed about the global centroid by the explosion
    transforms and drawn with its 6 colour-coded sides.
    """
    mpHexa = list(mpHexa)
    M1s, M2s = get_explosion_transforms(mpHexa, scl, rad)
    plotter = _base_plotter(off_screen)

    for j, p in enumerate(mpHexa):
        t = nrbtform(nrbtform(p, M1s[:, :, j]), M2s[:, :, j])
        if t.ndim == 2:
            for mesh in all_side_meshes(t, nsub, color=(0.7, 0.7, 0.7), alpha=1.0):
                plotter.add_mesh(mesh, smooth_shading=True, show_edges=True,
                                 edge_color="k", edge_opacity=0.1)
        else:
            for mesh in all_side_meshes(t, nsub):
                plotter.add_mesh(mesh, smooth_shading=True, show_edges=True,
                                 edge_color="k", edge_opacity=0.1)
    # The exploded view encodes side identity with per-face RGB colours
    # (uniform "rgba" point arrays); a scalar/colorbar would be redundant.
    # Explicitly drop any scalar bar (no-op when none is present) so this
    # view never shows a colour legend.
    _remove_scalar_bar(plotter)
    return _finalize(plotter)

def plot_reflection_lines(
    mpHexa: Sequence[Nrb],
    nsub: int = 120,
    off_screen: bool = True,
) -> pv.Plotter:
    """Render exterior boundary with a curvature (zebra/reflection) colouring.

    Mirrors the ``nrbZplot`` reflection-line mode: exterior faces are shaded
    using surface normals so curvature bands are visible.  The full MATLAB
    zebra map (``setZebraColorMap``) is replaced by a per-face normal-based
    colour, which gives a comparable pseudo-iso-curvature visual.
    """
    mpHexa = list(mpHexa)
    plotter = _base_plotter(off_screen)
    for mesh in exterior_boundary_meshes(mpHexa, nsub):
        mesh = mesh.copy()
        mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
        n = np.asarray(mesh.point_data["Normals"])  # (N, 3)
        # Map the vertical normal component to a warm/cold zebra scale.
        t = np.clip(0.5 + 0.5 * n[:, 2], 0.0, 1.0)
        rgb = np.column_stack(
            [
                0.5 + 0.5 * np.cos(t * np.pi * 6.0),
                0.5 + 0.5 * np.cos(t * np.pi * 6.0 - 2.094),
                0.5 + 0.5 * np.cos(t * np.pi * 6.0 - 4.188),
            ]
        )
        mesh["rgb"] = rgb
        plotter.add_mesh(
            mesh, scalars="rgb", rgb=True, smooth_shading=True,
            show_edges=False,
        )
    return _finalize(plotter)

def _slice_hexa_srf(
    hexa: Nrb, dim: int, knt: float,
) -> Optional[Nrb]:
    """Port of MATLAB ``sliceHexaSrfAlongDimKnot`` (smth_utils/extractHexaSlices.m).

    ``dim`` is 1-based: 1=U, 2=V, 3=W.  Inserts the knot ``knt`` enough times
    along ``dim`` so that a replicate row/column/plane of control points
    appears, then squeezes that replicate out to produce the sliced NURBS
    surface (degree preserved along the other two axes).
    """
    d = int(dim) - 1  # 0-based axis
    if d not in (0, 1, 2):
        raise ValueError("dim must be in {1,2,3}")
    errtol = 1e-6

    hord = hexa.order[d]
    hknts = np.asarray(hexa.knots[d], dtype=float)
    nreplic = int(hord - int(np.sum(hknts == knt)))
    if nreplic <= 0:
        # Knot already present with full multiplicity — nothing to insert.
        # (MATLAB's for-loop iterates 0 times in that case.)
        nreplic = 0

    h = hexa
    for _rr in range(nreplic):
        iknots: list[np.ndarray] = [np.array([]) for _ in range(3)]
        iknots[d] = np.array([knt])
        h = nrbkntins(h, iknots)

    # Locate the replicate position along ``dim`` (0-based axis ``d``).
    cfs = h.coefs[:3, ...]
    # diff along axis ``d`` of the coefficient array (axis ``d`` + 1 because
    # axis 0 is the spatial row).
    dcfs = np.diff(cfs, axis=d + 1)
    # L2 norm summed over all other axes (including the 3 spatial rows).
    # MATLAB: squeeze(sum(sum(sum(dcfs.^2,1), axis1, axis2))).
    axes_to_sum = [a for a in range(dcfs.ndim) if a != d + 1]
    dcfs_l2 = dcfs ** 2
    for ax in sorted(axes_to_sum, reverse=True):
        dcfs_l2 = dcfs_l2.sum(axis=ax)
    cfs_idx_arr = np.flatnonzero(dcfs_l2 < errtol)
    if cfs_idx_arr.size == 0:
        return None
    cfs_idx = int(cfs_idx_arr[0])

    # Squeeze the replicate plane/line out, keeping homogeneous coefs (4 rows).
    scfs = np.take(h.coefs, cfs_idx, axis=d + 1)
    # Knots for the resulting surface: the two axes other than ``d``.
    sknts = [hexa.knots[a] for a in range(3) if a != d]
    return nrbmak(scfs, sknts)

def _extract_hexa_slices(
    mpHexa: Sequence[Nrb],
    dims: Sequence[int] = (1, 2, 3),
    knts: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    knots_reduced: bool = False,
) -> List[Nrb]:
    """Port of MATLAB ``extractHexaSlices`` (smth_utils/extractHexaSlices.m).

    Returns a flat list of sliced NURBS surfaces (one per (hexa, dim, knot)
    combination).  ``knts`` is a 3-tuple of per-dimension knot arrays; a
    dimension with an all-NaN knot array triggers the per-hexa ``linspace``
    fallback described in the MATLAB source.
    """
    if knts is None:
        knts = (np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
    knts_red_factor = np.array([4, 4, 10])

    out: List[Nrb] = []
    for hh, hexa in enumerate(mpHexa, start=1):
        if any(np.any(np.isnan(np.asarray(k, dtype=float))) for k in knts):
            # All NaN: derive knots from each hexa's own number().
            if knots_reduced:
                u_smpls = max(4, hexa.number[0] - int(knts_red_factor[0]))
                v_smpls = max(4, hexa.number[1] - int(knts_red_factor[1]))
                w_smpls = max(4, hexa.number[2] - int(knts_red_factor[2]))
            else:
                u_smpls = hexa.number[0]
                v_smpls = hexa.number[1]
                w_smpls = hexa.number[2]
            u_knts = np.linspace(0.0, 1.0, int(u_smpls))
            v_knts = np.linspace(0.0, 1.0, int(v_smpls))
            w_knts = np.linspace(0.0, 1.0, int(w_smpls))
            hexa_knts = (u_knts[1:-1], v_knts[1:-1], w_knts[1:-1])
        else:
            hexa_knts = tuple(
                np.asarray(k, dtype=float).ravel() for k in knts
            )

        if len(dims) != len(hexa_knts):
            raise ValueError(
                "dims and knts must have the same length"
            )
        for d, kk in zip(dims, hexa_knts):
            for kv in kk:
                if kv == 0.0 or kv == 1.0:
                    continue
                sl = _slice_hexa_srf(hexa, int(d), float(kv))
                if sl is not None:
                    out.append(sl)
    return out

def plot_trabecular(
    mpHexa: Sequence[Nrb],
    uknots: Optional[np.ndarray] = None,
    vknots: Optional[np.ndarray] = None,
    wknots: Optional[np.ndarray] = None,
    hexaids: Optional[Sequence[int]] = None,
    ghost_all: bool = True,
    nsub: int = 48,
    off_screen: bool = True,
) -> pv.Plotter:
    """Render a sliced "trabecular" view (mirrors MATLAB ``nrbcellplotMP``).

    For each (optionally selected) patch, all six side faces are drawn
    semi-transparent (``GHOSTALL=true`` in the MATLAB GUI, see
    ``MAIN_gui/DemoSynthCAD.m::plotTrabecular``) together with interior
    slice surfaces at the given U/V/W parameters, exposing the hexa cell
    structure.

    Parameters
    ----------
    mpHexa
        Multi-patch hexa NURBS volume (list of :class:`Nrb`).
    uknots, vknots, wknots
        Per-dimension parametric parameters (1-D arrays) at which to slice.
        Defaults to ``linspace(0,1,5)`` (i.e. {0, .25, .5, .75, 1}) as in the
        MATLAB GUI, with 0 and 1 filtered out so only interior slices are
        drawn.
    hexaids
        Optional 1-based list of patch indices to slice.  The MATLAB GUI
        always passes ``HexaIDs = 1:2:N`` (odd patches).  If ``None``, all
        patches are sliced.
    ghost_all
        If ``True`` (default) all faces of each hexa are drawn ghosted
        (MATLAB ``GHOSTALL=true``); otherwise only the exterior boundary is
        ghosted.
    nsub
        Grid subdivision count for surface evaluation.
    """
    mpHexa = list(mpHexa)
    if uknots is None:
        uknots = np.linspace(0.0, 1.0, 5)
    if vknots is None:
        vknots = np.linspace(0.0, 1.0, 5)
    if wknots is None:
        wknots = np.linspace(0.0, 1.0, 5)

    if hexaids is None:
        sel = list(range(len(mpHexa)))
    else:
        sel = _indices(hexaids)

    knts = (np.asarray(uknots, dtype=float).ravel(),
            np.asarray(vknots, dtype=float).ravel(),
            np.asarray(wknots, dtype=float).ravel())
    dims = (1, 2, 3)

    plotter = _base_plotter(off_screen)

    # Ghost background — mirrors ``nrbghostplotMP(mpHexa,'PLOTALL',true)``.
    if ghost_all:
        _add_ghost_all(plotter, mpHexa, nsub)
    else:
        _add_exterior(plotter, mpHexa, nsub)

    # Slices — mirrors the trailing ``for dd / for ss`` loop in nrbcellplotMP.
    slices = _extract_hexa_slices(mpHexa, dims, knts)
    for sl in slices:
        if sl.ndim != 2:
            continue
        mesh = surface_mesh(sl, nsub=nsub)
        plotter.add_mesh(
            mesh,
            color="black",
            opacity=1.0,
            show_edges=True,
            smooth_shading=True,
        )
    return _finalize(plotter)
