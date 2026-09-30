"""
hexiga.viz.visualise_individual_qfs
====================================

Renders the individual QFS (Quasi-Fork-Simplex) branches of a multi-branch
junction: direction quivers, QFS point quads, extruded-bevel quads, and
optionally the B-Spline loop arcs of the base fork simplex.

Signature::

    visualiseIndividualQFSs(dirTubes, QFSfcs, QFSpts, idx, varargin)

Returns a populated ``pyvista.Plotter`` (headless-safe by default).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pyvista as pv

from ..junc import uvect
from ..nurbs import bspinterpcrv, nrbeval
from .plots import _base_plotter, _finalize

__all__ = ["visualise_individual_qfs"]

# ---------------------------------------------------------------------------
# Colour helper (MATLAB hsv(n) equivalent)
# ---------------------------------------------------------------------------

def _hsv_colors(n: int) -> np.ndarray:
    """Return *n* evenly spaced HSV colours (S=1, V=1), matching MATLAB hsv(n)."""
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    if n == 1:
        return np.array([[1.0, 0.0, 0.0]])
    hues = np.linspace(0.0, 1.0, n, endpoint=False)
    rgb = np.zeros((n, 3), dtype=float)
    for i, h in enumerate(hues):
        X = 1.0 - abs((h * 6.0) % 2.0 - 1.0)
        h6 = h * 6.0
        if h6 < 1:   rgb[i] = [1.0, X, 0.0]
        elif h6 < 2: rgb[i] = [X, 1.0, 0.0]
        elif h6 < 3: rgb[i] = [0.0, 1.0, X]
        elif h6 < 4: rgb[i] = [0.0, X, 1.0]
        elif h6 < 5: rgb[i] = [X, 0.0, 1.0]
        else:        rgb[i] = [1.0, 0.0, X]
    return rgb

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _add_line(plotter, p1, p2, color, line_width=2.0, opacity=1.0):
    """Add a single line segment to *plotter*."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    mesh = pv.PolyData(np.array([p1, p2]), lines=np.array([2, 0, 1]))
    c = color if isinstance(color, str) else tuple(np.asarray(color, dtype=float)[:3])
    plotter.add_mesh(mesh, color=c, line_width=line_width,
                     opacity=opacity, show_scalar_bar=False)

def _add_quad_outline(plotter, verts, color, line_width=2.0, opacity=1.0):
    """Add a closed quad outline (4 vertices, shape (4,3)) to *plotter*."""
    verts = np.asarray(verts, dtype=float)
    # Closed loop: 0 → 1 → 2 → 3 → 0  (5 elements = 4 segments)
    lines = np.array([5, 0, 1, 2, 3, 0])
    mesh = pv.PolyData(verts, lines=lines)
    c = color if isinstance(color, str) else tuple(np.asarray(color, dtype=float)[:3])
    plotter.add_mesh(mesh, color=c, line_width=line_width,
                     opacity=opacity, show_scalar_bar=False)

def _add_quiver(plotter, start, direction, scale, color, line_width=2.0):
    """Add a quiver arrow (line + tip marker) to *plotter*."""
    start = np.asarray(start, dtype=float)
    direction = np.asarray(direction, dtype=float)
    end = start + scale * direction
    _add_line(plotter, start, end, color, line_width)
    tip = pv.PolyData(np.array([end]))
    c = color if isinstance(color, str) else tuple(np.asarray(color, dtype=float)[:3])
    plotter.add_mesh(tip, color=c, point_size=4.0, render_points_as_spheres=True)

def _add_points(plotter, pts, color, point_size=8.0):
    """Add point markers to *plotter*."""
    pts = np.asarray(pts, dtype=float)
    mesh = pv.PolyData(pts)
    c = color if isinstance(color, str) else tuple(np.asarray(color, dtype=float)[:3])
    plotter.add_mesh(mesh, color=c, point_size=point_size,
                     render_points_as_spheres=True)

def _add_curve(plotter, pts, color, line_width=2.0, opacity=1.0):
    """Add a polyline (curve) to *plotter*."""
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n < 2:
        return
    lines = np.concatenate([[n], np.arange(n)])
    mesh = pv.PolyData(pts, lines=lines)
    c = color if isinstance(color, str) else tuple(np.asarray(color, dtype=float)[:3])
    plotter.add_mesh(mesh, color=c, line_width=line_width,
                     opacity=opacity, show_scalar_bar=False)

# ---------------------------------------------------------------------------
# getLoopArcs port (local function in visualiseIndividualQFSs.m)
# ---------------------------------------------------------------------------

def _get_loop_arcs(
    QFSfcs: np.ndarray,   # (4, f), 1-based
    QFSpts: np.ndarray,   # (3, P), Cartesian (already scaled)
    idx: np.ndarray,      # (N,), 1-based
) -> List[Dict]:
    """Port of ``getLoopArcs`` (local function in ``visualiseIndividualQFSs.m``).

    Computes B-Spline loop arcs for the base fork simplex. Each arc connects
    two adjacent QFS points along loopA (nodes 0-1-2) or loopB (nodes 2-3-0).
    Arcs shared between branches are averaged.

    Parameters
    ----------
    QFSfcs : (4, f) int array, 1-based
    QFSpts : (3, P) Cartesian
    idx : (N,) int array, 1-based

    Returns
    -------
    list of dict
        Each dict has keys:
        - ``pts``: (3, 50) Cartesian points along the arc
        - ``nodes``: (2,) int array of 1-based QFSpts indices
        - ``idx``: (K,) int array of 1-based branch indices sharing this arc
    """
    arcs: List[Dict] = []

    # 0-based index pairs into QFSfc for each of the 4 sub-arcs
    #   arc 0: loopA first half  → QFSfc[0], QFSfc[1]
    #   arc 1: loopA second half → QFSfc[1], QFSfc[2]
    #   arc 2: loopB first half  → QFSfc[2], QFSfc[3]
    #   arc 3: loopB second half → QFSfc[3], QFSfc[0]
    arc_pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)

    for jj in range(len(idx)):
        QFSfc = QFSfcs[:, idx[jj] - 1]          # (4,) 1-based QFSpts indices
        QFSpt = QFSpts[:, QFSfc - 1]             # (3, 4) Cartesian

        if len(arcs) == 0:
            # First branch: create 4 arcs
            loopA, _ = bspinterpcrv(QFSpt[:, [0, 1, 2]], 2, "equally_spaced")
            loopB, _ = bspinterpcrv(QFSpt[:, [2, 3, 0]], 2, "equally_spaced")

            specs = [
                (loopA, np.linspace(0,   0.5, 50), 0),
                (loopA, np.linspace(0.5, 1,   50), 1),
                (loopB, np.linspace(0,   0.5, 50), 2),
                (loopB, np.linspace(0.5, 1,   50), 3),
            ]
            for loop, t_range, aa in specs:
                pts = np.asarray(
                    nrbeval(loop, t_range, homogeneous=False), dtype=float
                )
                arcs.append({
                    "pts": pts,
                    "nodes": QFSfc[arc_pairs[aa]].astype(int),
                    "idx": np.array([idx[jj]], dtype=int),
                })
        else:
            # Subsequent branches: check / merge / create arcs
            loopA, _ = bspinterpcrv(QFSpt[:, [0, 1, 2]], 2, "equally_spaced")
            loopB, _ = bspinterpcrv(QFSpt[:, [2, 3, 0]], 2, "equally_spaced")
            loops = [loopA, loopA, loopB, loopB]
            t_ranges = [
                np.linspace(0,   0.5, 50),
                np.linspace(0.5, 1,   50),
                np.linspace(0,   0.5, 50),
                np.linspace(0.5, 1,   50),
            ]

            for aa in range(4):
                candidate_nodes = QFSfc[arc_pairs[aa]]  # (2,) 1-based
                c1, c2 = int(candidate_nodes[0]), int(candidate_nodes[1])

                # Find matching existing arc (order-independent)
                match_idx = None
                flip_flag = False
                for i, arc in enumerate(arcs):
                    n1, n2 = int(arc["nodes"][0]), int(arc["nodes"][1])
                    if n1 == c1 and n2 == c2:
                        match_idx, flip_flag = i, False
                        break
                    elif n1 == c2 and n2 == c1:
                        match_idx, flip_flag = i, True
                        break

                new_pts = np.asarray(
                    nrbeval(loops[aa], t_ranges[aa], homogeneous=False),
                    dtype=float,
                )

                if match_idx is not None:
                    existing = arcs[match_idx]
                    if flip_flag:
                        new_pts = new_pts[:, ::-1]
                    existing["pts"] = (existing["pts"] + new_pts) / 2.0
                    existing["idx"] = np.unique(
                        np.concatenate([
                            existing["idx"].astype(int),
                            np.array([idx[jj]], dtype=int),
                        ])
                    )
                else:
                    arcs.append({
                        "pts": new_pts,
                        "nodes": candidate_nodes.astype(int),
                        "idx": np.array([idx[jj]], dtype=int),
                    })

    return arcs

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def visualise_individual_qfs(
    dirTubes: np.ndarray,
    QFSfcs: np.ndarray,
    QFSpts: np.ndarray,
    idx: np.ndarray,
    **opts: Any,
) -> pv.Plotter:
    """

    Parameters
    ----------
    dirTubes : (3, N) array
        Direction tubes (one per branch).
    QFSfcs : (4, f) int array, 1-based
        QFS point indices per branch.
    QFSpts : (3, P) array
        Cartesian QFS points.
    idx : (N,) int array, 1-based
        Cluster index per branch.
    **opts
        MATLAB-style keyword options (case-insensitive):

        ================  ==============  ========================================
        Option            Default         Description
        ================  ==============  ========================================
        EBQFSFCS          None            Extruded-bevel QFSfcs (4×N, 1-based)
        EBQFSPTS          None            Extruded-bevel QFSpts (3×M)
        KAE               None            KAe list of dicts (vpt, e1, e2)
        LBLCIS            None            CIS labels (unused in render)
        JPT               [0, 0, 0]       Origin point
        NEWFIG            True            (no-op; always creates new Plotter)
        CRVSIMPLEX        False           Enable B-Spline loop arcs
        SHOWNODES         False           Show QFS point markers
        SCALEFACTOR       1.0             Scale applied to QFSpts
        QUIVERSCALE       1.0             Quiver arrow length scale
        QUIVERLINEWIDTH   2.0             Quiver line width
        QFSCOLORS         None (hsv)      Branch colours (n,3) or (3,)
        ARCLINEWIDTH      2.0             Arc line width
        VDIR              None            Atomic simplex direction (3,)
        PERPDIRS          None            Perpendicular directions (3,K)
        PRINCDIRSFLAG     False           Use principal (cluster-mean) directions
        SHOWARCS          True            Show arcs / QFS quad
        ACCESSORYSCALE    2.0             Scale for vdir / perpdirs
        NODESCOLORS       None            Marker face grayscale values (P,)
        OFFSCREEN         True            Headless rendering
        ================  ==============  ========================================

    Returns
    -------
    pyvista.Plotter
        Populated plotter. Call ``.show()`` for interactive display or
        ``.write_png("out.png")`` for headless rendering.
    """
    # ------------------------------------------------------------------
    # Parse options (MATLAB configureInputs equivalent)
    # ------------------------------------------------------------------
    o = {k.upper(): v for k, v in opts.items()}

    def _opt(key: str, default: Any = None) -> Any:
        return o.get(key, default)

    ebQFSfcs_opt = _opt("EBQFSFCS", None)
    ebQFSpts_opt = _opt("EBQFSPTS", None)
    KAe_opt      = _opt("KAE",      None)
    jpt          = np.asarray(_opt("JPT", [0.0, 0.0, 0.0]), dtype=float).ravel()
    crvSimplex   = bool(_opt("CRVSIMPLEX",   False))
    showNodes    = bool(_opt("SHOWNODES",    False))
    scaleFactor  = abs(float(_opt("SCALEFACTOR",    1.0)))
    quiverScale  = abs(float(_opt("QUIVERSCALE",    1.0)))
    quiverLineWd = abs(float(_opt("QUIVERLINEWIDTH", 2.0)))
    clrs_opt     = _opt("QFSCOLORS",  None)
    arcLineWd    = abs(float(_opt("ARCLINEWIDTH",   2.0)))
    vdir_opt     = _opt("VDIR",       None)
    perpdirs_opt = _opt("PERPDIRS",   None)
    showArcs     = bool(_opt("SHOWARCS",  True))
    princdirsFlg = bool(_opt("PRINCDIRSFLAG", False))
    accScale     = abs(float(_opt("ACCESSORYSCALE", 2.0)))
    mfc_opt      = _opt("NODESCOLORS", None)
    offScreen    = bool(_opt("OFFSCREEN", True))

    # ------------------------------------------------------------------
    # Convert inputs
    # ------------------------------------------------------------------
    dirTubes = np.asarray(dirTubes, dtype=float)
    QFSfcs   = np.asarray(QFSfcs,   dtype=int)
    idx      = np.asarray(idx,      dtype=int).ravel()
    QFSpts   = np.asarray(QFSpts,   dtype=float) * scaleFactor  # MATLAB: QFSpts * scaleFactor

    # ------------------------------------------------------------------
    # Colours
    # ------------------------------------------------------------------
    max_idx = int(np.max(idx))
    if clrs_opt is None or len(np.atleast_1d(clrs_opt)) == 0:
        clrs = _hsv_colors(max_idx)
    else:
        clrs_arr = np.asarray(clrs_opt, dtype=float)
        if clrs_arr.ndim == 1:
            # MATLAB: isvector → repmat to (max_idx, len)
            clrs = np.tile(clrs_arr.reshape(1, -1), (max_idx, 1))
        else:
            clrs = clrs_arr
    # Safety: ensure at least max_idx rows
    if clrs.shape[0] < max_idx:
        clrs = np.vstack([clrs, np.tile(clrs[-1:], (max_idx - clrs.shape[0], 1))])

    # ------------------------------------------------------------------
    # Setup plotter
    # ------------------------------------------------------------------
    plotter = _base_plotter(offScreen)

    # ------------------------------------------------------------------
    # Origin marker
    # ------------------------------------------------------------------
    _add_points(plotter, jpt.reshape(1, 3), "k", point_size=6.0)

    # ------------------------------------------------------------------
    # vdir (atomic simplex direction)
    # ------------------------------------------------------------------
    if vdir_opt is not None and np.asarray(vdir_opt).size > 0:
        vdir = np.asarray(vdir_opt, dtype=float).ravel()
        _add_line(plotter, jpt, jpt + accScale * vdir, "k", 2)
        _add_line(plotter, jpt, jpt - accScale * vdir, "k", 2)

    # ------------------------------------------------------------------
    # perpdirs (perpendicular directions)
    # ------------------------------------------------------------------
    if perpdirs_opt is not None and np.asarray(perpdirs_opt).size > 0:
        perpdirs = np.asarray(perpdirs_opt, dtype=float)
        if perpdirs.ndim == 1:
            perpdirs = perpdirs.reshape(3, 1)
        for pp in range(perpdirs.shape[1]):
            _add_line(plotter, jpt, jpt + accScale * perpdirs[:, pp], "k", 2, opacity=0.5)

    # ------------------------------------------------------------------
    # B-Spline loop arcs (if crvSimplex)
    # QFSpts passed here is already scaled (MATLAB scales before calling getLoopArcs).
    # ------------------------------------------------------------------
    arcs: List[Dict] = []
    if crvSimplex:
        arcs = _get_loop_arcs(QFSfcs, QFSpts, idx)

    # ------------------------------------------------------------------
    # Show QFS nodes
    # ------------------------------------------------------------------
    if showNodes:
        P = QFSpts.shape[1]
        if mfc_opt is not None and np.asarray(mfc_opt).size > 0:
            mfc = np.asarray(mfc_opt, dtype=float).ravel()
            for pp in range(P):
                gray = float(mfc[pp % len(mfc)])
                _add_points(plotter, QFSpts[:, pp:pp+1].T, [gray]*3, point_size=8.0)
        else:
            _add_points(plotter, QFSpts.T, "w", point_size=8.0)

    # ------------------------------------------------------------------
    # Extruded-bevel availability
    # ------------------------------------------------------------------
    has_eb = (
        ebQFSfcs_opt is not None and np.asarray(ebQFSfcs_opt).size > 0
        and ebQFSpts_opt is not None and np.asarray(ebQFSpts_opt).size > 0
    )
    ebQFSfcs_arr = np.asarray(ebQFSfcs_opt, dtype=int) if has_eb else None
    ebQFSpts_arr = np.asarray(ebQFSpts_opt, dtype=float) if has_eb else None

    # ------------------------------------------------------------------
    # Main loop over branches
    # ------------------------------------------------------------------
    for jj in range(len(idx)):
        # Quiver direction
        if princdirsFlg:
            mask = idx == idx[jj]
            prncdir = uvect(np.mean(dirTubes[:, mask], axis=1, keepdims=True)).ravel()
        else:
            prncdir = None
        dir_vec = prncdir if prncdir is not None else dirTubes[:, jj]
        color = clrs[idx[jj] - 1, :3]

        # Draw quiver
        _add_quiver(plotter, jpt, dir_vec, quiverScale, color, quiverLineWd)

        # Extract QFS data for this branch
        QFSfc = QFSfcs[:, idx[jj] - 1]   # (4,) 1-based
        QFSpt = QFSpts[:, QFSfc - 1]     # (3, 4) scaled Cartesian

        # Determine ebQFSpt
        if has_eb:
            ebQFSfc = ebQFSfcs_arr[:, idx[jj] - 1]
            ebQFSpt = ebQFSpts_arr[:, ebQFSfc - 1]
        else:
            ebQFSpt = np.empty((3, 0))

        if ebQFSpt.size == 0:
            # No extruded-bevel data
            if showArcs and not crvSimplex:
                # Draw QFSpt quad with coloured edges (MATLAB: patch FaceAlpha=0)
                _add_quad_outline(plotter, QFSpt.T + jpt, color, arcLineWd)
            elif showArcs and crvSimplex:
                # Draw arc curves for this branch (MATLAB: crvSimplex branch)
                offset_dir = prncdir if prncdir is not None else dir_vec
                offset = 0.03 * offset_dir
                for arc in arcs:
                    if idx[jj] in arc["idx"]:
                        arc_pts = arc["pts"].T + jpt + offset.reshape(1, 3)
                        _add_curve(plotter, arc_pts, color, arcLineWd)
        else:
            # Extruded-bevel data available
            # QFSpt quad (black edges)
            _add_quad_outline(plotter, QFSpt.T + jpt, "k", 2)
            # ebQFSpt quad (coloured, "dashed" → lower opacity)
            _add_quad_outline(plotter, ebQFSpt.T + jpt, color, 2, opacity=0.6)
            # Connecting lines: jpt → QFSpt corners
            for k in range(4):
                _add_line(plotter, jpt, jpt + QFSpt[:, k], "k", 0.5, opacity=0.5)
            # Connecting lines: QFSpt → ebQFSpt corners
            for k in range(4):
                _add_line(plotter, jpt + QFSpt[:, k], jpt + ebQFSpt[:, k], "k", 0.5, opacity=0.5)

        # KAe (if provided)
        if KAe_opt is not None and len(KAe_opt) > jj:
            ka = KAe_opt[jj]
            M  = np.asarray(ka.get("vpt", []), dtype=float).ravel()
            e1 = np.asarray(ka.get("e1",  []), dtype=float).ravel()
            e2 = np.asarray(ka.get("e2",  []), dtype=float).ravel()
            if len(M) == 3 and len(e1) == 3 and len(e2) == 3:
                M_pos = jpt + M
                _add_points(plotter, M_pos.reshape(1, 3), color, point_size=4.0)
                _add_line(plotter, M_pos, M_pos + e1, "m", 1.5)
                _add_line(plotter, M_pos, M_pos + e2, "c", 1.5)

    return _finalize(plotter)
