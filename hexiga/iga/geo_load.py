"""
hexiga.iga.geo_load
===================

Loads a multipatch NURBS geometry description.

Two public functions:

  * :func:`mp_geo_read_nurbs` — reads a ``.txt`` "nurbs mesh v2.x" file
    and returns ``[geometry, boundaries, interfaces, subdomains]`` where
    ``geometry`` is a list of :class:`hexiga.nurbs.Nrb` patches.

  * :func:`mp_geo_load` — top-level dispatcher:

    * ``in`` is a ``str`` path ending in ``.txt`` → ``mp_geo_read_nurbs``
      + shared post-processing (rdim, dnurbs, dnurbs2, map/map_der/map_der2,
      boundary extraction, boundary_interfaces).
    * ``in`` is a list of :class:`Nrb` (B-NURBS struct array) → per-patch
      geo_load + ``nrbmultipatch`` for interfaces/boundaries + single
      subdomain (with a warning "Automatically generating the interface
      and boundary information with nrbmultipatch").
    * ``in`` is a ``str`` path ending in ``.mat`` / ``.xml`` →
      ``NotImplementedError`` (not needed for the HexIGA demos; a future
      extension point).
    * anything else → ``ValueError('mp_geo_load: wrong input type')``.

Post-processing attaches, for every patch ``iptc``:

  * ``rdim`` — 1, 2 or 3, detected from ``abs(coefs(2:4, :)) > 1e-12``.
  * ``dnurbs`` / ``dnurbs2`` — first / second derivatives via
    :func:`hexiga.nurbs.nrbderiv`.
  * ``map`` / ``map_der`` / ``map_der2`` — callables wrapping
    :func:`hexiga.nurbs.nrbeval` / :func:`hexiga.nurbs.nrbdeval`.
  * ``boundary`` — a list of ``2*ndim`` sub-structures (one per side), each
    with its own ``nurbs``, ``dnurbs``, ``dnurbs2``, ``rdim``,
    ``map``/``map_der``/``map_der2`` — only when ``len(order) > 1``.

Finally, if the geometry has boundaries and ``len(order) > 1``,
``boundary_interfaces`` is the output of :func:`hexiga.nurbs.nrbmultipatch`
called on the flattened list of all boundary faces (in the same order as
``boundaries.patches`` / ``boundaries.faces``); otherwise ``[]``.

See :mod:`hexiga.nurbs.io` for the underlying ``.txt`` parser (``nrblead``).
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..nurbs.nrb import Nrb
from ..nurbs.io import nrblead
from ..nurbs.eval import nrbeval
from ..nurbs.deval import nrbdeval
from ..nurbs.deriv import nrbderiv
from ..nurbs.extract import nrbextract
from ..nurbs.multipatch import nrbmultipatch

__all__ = ["mp_geo_read_nurbs", "mp_geo_load"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _detect_rdim(patches: Sequence[Nrb]) -> int:
    """Faithful port of the ``rdim`` detection loop in ``mp_geo_load.m``.

    For every patch:
      if   any(|coefs(3, :)| > 1e-12): rdim = 3
      elif any(|coefs(2, :)| > 1e-12): rdim = max(rdim, 2)
      else:                            rdim = max(rdim, 1)

    (MATLAB uses 1-based row indexing; our ``coefs`` array uses the same
    convention: row 0 = x, row 1 = y, row 2 = z, row 3 = weight.)
    """
    rdim = 0
    for p in patches:
        c = np.asarray(p.coefs)
        if c.shape[0] >= 3 and np.any(np.abs(c[2, ...]) > 1e-12):
            rdim = 3
        elif c.shape[0] >= 2 and np.any(np.abs(c[1, ...]) > 1e-12):
            rdim = max(rdim, 2)
        else:
            rdim = max(rdim, 1)
    return rdim

def _make_maps(nurbs: Nrb, dnurbs: List[Nrb], dnurbs2: List[Nrb], rdim: int):
    """Return ``(map, map_der, map_der2)`` callables wrapping nrbeval / nrbdeval.

    MATLAB:  ``map(PTS) = geo_nurbs (nurbs, deriv, deriv2, PTS, 0, rdim)``.
    We expose the same three levels through our already-ported
    :func:`hexiga.nurbs.nrbeval` (value) and :func:`hexiga.nurbs.nrbdeval`
    (value + 1st/2nd derivatives).
    """
    def _map(pts: np.ndarray) -> np.ndarray:
        pnt, _ = nrbeval(nurbs, pts, homogeneous=True)
        return pnt[:rdim, ...]

    def _map_der(pts: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        pnt, jac, _ = nrbdeval(nurbs, dnurbs, None, pts)
        return (pnt[:rdim, ...], [j[:rdim, ...] for j in jac])

    def _map_der2(pts: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], Any]:
        # Pass ``None`` when there are no second derivatives (e.g. boundary
        # curves), matching MATLAB's geo_nurbs behaviour of returning no
        # Hessian in that case.
        d2 = dnurbs2 if dnurbs2 else None
        pnt, jac, hess = nrbdeval(nurbs, dnurbs, d2, pts)
        hess_r: Any
        if hess is None:
            hess_r = None
        elif isinstance(hess, (list, tuple)) and len(hess) > 0 \
                and isinstance(hess[0], (list, tuple)):
            hess_r = [[h[:rdim, ...] for h in row] for row in hess]
        else:
            hess_r = np.asarray(hess)[:rdim, ...]
        return (pnt[:rdim, ...], [j[:rdim, ...] for j in jac], hess_r)
    return _map, _map_der, _map_der2

def _build_dnurbs2(nurbs: Nrb) -> List[Nrb]:
    """Build the list of second partial derivatives.

    * For a curve (ndim=1): returns ``[second_deriv]`` — a length-1 list
      containing the single second-derivative Nrb, as expected by
      :func:`hexiga.nurbs.nrbdeval` for ndim==1.
    * For a surface/volume (ndim>=2): returns a 2-D list
      ``dnurbs2[i][j] = d^2 nrb / (d t_i d t_j)``, obtained by
      differentiating the ``i``-th first derivative with respect to
      direction ``j``.

    A second derivative along an axis requires original degree >= 2
    (order >= 3).  If any axis has degree < 2 we return the empty list
    (the Hessian is unavailable for a linear entity).
    """
    if any(deg < 2 for deg in nurbs.degree):
        return []
    dnurbs = nrbderiv(nurbs)
    ndim = nurbs.ndim
    if ndim == 1:
        di = nrbderiv(dnurbs[0])
        return [di[0]]
    out: List[List[Nrb]] = [[None] * ndim for _ in range(ndim)]
    for i in range(ndim):
        di = nrbderiv(dnurbs[i])
        for j in range(ndim):
            out[i][j] = di[j]
    return out

def _boundary_entry(bnurbs: Nrb, rdim: int) -> dict:
    """One ``boundary(ibnd)`` sub-structure (MATLAB layout)."""
    dnurbs = nrbderiv(bnurbs)
    dnurbs2 = _build_dnurbs2(bnurbs) if len(bnurbs.order) > 1 else []
    m, md, md2 = _make_maps(bnurbs, dnurbs, dnurbs2, rdim)
    return {
        'nurbs':    bnurbs,
        'dnurbs':   dnurbs,
        'dnurbs2':  dnurbs2,
        'rdim':     rdim,
        'map':      m,
        'map_der':  md,
        'map_der2': md2,
    }

def _build_patch_dict(nurbs: Nrb, rdim: int) -> dict:
    """Assemble one ``geometry(iptc)`` dict in the MATLAB layout.

    Keys (MATLAB → Python name mapping, kept identical for faithfulness):
      ``nurbs``, ``rdim``, ``dnurbs``, ``dnurbs2``,
      ``map``, ``map_der``, ``map_der2``, ``boundary`` (optional).
    """
    dnurbs = nrbderiv(nurbs)
    # MATLAB ground truth (mp_geo_load.m, lines 162-164): the main patch
    # always gets [deriv, deriv2] = nrbderiv(nurbs) -- there is NO
    # length(order)>1 gate here, so a curve also carries its second
    # derivative (and thus a Hessian via map_der2).
    dnurbs2 = _build_dnurbs2(nurbs)
    d: dict[str, Any] = {'nurbs': nurbs, 'rdim': rdim,
                         'dnurbs': dnurbs, 'dnurbs2': dnurbs2}
    m, md, md2 = _make_maps(nurbs, dnurbs, dnurbs2, rdim)
    d['map']      = m
    d['map_der']  = md
    d['map_der2'] = md2

    if len(nurbs.order) > 1:
        d['boundary'] = [_boundary_entry(b, rdim) for b in nrbextract(nurbs)]
    return d

def _compute_boundary_interfaces(geometry: List[dict],
                                 boundaries: List[dict],
                                 rdim: int) -> List[dict]:
    """Faithful port of the trailing ``boundary_interfaces = nrbmultipatch(bnd_nurbs)``
    block.

    For each entry in ``boundaries``, take
    ``bnd_nurbs(iptc) = geometry(patches(iptc) - 1).boundary[faces(iptc) - 1].nurbs``
    and feed the flat list to :func:`hexiga.nurbs.nrbmultipatch`.
    """
    if not boundaries:
        return []
    if len(geometry[0]['nurbs'].order) <= 1:
        return []

    patch_numbers = np.concatenate([np.atleast_1d(np.asarray(b['patches'])) for b in boundaries])
    side_numbers  = np.concatenate([np.atleast_1d(np.asarray(b['faces']))   for b in boundaries])

    bnd_nurbs: List[Nrb] = []
    for iptc in range(len(patch_numbers)):
        p = int(patch_numbers[iptc]) - 1   # 1-based → 0-based
        s = int(side_numbers[iptc])   - 1   # 1-based → 0-based
        bnd_nurbs.append(geometry[p]['boundary'][s]['nurbs'])

    interfaces, _ = nrbmultipatch(bnd_nurbs)
    return interfaces

# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def mp_geo_read_nurbs(filename: str
                      ) -> Tuple[List[Nrb], List[dict], List[dict], List[dict]]:
    """Read a ``.txt`` "nurbs mesh v2.x" file.

      The actual parsing is
    delegated to :func:`hexiga.nurbs.io.nrblead`; the return order here
    matches MATLAB: ``(geometry, boundaries, interfaces, subdomains)``.

    Parameters
    ----------
    filename : str
        Absolute or relative path to the ``.txt`` file.

    Returns
    -------
    geometry   : list[Nrb]
    boundaries : list[dict]
    interfaces : list[dict]
    subdomains : list[dict]
    """
    patches, interfaces, subdomains, boundaries = nrblead(filename)
    return patches, boundaries, interfaces, subdomains

def mp_geo_load(in_obj: Union[str, Sequence[Nrb]]
                ) -> Tuple[List[dict], List[dict], List[dict], List[dict], List[dict]]:
    """Create a multipatch geometry structure from a file or a list of Nrb.

      Returns

    ``(geometry, boundaries, interfaces, subdomains, boundary_interfaces)``

    where ``geometry`` is a list of dicts (one per patch) with keys
    ``nurbs``, ``rdim``, ``dnurbs``, ``dnurbs2``, ``map``, ``map_der``,
    ``map_der2`` and, for ndim > 1, ``boundary`` (list of side dicts).

    Parameters
    ----------
    in_obj : str | Sequence[Nrb]
        Either a ``.txt`` file path, or a list of
        :class:`~hexiga.nurbs.Nrb` objects (the MATLAB "B-NURBS struct
        array" case).
    """
    # ---- 1. dispatch on input type ----------------------------------------
    if isinstance(in_obj, (str, os.PathLike)):
        fname = os.fspath(in_obj)
        ext = os.path.splitext(fname)[1].lower()
        if ext == '.txt':
            patches, boundaries, interfaces, subdomains = mp_geo_read_nurbs(fname)
        elif ext in ('.mat', '.xml'):
            raise NotImplementedError(
                f"mp_geo_load: extension '{ext}' is not needed for the HexIGA "
                f"demos and has not been ported. Use a '.txt' file or pass a "
                f"list of Nrb objects directly.")
        else:
            raise ValueError('mp_geo_load: unknown file extension')
    elif (isinstance(in_obj, (list, tuple)) and len(in_obj) > 0
          and all(isinstance(x, Nrb) for x in in_obj)):
        warnings.warn(
            "Automatically generating the interface and boundary information "
            "with nrbmultipatch", stacklevel=2)
        patches   = list(in_obj)
        interfaces, boundaries = nrbmultipatch(patches)
        subdomains = [{'name': 'SUBDOMAIN 1',
                       'patches': list(range(1, len(patches) + 1))}]
    else:
        raise ValueError('mp_geo_load: wrong input type')

    # ---- 2. shared post-processing (rdim, derivs, maps, boundary) ---------
    rdim = _detect_rdim(patches)
    geometry: List[dict] = [_build_patch_dict(p, rdim) for p in patches]

    # ---- 3. boundary_interfaces ------------------------------------------
    boundary_interfaces = _compute_boundary_interfaces(geometry, boundaries, rdim)

    return geometry, boundaries, interfaces, subdomains, boundary_interfaces
