"""
hexiga.export.stl
=================

Export a NURBS surface -- or the surface of a NURBS solid / multi-patch
-- to a ``.stl`` file (binary by default, ASCII on request), preserving
the standard NaN-border handling and facet / normal conventions.

MATLAB signature (verbatim)::

    nrbExportSTL(nurbs, 'OPT1', val1, 'OPT2', val2, ...)

with options:

* ``'FILENAME'``     -- output base name (default ``'srf'``; ``.stl`` is
  appended unless already present).
* ``'FOLDERPATH'``   -- output folder (default ``pwd``; must end in ``'/'``).
* ``'SUBDIVISIONS'`` -- sampling factor (default ``3``; ``0`` -> ``5``).
* ``'MPINTERFACES'`` -- for a multi-patch, export the internal interfaces
  (default ``false``).
* ``'MPBOUNDARIES'`` -- for a multi-patch, export the external boundaries
  (default ``true``).
* ``'VERBOSE'``      -- print progress (default ``false``).

Local helpers in the MATLAB source, ported 1:1:

* ``getInputs``            -> :func:`_get_inputs`
* ``getScalarPatchSurfs``  -> :func:`_get_scalar_patch_surfs`
* ``getMultiPatchSurfs``   -> :func:`_get_multi_patch_surfs`
* ``evalSurface``          -> :func:`_eval_surface`
* ``getXYZcatFromSrfs``    -> :func:`_get_xyz_cat_from_srfs`
* ``surf2stl``             -> :func:`surf2stl`
* ``local_write_facet``    -> :func:`_write_facet`
* ``local_find_normal``    -> :func:`_find_normal`
* ``checkFlipVol``         -> :func:`_check_flip_vol`
"""

from __future__ import annotations

import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..nurbs import Nrb, nrbeval, nrbderiv, nrbdeval, nrbextract, nrbmultipatch

__all__ = ["nrb_export_stl", "surf2stl"]

# --------------------------------------------------------------------------
# Option parsing
# --------------------------------------------------------------------------
def _get_inputs(nurbs, varargs: Sequence) -> Dict:
    """Port of the local ``getInputs`` helper.

    Parses a MATLAB-style ``('KEY', value, ...)``, argument list into a
    plain dict.  Accepted keys (case-insensitive, matching MATLAB):
    ``FILENAME``, ``FOLDERPATH``, ``SUBDIVISIONS``, ``MPINTERFACES``,
    ``MPBOUNDARIES``, ``VERBOSE``.
    """
    opts: Dict = {
        "FILENAME": "srf",
        "FOLDERPATH": os.getcwd() + os.sep,
        "SUBDIVISIONS": 3,
        "MPINTERFACES": False,
        "MPBOUNDARIES": True,
        "VERBOSE": False,
    }

    if len(varargs) % 2 != 0:
        raise ValueError(
            "nrbExportSTL: options must be supplied as (NAME, value) pairs."
        )

    for i in range(0, len(varargs), 2):
        key = str(varargs[i]).upper()
        val = varargs[i + 1]
        if key == "FILENAME":
            opts["FILENAME"] = str(val)
        elif key == "FOLDERPATH":
            opts["FOLDERPATH"] = str(val)
        elif key == "SUBDIVISIONS":
            opts["SUBDIVISIONS"] = int(val)
        elif key == "MPINTERFACES":
            opts["MPINTERFACES"] = bool(val)
        elif key == "MPBOUNDARIES":
            opts["MPBOUNDARIES"] = bool(val)
        elif key == "VERBOSE":
            opts["VERBOSE"] = bool(val)
        else:
            raise ValueError(f"nrbExportSTL: unknown option '{varargs[i]}'.")

    # MATLAB getInputs: FOLDERPATH must end with a path separator.
    fp = opts["FOLDERPATH"]
    if not fp.endswith(os.sep):
        opts["FOLDERPATH"] = fp + os.sep

    # MATLAB getInputs: append '.stl' if not already present (case-insens).
    fn = opts["FILENAME"]
    if len(fn) < 4 or fn[-4:].lower() != ".stl":
        opts["FILENAME"] = fn + ".stl"

    # MATLAB: SUBDIVISIONS of 0 is reinterpreted as 5 with a disp.
    if opts["SUBDIVISIONS"] == 0:
        opts["SUBDIVISIONS"] = 5
        print(
            " <!> nrbExportSTL: SUBDIVISIONS = 0 detected - "
            "Using SUBDIVISIONS = 5 instead."
        )

    return opts

# --------------------------------------------------------------------------
# Volume-orientation helper (MATLAB checkFlipVol)
# --------------------------------------------------------------------------
def _check_flip_vol(nurbs: Nrb) -> bool:
    """Port of the local ``checkFlipVol`` helper.

    MATLAB::

        [pnt, jac] = nrbdeval(nurbs, nrbderiv(nurbs), [0;0;0]);
        flipVol = sign(det([jac{1}, jac{2}, jac{3}])) < 0;

    i.e. evaluate the first partials at the origin and inspect the sign of
    the Jacobian determinant.  Returns ``True`` when the parametric
    orientation is "inverted" (negative volume), which drives which set of
    sides gets its surface flipped so that the exported STL has outward
    normals.
    """
    dnurbs = nrbderiv(nurbs)
    ndim = nurbs.ndim
    # Parametric point {0,0,...,0} -- MATLAB uses the column vector [0;0;0],
    # i.e. a single scattered point of shape (ndim, 1).
    tt = np.zeros((ndim, 1))
    _, jac, _ = nrbdeval(nurbs, dnurbs, tt=tt)
    # jac is a list of ndim arrays, each (3,1) for a single point.  Stack.
    J = np.column_stack([np.asarray(j).reshape(3) for j in jac])
    det = np.linalg.det(J)
    return bool(np.sign(det) < 0)

# --------------------------------------------------------------------------
# Surface extraction
# --------------------------------------------------------------------------
def _eval_surface(nurbs: Nrb, subdivisions: int, flipflag: bool) -> Dict[str, np.ndarray]:
    """Port of the local ``evalSurface`` helper.

    MATLAB::

        Usbdv = subdivisions * nurbs.number(1);
        Vsbdv = subdivisions * nurbs.number(2);
        Ugrid = linspace(nurbs.knots{1}(order(1)),
                         nurbs.knots{1}(end - order(1) + 1), Usbdv);
        Vgrid = linspace(nurbs.knots{2}(order(2)),
                         nurbs.knots{2}(end - order(2) + 1), Vsbdv);
        if flipflag, Ugrid = Ugrid(end:-1:1); end
        pts = nrbeval(nurbs, {Ugrid, Vgrid});
        srfs.X = padarray(squeeze(pts(1,:)), [1 1], NaN);
        srfs.Y = padarray(squeeze(pts(2,:)), [1 1], NaN);
        srfs.Z = padarray(squeeze(pts(3,:)), [1 1], NaN);

    Python (0-based indexing, numpy column-major -> row-major):

    * ``knots{1}(order(1))``            -> ``k[order-1]``
    * ``knots{1}(end-order(1)+1)``      -> ``k[-(order-1)-1]``
    * ``flipflag``  -> ``Ugrid = Ugrid[::-1]``
    * ``padarray(X, [1 1], NaN)``       -> build a ``(nu+2, nv+2)`` NaN array
      and place the evaluated ``nu x nv`` block at ``[1:-1, 1:-1]``.
    """
    number = nurbs.number
    order = nurbs.order
    knots = nurbs.knots

    Usbdv = int(subdivisions * number[0])
    Vsbdv = int(subdivisions * number[1])

    k1 = knots[0]
    k2 = knots[1]
    Ugrid = np.linspace(k1[order[0] - 1], k1[-(order[0] - 1) - 1], Usbdv)
    Vgrid = np.linspace(k2[order[1] - 1], k2[-(order[1] - 1) - 1], Vsbdv)

    if flipflag:
        Ugrid = Ugrid[::-1]

    cp, cw = nrbeval(nurbs, [Ugrid, Vgrid], homogeneous=True)
    pts = cp / cw  # Cartesian (3, Usbdv, Vsbdv)

    # MATLAB padarray(squeeze(pts(i,:)), [1 1], NaN):
    # the interior is the (Usbdv x Vsbdv) block, surrounded by a 1-wide NaN
    # border -> shape (Usbdv+2, Vsbdv+2).
    def _pad(x: np.ndarray) -> np.ndarray:
        out = np.full((x.shape[0] + 2, x.shape[1] + 2), np.nan)
        out[1:-1, 1:-1] = x
        return out

    return {
        "X": _pad(np.asarray(pts[0], dtype=float)),
        "Y": _pad(np.asarray(pts[1], dtype=float)),
        "Z": _pad(np.asarray(pts[2], dtype=float)),
    }

def _get_scalar_patch_surfs(
    nurbs: Nrb,
    opts: Dict,
    side_ids: Optional[Sequence[int]] = None,
    flip_vol: Optional[bool] = None,
) -> List[Dict[str, np.ndarray]]:
    """Port of the local ``getScalarPatchSurfs`` helper.

    MATLAB logic (verbatim):

    * solid (``length(nurbs.knots) > 2``): extract each requested side
      surface (``nrbextract(nurbs, sideIDs)``) and flip ``Ugrid`` depending
      on the side and the volume orientation::

          if flipVol
              flipflag = ~ismember(side, [2 3 6]);
          else
              flipflag = ~ismember(side, [1 4 5]);
          end

    * surface (``length(nurbs.knots) == 2``): export the whole surface,
      no flip.
    * curve: print a warning and return an empty list.
    """
    ndim = nurbs.ndim
    subdivisions = opts["SUBDIVISIONS"]

    if ndim == 1:
        print(
            " <!> nrbExportSTL: The Input NURBS entity is a 1D curve - "
            "STL export requires at least a 2D surface!"
        )
        return []

    if ndim == 2:
        # Surface: whole surface, no flip.
        return [_eval_surface(nurbs, subdivisions, flipflag=False)]

    # ndim == 3 : solid.
    if side_ids is None:
        side_ids = list(range(1, 7))
    if flip_vol is None:
        flip_vol = _check_flip_vol(nurbs)

    surfs: List[Dict[str, np.ndarray]] = []
    for side in side_ids:
        side = int(side)
        if flip_vol:
            flipflag = side not in (2, 3, 6)
        else:
            flipflag = side not in (1, 4, 5)
        side_surfs = nrbextract(nurbs, [side])
        for s in side_surfs:
            surfs.append(_eval_surface(s, subdivisions, flipflag))
    return surfs

def _get_multi_patch_surfs(nurbs: List[Nrb], opts: Dict) -> List[Dict[str, np.ndarray]]:
    """Port of the local ``getMultiPatchSurfs`` helper.

    MATLAB logic (verbatim):

    * ``MPINTERFACES and MPBOUNDARIES``  -> all six sides of every patch.
    * ``MPINTERFACES only``              -> the internal interface surfaces
      of each interface (``patch1/side1`` + ``patch2/side2``).
    * ``MPBOUNDARIES only``              -> the external boundary surfaces
      grouped per patch (``boundaries(:).patches`` / ``.faces``).
    """
    if not isinstance(nurbs, (list, tuple)):
        # Single patch: treat as a scalar patch.
        return _get_scalar_patch_surfs(nurbs, opts)

    mpinterfaces = opts["MPINTERFACES"]
    mpboundaries = opts["MPBOUNDARIES"]

    if not (mpinterfaces or mpboundaries):
        raise ValueError(
            "nrbExportSTL: for a multi-patch, enable MPINTERFACES and/or "
            "MPBOUNDARIES."
        )

    # Resolve the interface / boundary side structure once.
    interfaces, boundary = nrbmultipatch(list(nurbs))

    surfs: List[Dict[str, np.ndarray]] = []

    # --- External boundaries (grouped per patch, MATLAB getMultiPatchSurfs)
    if mpboundaries:
        # Group boundary entries by patch id (1-based).
        per_patch: Dict[int, List[int]] = {}
        for b in boundary:
            pid = int(b["patches"])
            fid = int(b["faces"])
            per_patch.setdefault(pid, []).append(fid)
        for pid in sorted(per_patch):
            patch = nurbs[pid - 1]
            flip_vol = (
                _check_flip_vol(patch) if patch.ndim == 3 else None
            )
            surfs.extend(
                _get_scalar_patch_surfs(
                    patch, opts, per_patch[pid], flip_vol
                )
            )

    # --- Internal interfaces (both sides of each interface)
    if mpinterfaces:
        for itf in interfaces:
            p1 = int(itf["patch1"])
            s1 = int(itf["side1"])
            p2 = int(itf["patch2"])
            s2 = int(itf["side2"])
            for pid, sid in ((p1, s1), (p2, s2)):
                patch = nurbs[pid - 1]
                flip_vol = (
                    _check_flip_vol(patch) if patch.ndim == 3 else None
                )
                surfs.extend(
                    _get_scalar_patch_surfs(patch, opts, [sid], flip_vol)
                )

    if not surfs:
        print(
            "nrbExportSTL: no interface or boundary sides found for the "
            "multi-patch - nothing to export."
        )
    return surfs

# --------------------------------------------------------------------------
# Concatenation helper
# --------------------------------------------------------------------------
def _get_xyz_cat_from_srfs(
    srfs: List[Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of the local ``getXYZcatFromSrfs`` helper.

    MATLAB (verbatim, ``export_utils/nrbExportSTL.m``)::

        Xcat = []; Ycat = []; Zcat = [];
        sizes = zeros(length(srfs),2);
        for ss = 1:length(srfs)
            sizes(ss,:) = size(srfs(ss).X);
        end
        uSizes  = unique(sizes(:));
        chkSizes = false(size(uSizes));
        for uS = 1:length(uSizes)
           chkSizes(uS) = all(any(ismember(sizes,uSizes(uS)),2),1);
        end
        cmmSize = uSizes(chkSizes);
        for ss = 1:length(srfs)
            if isequal( size(srfs(ss).X,1) , cmmSize )
                Xcat = cat(2,Xcat,srfs(ss).X);   ...
            else
                Xcat = cat(2,Xcat,flipud( transpose( srfs(ss).X ) )); ...
            end
        end

    i.e. find the single common dimension ``cmmSize`` shared by every
    surface (the row extent, when all surfaces are row-aligned).  A surface
    whose *first* dimension equals ``cmmSize`` is concatenated as-is along
    the columns; any surface oriented the other way is transposed +
    row-flipped first (``flipud(transpose(X))`` == ``X.T[::-1, :]``).
    All pieces are concatenated along the columns (numpy ``axis=1``).
    """
    if len(srfs) == 1:
        return srfs[0]["X"], srfs[0]["Y"], srfs[0]["Z"]

    def _orient(key: str) -> list:
        pieces: list = []
        for s in srfs:
            arr = np.asarray(s[key])
            if arr.shape[0] == cmm_size:
                pieces.append(arr)
            else:
                pieces.append(arr.T[::-1, :])
        return pieces

    # --- common-dimension detection (MATLAB unique / ismember logic) -------
    # sizes(ss, :) = [rows, cols]
    sizes = np.array([[s["X"].shape[0], s["X"].shape[1]] for s in srfs])
    unique_vals = np.unique(sizes.ravel())
    # A value v is "common" if EVERY surface has v in its {rows, cols}:
    #   chk(v) = all( any( sizes == v, axis=1 ) )   (MATLAB all(any(ismember(sizes,v),2),1))
    chk = np.array(
        [np.all(np.any(sizes == v, axis=1)) for v in unique_vals], dtype=bool
    )
    cmm_candidates = unique_vals[chk]
    cmm_size = int(cmm_candidates[0]) if cmm_candidates.size else None

    if cmm_size is None:
        # No shared dimension: fall back to the orientation-preserving
        # concatenation of the first surface's orientation.
        cmm_size = srfs[0]["X"].shape[0]

    Xcat = np.concatenate(_orient("X"), axis=1)
    Ycat = np.concatenate(_orient("Y"), axis=1)
    Zcat = np.concatenate(_orient("Z"), axis=1)
    return Xcat, Ycat, Zcat

# --------------------------------------------------------------------------
# STL writers (binary + ASCII)
# --------------------------------------------------------------------------
def _find_normal(p1, p2, p3) -> np.ndarray:
    """Port of the local ``local_find_normal`` helper.

    MATLAB::

        v1 = p2 - p1;
        v2 = p3 - p1;
        n  = cross(v1, v2);
        n  = n / norm(n);
    """
    v1 = np.asarray(p2) - np.asarray(p1)
    v2 = np.asarray(p3) - np.asarray(p1)
    n = np.cross(v1, v2)
    nn = np.linalg.norm(n)
    if nn > 0.0:
        n = n / nn
    return n

def _write_facet_binary(f, p1, p2, p3) -> int:
    """Port of ``local_write_facet`` (binary branch).

    Binary STL facet layout (50 bytes):

    * 3 x float32   -- normal
    * 9 x float32   -- 3 vertices x 3 components
    * 1 x uint16    -- attribute byte count (always 0)

    Returns the number of bytes written (always 50 on success).
    """
    n = _find_normal(p1, p2, p3)
    pack = struct.pack(
        "<12fH",
        float(n[0]), float(n[1]), float(n[2]),
        float(p1[0]), float(p1[1]), float(p1[2]),
        float(p2[0]), float(p2[1]), float(p2[2]),
        float(p3[0]), float(p3[1]), float(p3[2]),
        0,
    )
    f.write(pack)
    return len(pack)

def _write_facet_ascii(f, p1, p2, p3) -> None:
    """Port of ``local_write_facet`` (ASCII branch)."""
    n = _find_normal(p1, p2, p3)
    f.write("    facet normal % .7E % .7E % .7E\n" % (n[0], n[1], n[2]))
    f.write("      outer loop\n")
    f.write("        vertex % .7E % .7E % .7E\n" % (p1[0], p1[1], p1[2]))
    f.write("        vertex % .7E % .7E % .7E\n" % (p2[0], p2[1], p2[2]))
    f.write("        vertex % .7E % .7E % .7E\n" % (p3[0], p3[1], p3[2]))
    f.write("      endloop\n")
    f.write("    endfacet\n")

def _is_nan_row(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, i: int, j: int
) -> bool:
    """Check whether the (i, j) grid sample carries a NaN in any channel."""
    return (
        np.isnan(x[i, j]) or np.isnan(y[i, j]) or np.isnan(z[i, j])
    )

def surf2stl(
    filename: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mode: str = "binary",
) -> int:
    """Port of the local ``surf2stl`` helper.

    Rasterises a NURBS surface sampled on a regular ``(nu+2, nv+2)`` grid
    (with a 1-wide NaN border) into a triangulated STL file.  Two triangles
    per interior grid cell; any facet that touches a NaN sample is skipped.

    Parameters
    ----------
    filename
        Output file path (with extension).
    x, y, z
        3-D arrays of shape ``(3, nu+2, nv+2)`` where row ``0`` / ``-1``
        and column ``0`` / ``-1`` are the NaN border.
    mode
        ``'binary'`` (default) or ``'ascii'``.
    """
    mode = mode.lower()
    if mode not in ("binary", "ascii"):
        raise ValueError("surf2stl: mode must be 'binary' or 'ascii'.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if x.ndim != 2:
        raise ValueError("surf2stl: x must be a 2-D matrix.")
    if not (x.shape == y.shape == z.shape):
        raise ValueError("surf2stl: x, y, z must all share the same shape.")

    nu, nv = x.shape
    ncells_i = max(0, nu - 1)
    ncells_j = max(0, nv - 1)

    def _ok(i: int, j: int) -> bool:
        return not _is_nan_row(x, y, z, i, j)

    if mode == "binary":
        # Count valid facets first (header needs the count).
        count = 0
        for i in range(ncells_i):
            for j in range(ncells_j):
                if _ok(i, j) and _ok(i, j + 1) and _ok(i + 1, j + 1):
                    count += 1
                if _ok(i + 1, j + 1) and _ok(i + 1, j) and _ok(i, j):
                    count += 1

        # MATLAB: 80-byte title + uint32 facet count + facets.
        with open(filename, "wb") as f:
            f.write(b" " * 80)
            f.write(struct.pack("<I", count))
            for i in range(ncells_i):
                for j in range(ncells_j):
                    if _ok(i, j) and _ok(i, j + 1) and _ok(i + 1, j + 1):
                        p1 = np.array([x[i, j],     y[i, j],     z[i, j]])
                        p2 = np.array([x[i, j + 1], y[i, j + 1], z[i, j + 1]])
                        p3 = np.array([x[i + 1, j + 1], y[i + 1, j + 1], z[i + 1, j + 1]])
                        _write_facet_binary(f, p1, p2, p3)
                    if _ok(i + 1, j + 1) and _ok(i + 1, j) and _ok(i, j):
                        p1 = np.array([x[i + 1, j + 1], y[i + 1, j + 1], z[i + 1, j + 1]])
                        p2 = np.array([x[i + 1, j],     y[i + 1, j],     z[i + 1, j]])
                        p3 = np.array([x[i, j],         y[i, j],         z[i, j]])
                        _write_facet_binary(f, p1, p2, p3)
        return count
    else:
        with open(filename, "w", newline="") as f:
            f.write("solid nrbExportSTL\n")
            for i in range(ncells_i):
                for j in range(ncells_j):
                    if _ok(i, j) and _ok(i, j + 1) and _ok(i + 1, j + 1):
                        p1 = np.array([x[i, j],     y[i, j],     z[i, j]])
                        p2 = np.array([x[i, j + 1], y[i, j + 1], z[i, j + 1]])
                        p3 = np.array([x[i + 1, j + 1], y[i + 1, j + 1], z[i + 1, j + 1]])
                        _write_facet_ascii(f, p1, p2, p3)
                    if _ok(i + 1, j + 1) and _ok(i + 1, j) and _ok(i, j):
                        p1 = np.array([x[i + 1, j + 1], y[i + 1, j + 1], z[i + 1, j + 1]])
                        p2 = np.array([x[i + 1, j],     y[i + 1, j],     z[i + 1, j]])
                        p3 = np.array([x[i, j],         y[i, j],         z[i, j]])
                        _write_facet_ascii(f, p1, p2, p3)
            f.write("endsolid nrbExportSTL\n")
        return -1  # ASCII: no binary facet count; return -1 as a sentinel.

# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def nrb_export_stl(
    nurbs: Union[Nrb, List[Nrb]],
    *varargs,
) -> Dict[str, str]:
    """Port of ``nrbExportSTL.m``.

    Export a NURBS surface (or the surface of a NURBS solid / multi-patch)
    to a binary or ASCII ``.stl`` file.

    Parameters
    ----------
    nurbs
        A :class:`Nrb` (curve / surface / volume) *or* a list of :class:`Nrb`
        patches (multi-patch).
    *varargs
        MATLAB-style ``(NAME, value, ...)`` option pairs.  See the module
        docstring for the accepted names and defaults.

    Returns
    -------
    dict
        ``{'path': <full output file path>}`` so callers can locate the
        result programmatically.
    """
    opts = _get_inputs(nurbs, varargs)
    if opts["VERBOSE"]:
        print(f"nrbExportSTL: writing '{opts['FILENAME']}' ...")

    if isinstance(nurbs, (list, tuple)):
        surfs = _get_multi_patch_surfs(list(nurbs), opts)
    else:
        surfs = _get_scalar_patch_surfs(nurbs, opts)

    if not surfs:
        print("nrbExportSTL: nothing to export (empty surface list).")
        return {"path": ""}

    X, Y, Z = _get_xyz_cat_from_srfs(surfs)
    out_path = opts["FOLDERPATH"] + opts["FILENAME"]
    surf2stl(out_path, X, Y, Z, mode="binary")

    if opts["VERBOSE"]:
        print(f"nrbExportSTL: done -> {out_path}")
    return {"path": out_path}
