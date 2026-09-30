"""
io.py — File I/O for NURBS multipatch meshes (v2.1 format).

Provides:
  - ``nrblead`` — reads a ``.txt`` multipatch mesh file.
  - ``nrbsave`` — round-trip writer (inverse of the reader).

File format (NURBS multipatch mesh v2.1):

  # comments
  # comments
  ndim rdim npatch nintrfc nsubd
  PATCH 1
  deg_u deg_v [deg_w]
  n_u n_v [n_w]
  knot vector line 1
  knot vector line 2
  [knot vector line 3]
  coef row 1   (x-coords, all CPs, row-major)
  coef row 2   (y)
  coef row 3   (z)
  coef row 4   (weights)
  PATCH 2
  ...
  INTERFACE 1
  p1 s1
  p2 s2
  ornt          (2D)  or  flag ornt1 ornt2  (3D)
  ...
  SUBDOMAIN 1
  p1 p2 ...
  ...
  BOUNDARY 1
  nsides
  patch1 face1
  ...
"""
from __future__ import annotations
import os
from typing import Optional, Sequence

import numpy as np

from .nrb import Nrb

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _next_data_line(fid) -> str:
    """Skip comment lines starting with '#', return the next non-empty line.

    Faithful to MATLAB's ``line = fgetl(fid); while line(1) == '#' ...``
    loop.
    """
    while True:
        line = fid.readline()
        if line == '':
            raise EOFError("unexpected end of file in .txt NURBS mesh")
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        return line

def _parse_line(line: str) -> np.ndarray:
    return np.array([float(t) for t in line.split()], dtype=float)

# ---------------------------------------------------------------------------
# main reader
# ---------------------------------------------------------------------------
def nrblead(filename: str):
    """Read a NURBS multipatch mesh file (v2.1 format).

    Parameters
    ----------
    filename : str
        Path to the ``.txt`` file.

    Returns
    -------
    patches : list[Nrb]
        The NURBS patches.
    interfaces : list[dict]
        Each dict has keys ``'ref'``, ``'patch1'``, ``'side1'``,
        ``'patch2'``, ``'side2'``, and (2D) ``'ornt'`` or (3D)
        ``'flag'``, ``'ornt1'``, ``'ornt2'``.
    subdomains : list[dict]
        Each dict has keys ``'name'`` and ``'patches'``.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    with open(filename, 'r', encoding='utf-8') as fid:
        # ---- header ----
        line = _next_data_line(fid)
        vec = _parse_line(line)
        ndim   = int(vec[0])
        rdim   = int(vec[1]) if len(vec) > 1 else ndim
        npatch = int(vec[2]) if len(vec) > 2 else 1
        if len(vec) == 4:
            nintrfc = int(vec[3])
            nsubd   = 0
        elif len(vec) == 5:
            nintrfc = int(vec[3])
            nsubd   = int(vec[4])
        else:
            nintrfc = 0
            nsubd   = 0

        # ---- patches ----
        patches: list[Nrb] = []
        for iptc in range(npatch):
            # Optional PATCH label (non-numeric line)
            line = _next_data_line(fid)
            try:
                _parse_line(line)
            except ValueError:
                line = _next_data_line(fid)  # skip the label, read degree

            deg = _parse_line(line)
            order = (deg + 1).astype(int)

            line = _next_data_line(fid)
            number = _parse_line(line).astype(int)

            knots = tuple(
                _parse_line(_next_data_line(fid)) for _ in range(ndim)
            )

            # rdim+1 = 4 coefs rows (x, y, z, w)
            coefs = np.zeros((rdim + 1, *number))
            for idim in range(rdim + 1):
                line = _next_data_line(fid)
                vals = _parse_line(line)
                coefs[idim, ...] = vals.reshape(number, order='F')

            patches.append(Nrb(
                number=tuple(int(n) for n in number),
                order=tuple(int(o) for o in order),
                knots=knots,
                coefs=coefs,
            ))

        # ---- interfaces (faithful to mp_geo_read_nurbs.m) ----
        #   ref = line        (always the label line, e.g. "INTERFACE 1")
        #   patch1 side1
        #   patch2 side2
        #   ornt           (2D)   or   flag ornt1 ornt2   (3D)
        interfaces: list[dict] = []
        for intrfc in range(nintrfc):
            ref = _next_data_line(fid)
            v1  = _parse_line(_next_data_line(fid))
            v2  = _parse_line(_next_data_line(fid))
            d: dict = {
                'ref':    ref,
                'patch1': int(v1[0]), 'side1': int(v1[1]),
                'patch2': int(v2[0]), 'side2': int(v2[1]),
            }
            v3 = _parse_line(_next_data_line(fid))
            if ndim == 2:
                d['ornt'] = int(v3[0])
            elif ndim == 3:
                d['flag']  = int(v3[0])
                d['ornt1'] = int(v3[1])
                d['ornt2'] = int(v3[2])
            interfaces.append(d)

        # ---- subdomains (faithful to mp_geo_read_nurbs.m) ----
        #   name
        #   p1 p2 ...
        subdomains: list[dict] = []
        for isub in range(nsubd):
            name = _next_data_line(fid)
            line = _next_data_line(fid)
            subdomains.append({
                'name': name,
                'patches': _parse_line(line).astype(int).tolist(),
            })

        # ---- boundaries (optional, loop-until-EOF — faithful to MATLAB) ----
        #   while (line ~= -1)
        #     boundaries(bnd).name = line
        #     nsides ...
        #     ...
        #   end
        boundaries: list[dict] = []
        while True:
            try:
                bname = _next_data_line(fid)
            except EOFError:
                break
            nsides = int(_parse_line(_next_data_line(fid))[0])
            b: dict = {'name': bname, 'nsides': nsides,
                       'patches': [], 'faces': []}
            for _ in range(nsides):
                pv = _parse_line(_next_data_line(fid))
                b['patches'].append(int(pv[0]))
                b['faces'].append(int(pv[1]))
            boundaries.append(b)

    return patches, interfaces, subdomains, boundaries

def _is_numeric(line: str) -> bool:
    try:
        _parse_line(line)
        return True
    except ValueError:
        return False

# ---------------------------------------------------------------------------
# writer (round-trip)
# ---------------------------------------------------------------------------
def nrbsave(filename: str,
            patches: Sequence[Nrb],
            interfaces: Optional[Sequence[dict]] = None,
            subdomains: Optional[Sequence[dict]] = None,
            boundaries: Optional[Sequence[dict]] = None,
            header_comments: Optional[Sequence[str]] = None):
    """Write a NURBS multipatch mesh to a v2.1 format ``.txt`` file.

    Inverse of ``nrblead``.  Produces a file that can be read back
    by both ``nrblead`` and MATLAB's ``mp_geo_read_nurbs``.
    """
    if not patches:
        raise ValueError("patches list is empty")

    p0 = patches[0]
    ndim = len(p0.number)
    rdim = p0.coefs.shape[0] - 1
    npatch = len(patches)
    nintrfc = len(interfaces) if interfaces else 0
    nsubd = len(subdomains) if subdomains else 0

    with open(filename, 'w', encoding='utf-8') as fid:
        if header_comments:
            for c in header_comments:
                fid.write(f"# {c}\n")
        fid.write("#\n")
        fid.write(f"{ndim} {rdim} {npatch} {nintrfc} {nsubd}\n")

        for iptc, nrb in enumerate(patches, 1):
            fid.write(f"PATCH {iptc}\n")
            fid.write(" ".join(str(o - 1) for o in nrb.order) + "\n")
            fid.write(" ".join(str(n) for n in nrb.number) + "\n")
            for k in nrb.knots:
                fid.write(" ".join(repr(float(x)) for x in k) + "\n")
            for irow in range(rdim + 1):
                row = nrb.coefs[irow, ...].reshape(-1, order='F')
                fid.write(" ".join(repr(float(x)) for x in row) + "\n")

        if interfaces:
            for intrfc, iface in enumerate(interfaces, 1):
                fid.write(f"INTERFACE {intrfc}\n")
                fid.write(f"{iface['patch1']} {iface['side1']}\n")
                fid.write(f"{iface['patch2']} {iface['side2']}\n")
                if ndim == 2:
                    fid.write(f"{iface['ornt']}\n")
                elif ndim == 3:
                    fid.write(f"{iface['flag']} {iface['ornt1']} {iface['ornt2']}\n")

        if subdomains:
            for sd in subdomains:
                fid.write(f"{sd['name']}\n")
                fid.write(" ".join(str(p) for p in sd['patches']) + "\n")

        if boundaries:
            for bnd in boundaries:
                fid.write(f"{bnd['name']}\n")
                fid.write(f"{bnd['nsides']}\n")
                for p, f in zip(bnd['patches'], bnd['faces']):
                    fid.write(f"{p} {f}\n")
