"""

MATLAB ground truth:

* :class:`HexIGASim` GUI class mirrors the GUIDE figure
  ``HexIGASim.fig`` and its callback functions (``HexIGASim.m``).

The license/token gate functions from the MATLAB source
(``chkLicense``, ``chkToken``, ``getUTCtsLive``, ``readToken``, ``chkUTC``,
``chkCHS``) are intentionally OMITTED in the Python port -- the Python
package is distributed unlicensed, per project policy.

The headless core (no GUI required):

* :func:`get_void_geom_data`      -- empty ``GeomDATA`` container
* :func:`get_void_vol_source`     -- zero ``VolumetricSource``
* :func:`get_void_attribs`        -- empty ``SideAttribs``
* :func:`get_void_boundary_handles` -- empty ``SideHandle``
* :func:`get_uv_w_dirichlet`      -- build the 6-entry Dirichlet mask
* :func:`gen_boundaries_attributes` -- per-side Srf / pt3D / n3D / isDirichlet
* :func:`gen_arrow_boundary_handles` -- one ``SideHandle`` per side
* :func:`chk_valid_bc_vol_source` -- BC + volume-source validity check
* :func:`import_geometry`         -- headless ``importGeometryFile`` wrapper
* :func:`run_sim`                 -- convert + (validate) + solve + store

GUI (only when ``tkinter`` is importable)::

    app = HexIGASim(); app.build(); app.mainloop()

The GUI mirrors the MATLAB figure: radio buttons for the simulation type
(PressFlow / ElasticDeform / ElectricField), per-axis Dirichlet checkboxes
(U / V / W), side-specific controls (Side Dirichlet, Flip Dir, Magnitude,
f components, EIG), Load/Save/Export/Reset buttons and an output panel.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

__all__ = [
    "GeomDATA",
    "get_void_geom_data",
    "get_void_vol_source",
    "get_void_attribs",
    "get_void_boundary_handles",
    "get_uv_w_dirichlet",
    "gen_boundaries_attributes",
    "gen_arrow_boundary_handles",
    "chk_valid_bc_vol_source",
    "import_geometry",
    "save_boundaries",
    "import_boundaries",
    "run_sim",
    "HexIGASim",
    "main",
]

# ---------------------------------------------------------------------------
# Data containers (mirrors of the MATLAB GeomDATA struct)
# ---------------------------------------------------------------------------

@dataclass
class GeomDATA:
    """Python mirror of the MATLAB ``GeomDATA`` struct.

    Field names match the MATLAB struct exactly (see
    ``getVoidGeomDATA`` in ``HexIGASim.m``).
    """

    FileName: str = ""
    FilePath: str = ""
    mpHexa: Any = None          # multi-patch NURBS (tuple/list of Nrb)
    bndrs: Any = None           # mp_geo_load boundary interfaces
    bndrsAttrbs: List[Any] = field(default_factory=list)
    bndrsHandles: List[Any] = field(default_factory=list)
    volSource: Any = None       # VolumetricSource dataclass

    # Populated by run_sim
    FluidDATA: Any = None
    ElasticDATA: Any = None
    MaxwellDATA: Any = None

    def is_loaded(self) -> bool:
        """True when a geometry (mpHexa) has been imported."""
        return self.mpHexa is not None and len(self.bndrsAttrbs) > 0

def get_void_geom_data() -> GeomDATA:
    """Port of ``getVoidGeomDATA.m`` -- returns an empty ``GeomDATA``.

    MATLAB:
    ::

        GeomDATA = struct( ...
            'FileName',[],'FilePath',[],'mpHexa',[], ...
            'bndrs',[],'bndrsAttrbs',[],'bndrsHandles',[], ...
            'volSource',[],'FluidDATA',[],'ElasticDATA',[],'MaxwellDATA',[]);
    """
    return GeomDATA()

def get_void_vol_source():
    """Port of ``getVoidVolSource.m`` -- zero volumetric source.

    MATLAB: ``struct('f',0.0,'fx',0.0,'fy',0.0,'fz',0.0)``
    """
    from ..iga import VolumetricSource
    return VolumetricSource(f=0.0, fx=0.0, fy=0.0, fz=0.0)

def get_void_attribs():
    """Port of ``getVoidAttribs.m`` -- empty ``SideAttribs``.

    MATLAB: ``struct('Srf',[],'pt3D',[],'n3D',[],'mag',[],'isDirichlet',[])``
    """
    from ..iga import SideAttribs
    return SideAttribs(Srf=None, pt3D=None, n3D=None)

def get_void_boundary_handles():
    """Port of ``getVoidBoundaryHandles.m`` -- empty ``SideHandle``.

    MATLAB: ``struct('ArrowHandles',[],'isSelected',[],'isPersistent',[])``
    """
    from ..iga import SideHandle
    return SideHandle(isPersistent=False)

# ---------------------------------------------------------------------------
# Boundary-attributes generation
# ---------------------------------------------------------------------------

def get_uv_w_dirichlet(u_dirichlet: bool, v_dirichlet: bool,
                       w_dirichlet: bool) -> List[bool]:
    """Port of ``getUVWDirichlet.m`` -- build the 6-entry Dirichlet mask.

    MATLAB:
    ::

        UVWDirichlet = logical([checkboxUDirichlet, checkboxUDirichlet, ...
                                checkboxVDirichlet, checkboxVDirichlet, ...
                                checkboxWDirichlet, checkboxWDirichlet]);

    NOTE (faithful to the MATLAB ground-truth): the U / V / W entries are
    each duplicated in the 6-vector -- one per ``-`` and one per ``+``
    side.  This behaviour is preserved verbatim.
    """
    return [
        bool(u_dirichlet), bool(u_dirichlet),
        bool(v_dirichlet), bool(v_dirichlet),
        bool(w_dirichlet), bool(w_dirichlet),
    ]

def gen_boundaries_attributes(mpHexa, bndrs, uvw_dirichlet):
    """Port of ``genBoundariesAttributes.m`` -- per-side boundary attributes.

    MATLAB:
    ::

        for bb=1:length(bndrs)
            Srfs = nrbextract(mpHexa(bndrs(bb).patches));
            Srf = Srfs(bndrs(bb).faces);
            [d1Srf,d2Srf] = nrbderiv(Srf);
            [pt3D,jac,~] = nrbdeval(Srf,d1Srf,d2Srf,{.5,.5});
            n3D = uvect(cross(uvect(jac{1}),uvect(jac{2})));
            mag = 0;
            isDirichlet = UVWDirichlet(bndrs(bb).faces);
            Attrbs(bb).Srf=Srf; Attrbs(bb).pt3D=pt3D;
            Attrbs(bb).n3D=n3D; Attrbs(bb).mag=mag;
            Attrbs(bb).isDirichlet=isDirichlet;
        end

    Parameters
    ----------
    mpHexa
        Multi-patch NURBS volume (sequence of :class:`Nrb`).
    bndrs
        ``mp_geo_load`` boundary-interfaces list; each element must
        expose ``patches`` (1-based list of patch indices) and
        ``faces`` (1-based side index on the extracted surface).
    uvw_dirichlet
        Length-6 boolean list (see :func:`get_uv_w_dirichlet`).

    Returns
    -------
    list[SideAttribs]
        One ``SideAttribs`` per boundary side, in the same order as
        ``bndrs``.
    """
    from ..iga import SideAttribs
    from ..nurbs import nrbextract, nrbderiv, nrbdeval
    from ..junc import uvect

    attrbs: List[SideAttribs] = []
    # MATLAB wraps the loop in warning('off') / warning('on'); we mirror
    # with warnings.catch_warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for bb in bndrs:
            # ``bb`` is a dict from ``mp_geo_load``:
            #   {'name':..., 'nsides':..., 'patches':[...], 'faces':[...]}
            # both ``patches`` and ``faces`` are lists of 1-based indices.
            patches = list(bb["patches"])       # 1-based
            faces = int(bb["faces"][0])         # 1-based (single entry per side)
            # MATLAB:  Srfs = nrbextract( mpHexa( bndrs(bb).patches ) );
            #           Srf  = Srfs( bndrs(bb).faces );
            # ``nrbextract`` over a *list* of patches returns the 6 boundary
            # faces of each patch, concatenated; ``faces`` then indexes into
            # that flat list.  We reproduce that faithfully.
            Srfs: List[Any] = []
            for p in patches:
                Srfs.extend(nrbextract(mpHexa[p - 1]))
            Srf = Srfs[faces - 1]

            d1Srf = nrbderiv(Srf)
            # MATLAB discards the Hessian (``[pt3D,jac,~]``); we only need
            # jac{1}, jac{2} for the surface normal, so pass ``None`` for the
            # second derivative (matches nrbdeval's optional d2Srf).
            # MATLAB ``{uSmpl,vSmpl}`` = ``{.5,.5}`` is a parametric grid of
            # two scalars; the Python ``tt`` grid is a list of ``ndim`` 1-D
            # arrays, one per parametric direction.
            pt3D, jac, _ = nrbdeval(
                Srf, d1Srf, None, tt=[np.array([0.5]), np.array([0.5])])
            # Shapes come back as (3, 1, 1); flatten each to a length-3 vector
            # before the cross product (mirrors MATLAB's column-vector ``jac{1}``/``jac{2}``).
            jac1 = uvect(np.asarray(jac[0]).reshape(-1)[:3])
            jac2 = uvect(np.asarray(jac[1]).reshape(-1)[:3])
            n3D = uvect(np.cross(jac1, jac2))

            isDirichlet = bool(uvw_dirichlet[faces - 1])
            attrbs.append(SideAttribs(
                Srf=Srf,
                pt3D=np.asarray(pt3D[:, 0]).ravel(),
                n3D=np.asarray(n3D).ravel(),
                mag=0.0,
                isDirichlet=isDirichlet,
            ))
    return attrbs

def gen_arrow_boundary_handles(geom_data: GeomDATA) -> List[Any]:
    """Port of ``genArrowBoundaryHandles.m`` -- one handle per side.

    MATLAB:
    ::

        n = length(GeomDATA.bndrsAttrbs);
        hdls = repmat(getVoidBoundaryHandles(), 1, n);
    """
    return [get_void_boundary_handles() for _ in geom_data.bndrsAttrbs]

# ---------------------------------------------------------------------------
# BC validity
# ---------------------------------------------------------------------------

def chk_valid_bc_vol_source(sim_data: Any) -> bool:
    """Port of ``chkValidBCVolSource.m``.

    MATLAB:
    ::

        isValidVolSource = (f>0 && (fx!=0 || fy!=0 || fz!=0));
        isValidDirichlet = ~isempty(SimDATA.BoundaryConditions.dirichlet.IDs);
        isValidLeading = NOT ( isempty(leading.IDs) || ...
                               ( isempty(leading.m) || all(leading.m==0) ) );
        isValid = (isValidLeading & isValidDirichlet) | ...
                  (isValidVolSource & isValidDirichlet);

    A simulation is valid when at least one *leading* (Neumann) side has a
    non-zero magnitude AND a Dirichlet side exists, OR the volume source
    is non-zero AND a Dirichlet side exists.
    """
    BC = sim_data.BoundaryConditions
    VS = sim_data.VolumetricSource

    dirichlet = BC.dirichlet
    leading = BC.leading

    f = float(VS.f)
    fx, fy, fz = float(VS.fx), float(VS.fy), float(VS.fz)

    is_valid_vol_source = (f > 0.0) and (fx != 0.0 or fy != 0.0 or fz != 0.0)
    is_valid_dirichlet = dirichlet is not None and len(dirichlet.IDs) > 0
    is_valid_leading = not (
        (leading is None) or
        len(leading.IDs) == 0 or
        len(leading.m) == 0 or
        (len(leading.m) > 0 and np.all(leading.m == 0.0))
    )
    return (is_valid_leading and is_valid_dirichlet) or \
           (is_valid_vol_source and is_valid_dirichlet)

# ---------------------------------------------------------------------------
# Geometry import (headless)
# ---------------------------------------------------------------------------

def import_geometry(path: str,
                    u_dirichlet: bool = True,
                    v_dirichlet: bool = True,
                    w_dirichlet: bool = True) -> Tuple[GeomDATA, bool]:
    """Headless port of ``importGeometryFile.m``.

    Mirrors the ``.txt`` branch of the MATLAB function (the ``.mat``
    branch is not ported because the MATLAB ``.mat`` format is not
    natively readable in Python).

    MATLAB (abridged):
    ::

        GeomDATA = getVoidGeomDATA;
        [geometry, GeomDATA.bndrs] = mp_geo_load(path);
        GeomDATA.mpHexa = cat(2, geometry.nurbs);
        UVWDirichlet = getUVWDirichlet(handles);
        GeomDATA.bndrsAttrbs = genBoundariesAttributes(mpHexa, bndrs, UVWDirichlet);
        GeomDATA.bndrsHandles = genArrowBoundaryHandles(GeomDATA);
        GeomDATA.volSource = getVoidVolSource;

    Returns
    -------
    (GeomDATA, isLoaded)
        ``isLoaded`` is ``False`` on failure (mirrors MATLAB's
        ``try/catch``).
    """
    from ..iga import mp_geo_load

    geom = get_void_geom_data()
    try:
        # Python ``mp_geo_load`` returns five values (faithful to MATLAB):
        #   [geometry, boundaries, interfaces, subdomains, boundary_interfaces]
        (geometry, bndrs, _interfaces, _subdomains, _bndrs_ifaces) = mp_geo_load(path)
        # MATLAB:  GeomDATA.mpHexa = cat(2, geometry.nurbs);
        # ``geometry`` is a list of per-patch dicts, each with a ``'nurbs'``
        # entry; concatenating all patches into one flat Nrb sequence.
        geom.mpHexa = tuple(g["nurbs"] for g in geometry)
        geom.bndrs = bndrs
        uvw = get_uv_w_dirichlet(u_dirichlet, v_dirichlet, w_dirichlet)
        geom.bndrsAttrbs = gen_boundaries_attributes(
            geom.mpHexa, bndrs, uvw)
        geom.bndrsHandles = gen_arrow_boundary_handles(geom)
        geom.volSource = get_void_vol_source()
        geom.FileName = os.path.basename(path)
        geom.FilePath = os.path.dirname(os.path.abspath(path))
        return geom, True
    except Exception as e:
        warnings.warn(f"import_geometry failed: {e}")
        return geom, False

# ---------------------------------------------------------------------------
# Boundary-data persistence (numpy ``.npz``)
# ---------------------------------------------------------------------------
# MATLAB ``HexIGASim.m`` loads/saves boundary data as ``.mat`` structs with the
# fields ``FileName`` / ``bndrsAttrbs`` / ``bndrsHandles`` (+ optional
# ``volSource``).  The Python port replaces the ``.mat`` with a **numpy**
# ``.npz`` (zip-of-arrays) container carrying the same fields, per the user
# directive ("should be replaced by numpy analog formats that are compatible
# with python").  The NURBS surface (``Srf``) is geometry-derived and is
# regenerated by :func:`import_geometry`; it is not persisted here -- the
# persisted records hold the per-side scalar/attribute data that the MATLAB
# boundary files carry.

def _serialise_boundary(geom_data: GeomDATA) -> "dict[str, Any]":
    """Pack ``geom_data``'s boundary records into a plain ``npz``-compatible
    dict of numpy arrays (one column per side)."""
    attrbs = list(geom_data.bndrsAttrbs)
    handles = list(geom_data.bndrsHandles)
    n = max(len(attrbs), len(handles))

    def _attr_col(attr_name: str, default: Any) -> np.ndarray:
        col = np.full(n, default, dtype=object)
        for i in range(n):
            if i < len(attrbs):
                col[i] = getattr(attrbs[i], attr_name, default)
        return col

    def _hdl_col(hdl_name: str, default: Any) -> np.ndarray:
        col = np.full(n, default, dtype=object)
        for i in range(n):
            if i < len(handles):
                col[i] = getattr(handles[i], hdl_name, default)
        return col

    vs = geom_data.volSource
    vol_f = float(getattr(vs, "f", 0.0))
    vol_fx = float(getattr(vs, "fx", 0.0))
    vol_fy = float(getattr(vs, "fy", 0.0))
    vol_fz = float(getattr(vs, "fz", 0.0))
    return {
        "FileName": np.asarray(geom_data.FileName, dtype=object),
        "nSides": np.asarray(n, dtype=np.int64),
        "attr_mag": _attr_col("mag", 0.0),
        "attr_isDirichlet": _attr_col("isDirichlet", False),
        "attr_pt3D": _attr_col("pt3D", None),
        "attr_n3D": _attr_col("n3D", None),
        "hdl_isSelected": _hdl_col("isSelected", False),
        "hdl_isPersistent": _hdl_col("isPersistent", False),
        "vol_f": np.asarray(vol_f, dtype=np.float64),
        "vol_fx": np.asarray(vol_fx, dtype=np.float64),
        "vol_fy": np.asarray(vol_fy, dtype=np.float64),
        "vol_fz": np.asarray(vol_fz, dtype=np.float64),
    }

def save_boundaries(geom_data: GeomDATA, path: str) -> bool:
    """Headless port of the MATLAB *Save Boundaries* action -- persist the
    boundary records to a numpy ``.npz`` file."""
    try:
        np.savez(path, **_serialise_boundary(geom_data))
        return True
    except Exception as e:
        warnings.warn(f"save_boundaries failed: {e}")
        return False

def import_boundaries(path: str) -> Tuple[Any, bool]:
    """Headless port of MATLAB ``importBoundaries`` -- load boundary records
    from a numpy ``.npz`` file and rebuild the dataclass records.

    Returns ``(bb_data, isBBLoaded)`` where ``bb_data`` is a dict with keys
    ``FileName`` / ``bndrsAttrbs`` / ``bndrsHandles`` / ``volSource`` (mirrors
    the MATLAB output struct) or ``None`` on failure.
    """
    try:
        with np.load(path, allow_pickle=True) as z:
            if "attr_mag" not in z or "nSides" not in z:
                return None, False
            n = int(z["nSides"])
            file_name = str(np.asarray(z["FileName"], dtype=object))
            mag = np.asarray(z["attr_mag"], dtype=object)
            is_dir = np.asarray(z["attr_isDirichlet"], dtype=object)
            pt3d = np.asarray(z["attr_pt3D"], dtype=object)
            n3d = np.asarray(z["attr_n3D"], dtype=object)
            sel = np.asarray(z["hdl_isSelected"], dtype=object)
            pers = np.asarray(z["hdl_isPersistent"], dtype=object)
            vol_f = float(z["vol_f"]) if "vol_f" in z else 0.0
            vol_fx = float(z["vol_fx"]) if "vol_fx" in z else 0.0
            vol_fy = float(z["vol_fy"]) if "vol_fy" in z else 0.0
            vol_fz = float(z["vol_fz"]) if "vol_fz" in z else 0.0

        from ..iga import SideAttribs, SideHandle, VolumetricSource
        attrbs = [
            SideAttribs(
                Srf=None,
                pt3D=(pt3d[i] if pt3d[i] is not None else None),
                n3D=(n3d[i] if n3d[i] is not None else None),
                mag=float(mag[i]) if mag[i] is not None else 0.0,
                isDirichlet=bool(is_dir[i]) if is_dir[i] is not None else False,
            )
            for i in range(n)
        ]
        handles = [
            SideHandle(
                isSelected=bool(sel[i]) if sel[i] is not None else False,
                isPersistent=bool(pers[i]) if pers[i] is not None else False,
            )
            for i in range(n)
        ]
        vol = VolumetricSource(
            f=vol_f,
            fx=vol_fx,
            fy=vol_fy,
            fz=vol_fz,
        )
        bb = {
            "FileName": file_name,
            "bndrsAttrbs": attrbs,
            "bndrsHandles": handles,
            "volSource": vol,
        }
        return bb, True
    except Exception as e:
        warnings.warn(f"import_boundaries failed: {e}")
        return None, False

# ---------------------------------------------------------------------------
# Simulation runner (headless)
# ---------------------------------------------------------------------------

def run_sim(geom_data: GeomDATA, kind: str = "fluid") -> Tuple[GeomDATA, bool]:
    """Headless port of ``pushbuttonRunSim_Callback``.

    Dispatches to the fluid / elastic / Maxwell solver based on ``kind``
    (``'fluid'`` | ``'elastic'`` | ``'maxwell'``).  Mirrors the MATLAB
    callback's conversion -> validation -> solve -> store pipeline.

    Returns
    -------
    (GeomDATA, ok)
        ``GeomDATA.FluidDATA`` / ``ElasticDATA`` / ``MaxwellDATA`` is
        populated on success.  ``ok`` is ``False`` when validation
        fails (BC or volume source missing) or the solver raises.
    """
    from ..iga import (
        convert_geom_data_2_fluid_data,
        convert_geom_data_2_elastic_data,
        convert_geom_data_2_maxwell_data,
        run_fluid_simulation,
        run_elastic_simulation,
        run_maxwell_simulation,
    )

    kind = kind.lower()
    if kind in ("fluid", "stokes", "pressflow", "flow"):
        sim_data = convert_geom_data_2_fluid_data(geom_data)
        if not chk_valid_bc_vol_source(sim_data):
            warnings.warn("Missing Valid Boundary Conditions!")
            return geom_data, False
        sim_data = run_fluid_simulation(sim_data)
        geom_data.FluidDATA = sim_data
        return geom_data, True
    elif kind in ("elastic", "elasticdeform", "deform"):
        sim_data = convert_geom_data_2_elastic_data(geom_data)
        if not chk_valid_bc_vol_source(sim_data):
            warnings.warn("Missing Valid Boundary Conditions!")
            return geom_data, False
        sim_data = run_elastic_simulation(sim_data)
        geom_data.ElasticDATA = sim_data
        return geom_data, True
    elif kind in ("maxwell", "electricfield", "eig"):
        sim_data = convert_geom_data_2_maxwell_data(geom_data)
        # NOTE: MATLAB does NOT gate Maxwell on chkValidBCVolSource.
        sim_data = run_maxwell_simulation(sim_data)
        geom_data.MaxwellDATA = sim_data
        return geom_data, True
    else:
        raise ValueError(
            f"Unknown simulation kind: {kind!r} "
            f"(expected 'fluid' | 'elastic' | 'maxwell')"
        )

# ---------------------------------------------------------------------------
# GUI (optional -- only when tkinter is available)
# ---------------------------------------------------------------------------

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
    _TK_AVAILABLE = True
except Exception:  # pragma: no cover - tkinter is optional at runtime
    tk = None
    ttk = None
    filedialog = None
    messagebox = None
    simpledialog = None
    _TK_AVAILABLE = False

if _TK_AVAILABLE:

    class HexIGASim:
        """Tk GUI mirroring the MATLAB ``HexIGASim`` figure.

        Layout (top to bottom, left to right, mirroring the MATLAB GUIDE
        figure):

        * Radio buttons: ``PressFlow`` / ``ElasticDeform`` / ``ElectricField``
        * Per-axis Dirichlet checkboxes: ``U`` / ``V`` / ``W``
        * Geometry controls: Load Geom, Save Geometry, Load Boundaries,
          Save Boundaries, Reset, Export Solution
        * Side controls: Pick Side, Side ID label, Side Dirichlet,
          Flip Dir, Magnitude (edit)
        * Volume source: f (mag), fx, fy, fz (edit fields)
        * Maxwell: EIG (edit)
        * Run Sim button
        * Output panel (text widget, read-only)

        Constructable without showing -- ``build()`` creates widgets,
        ``mainloop()`` starts the event loop.  Headless smoke tests call
        ``build()`` only.
        """

        def __init__(self, parent: Any = None):
            if not _TK_AVAILABLE:
                raise RuntimeError("tkinter is not available")
            if parent is None:
                self.root = tk.Tk()
                self._owns_root = True
            else:
                self.root = parent
                self._owns_root = False
            self.root.title("HexIGA - Simulation (HexIGASim)")
            self.root.geometry("720x600")

            self.geom_data: GeomDATA = get_void_geom_data()
            self.side_picked: Optional[int] = None  # 1-based side ID

            # Track widget handles (mirrors MATLAB 'handles' struct)
            self.btns: dict = {}
            self.entries: dict = {}
            self.checks: dict = {}
            self.check_vars: dict = {}
            self.radios: dict = {}
            self.texts: dict = {}

            self._build()

        # -- widget construction -------------------------------------------

        def _build(self) -> None:
            """Create all widgets (called from ``__init__`` and from
            ``build()`` for re-use in tests)."""
            self.build()

        def build(self) -> None:
            """Create all widgets.

            Split out from ``__init__`` so that headless smoke tests can
            call ``build()`` on a pre-constructed instance.
            """
            if not _TK_AVAILABLE:
                return
            root = self.root

            # ---- Simulation type ------------------------------------------
            sim_frame = ttk.LabelFrame(root, text="Simulation Type")
            sim_frame.pack(fill="x", padx=8, pady=4)

            self._sim_var = tk.StringVar(value="fluid")
            self.radios["PressFlow"] = ttk.Radiobutton(
                sim_frame, text="PressFlow (Stokes)",
                value="fluid", variable=self._sim_var)
            self.radios["ElasticDeform"] = ttk.Radiobutton(
                sim_frame, text="ElasticDeform (Linear Elasticity)",
                value="elastic", variable=self._sim_var)
            self.radios["ElectricField"] = ttk.Radiobutton(
                sim_frame, text="ElectricField (Maxwell Eig)",
                value="maxwell", variable=self._sim_var)
            self.radios["PressFlow"].grid(row=0, column=0, padx=4, pady=2,
                                          sticky="w")
            self.radios["ElasticDeform"].grid(row=0, column=1, padx=4, pady=2,
                                              sticky="w")
            self.radios["ElectricField"].grid(row=0, column=2, padx=4, pady=2,
                                              sticky="w")

            # ---- Per-axis Dirichlet ---------------------------------------
            dir_frame = ttk.LabelFrame(root, text="Dirichlet Sides")
            dir_frame.pack(fill="x", padx=8, pady=4)

            self.check_vars["U"] = tk.BooleanVar(value=True)
            self.check_vars["V"] = tk.BooleanVar(value=True)
            self.check_vars["W"] = tk.BooleanVar(value=True)
            self.checks["U"] = ttk.Checkbutton(
                dir_frame, text="U", variable=self.check_vars["U"])
            self.checks["V"] = ttk.Checkbutton(
                dir_frame, text="V", variable=self.check_vars["V"])
            self.checks["W"] = ttk.Checkbutton(
                dir_frame, text="W", variable=self.check_vars["W"])
            self.checks["U"].grid(row=0, column=0, padx=4, pady=2, sticky="w")
            self.checks["V"].grid(row=0, column=1, padx=4, pady=2, sticky="w")
            self.checks["W"].grid(row=0, column=2, padx=4, pady=2, sticky="w")

            # ---- Geometry controls ----------------------------------------
            geom_frame = ttk.LabelFrame(root, text="Geometry")
            geom_frame.pack(fill="x", padx=8, pady=4)

            def _btn(parent, row, col, text, cmd, key):
                b = ttk.Button(parent, text=text, command=cmd)
                b.grid(row=row, column=col, padx=4, pady=2)
                self.btns[key] = b
                return b

            _btn(geom_frame, 0, 0, "Load Geom", self._on_load_geom,
                 "LoadGeom")
            _btn(geom_frame, 0, 1, "Save Geometry", self._on_save_geom,
                 "SaveGeometry")
            _btn(geom_frame, 0, 2, "Load Boundaries",
                 self._on_load_boundaries, "LoadBoundaries")
            _btn(geom_frame, 0, 3, "Save Boundaries",
                 self._on_save_boundaries, "SaveBoundaries")
            _btn(geom_frame, 0, 4, "Reset", self._on_reset, "Reset")
            _btn(geom_frame, 0, 5, "Export Solution",
                 self._on_export_solution, "ExportSolution")

            # ---- Side controls --------------------------------------------
            side_frame = ttk.LabelFrame(root, text="Side Options")
            side_frame.pack(fill="x", padx=8, pady=4)

            self.texts["SideID"] = ttk.Label(side_frame, text="Side ID: #")
            self.texts["SideID"].grid(row=0, column=0, padx=4, pady=2,
                                      sticky="w")

            self.check_vars["SideDirichlet"] = tk.BooleanVar(value=False)
            self.checks["SideDirichlet"] = ttk.Checkbutton(
                side_frame, text="Dirichlet",
                variable=self.check_vars["SideDirichlet"],
                command=self._on_side_dirichlet_toggle)
            self.checks["SideDirichlet"].grid(row=0, column=1, padx=4,
                                              pady=2, sticky="w")
            self.checks["SideDirichlet"].config(state="disabled")

            self.btns["PickSide"] = ttk.Button(
                side_frame, text="Pick Side", command=self._on_pick_side)
            self.btns["PickSide"].grid(row=0, column=2, padx=4, pady=2)

            self.btns["FlipDir"] = ttk.Button(
                side_frame, text="Flip Dir", command=self._on_flip_dir)
            self.btns["FlipDir"].grid(row=0, column=3, padx=4, pady=2)
            self.btns["FlipDir"].config(state="disabled")

            ttk.Label(side_frame, text="Mag").grid(row=0, column=4,
                                                   padx=2, pady=2,
                                                   sticky="e")
            self.entries["Mag"] = ttk.Entry(side_frame, width=8)
            self.entries["Mag"].insert(0, "0.00")
            self.entries["Mag"].grid(row=0, column=5, padx=4, pady=2)
            self.entries["Mag"].config(state="disabled")

            # ---- Volume source --------------------------------------------
            vs_frame = ttk.LabelFrame(root, text="Volumetric Source")
            vs_frame.pack(fill="x", padx=8, pady=4)

            def _entry(parent, row, col, label, key, default="0.0"):
                ttk.Label(parent, text=label).grid(row=row, column=col,
                                                   padx=2, pady=2, sticky="e")
                e = ttk.Entry(parent, width=8)
                e.insert(0, default)
                e.grid(row=row, column=col + 1, padx=4, pady=2)
                self.entries[key] = e
                return e

            _entry(vs_frame, 0, 0, "f  ", "fmag")
            _entry(vs_frame, 0, 2, "fx", "fx")
            _entry(vs_frame, 0, 4, "fy", "fy")
            _entry(vs_frame, 0, 6, "fz", "fz")

            # ---- Maxwell EIG ----------------------------------------------
            eig_frame = ttk.LabelFrame(root, text="Maxwell")
            eig_frame.pack(fill="x", padx=8, pady=4)
            _entry(eig_frame, 0, 0, "EIG", "EIG", default="1.0")

            # ---- Run -------------------------------------------------------
            run_frame = ttk.Frame(root)
            run_frame.pack(fill="x", padx=8, pady=6)
            self.btns["RunSim"] = ttk.Button(
                run_frame, text="Run Simulation",
                command=self._on_run_sim)
            self.btns["RunSim"].pack(side="left", padx=4)

            # ---- Output panel ---------------------------------------------
            out_frame = ttk.LabelFrame(root, text="Output")
            out_frame.pack(fill="both", expand=True, padx=8, pady=4)
            self.output = tk.Text(out_frame, height=10, state="disabled",
                                  wrap="word")
            sb = ttk.Scrollbar(out_frame, command=self.output.yview)
            self.output.config(yscrollcommand=sb.set)
            self.output.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")

            self._apply_state()

        # -- state helpers --------------------------------------------------

        def _apply_state(self) -> None:
            """Enable/disable widgets based on the current state."""
            if not _TK_AVAILABLE:
                return
            loaded = self.geom_data.is_loaded()
            state = "normal" if loaded else "disabled"

            for key in ("PickSide", "FlipDir", "ExportSolution", "SaveGeometry"):
                if key in self.btns:
                    self.btns[key].config(state="disabled"
                                          if key == "PickSide" and not loaded
                                          else state)
            self.btns["PickSide"].config(state="normal" if loaded
                                         else "disabled")
            self.btns["SaveGeometry"].config(state="normal" if loaded
                                             else "disabled")
            self.btns["ExportSolution"].config(state="disabled")

            # Side options are disabled until a side is picked.
            if self.side_picked is None:
                self.checks["SideDirichlet"].config(state="disabled")
                self.btns["FlipDir"].config(state="disabled")
                self.entries["Mag"].config(state="disabled")
            else:
                self.checks["SideDirichlet"].config(state="normal")
                is_dirichlet = self.geom_data.bndrsAttrbs[
                    self.side_picked - 1].isDirichlet
                if is_dirichlet:
                    self.btns["FlipDir"].config(state="disabled")
                    self.entries["Mag"].config(state="disabled")
                else:
                    self.btns["FlipDir"].config(state="normal")
                    self.entries["Mag"].config(state="normal")

        def _log(self, msg: str) -> None:
            if not _TK_AVAILABLE:
                return
            self.output.config(state="normal")
            self.output.insert("end", msg + "\n")
            self.output.see("end")
            self.output.config(state="disabled")

        # -- callbacks ------------------------------------------------------

        def _on_load_geom(self) -> None:
            if not _TK_AVAILABLE:
                return
            path = filedialog.askopenfilename(
                title="Load Geometry",
                filetypes=[("Text / MAT", "*.txt *.mat"),
                           ("All files", "*.*")])
            if not path:
                return
            u = bool(self.check_vars["U"].get())
            v = bool(self.check_vars["V"].get())
            w = bool(self.check_vars["W"].get())
            self._log(f" * Loading geometry: {path}")
            geom, ok = import_geometry(path, u, v, w)
            if ok:
                self.geom_data = geom
                self.side_picked = None
                self._log(f" * Loaded: {len(geom.bndrsAttrbs)} boundary sides")
                self._log(f" * mpHexa: {len(geom.mpHexa)} patches")
            else:
                self._log(" * [ERR] Failed to load geometry")
            self._apply_state()

        def _on_save_geom(self) -> None:
            if not _TK_AVAILABLE:
                return
            path = filedialog.asksaveasfilename(
                title="Save Geometry",
                defaultextension=".txt",
                filetypes=[("Text", "*.txt"), ("All files", "*.*")])
            if not path:
                return
            # NOTE: full .mat export is not ported; we write a .txt stub.
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(f"# HexIGA GeomDATA saved from {self.geom_data.FileName}\n")
                    fh.write(f"# Patches: {len(self.geom_data.mpHexa) if self.geom_data.mpHexa else 0}\n")
                    fh.write(f"# Sides  : {len(self.geom_data.bndrsAttrbs)}\n")
                self._log(f" * Saved geometry stub: {path}")
            except Exception as e:
                self._log(f" * [ERR] Save failed: {e}")

        def _on_save_boundaries(self) -> None:
            if self.geom_data.mpHexa is None:
                self._log(" * [WRN] No geometry loaded — nothing to save.")
                return
            path = filedialog.asksaveasfilename(
                parent=self.root, title="Save Boundaries",
                defaultextension=".npz",
                filetypes=[("Numpy archive", "*.npz"), ("All files", "*.*")])
            if not path:
                return
            try:
                ok = save_boundaries(self.geom_data, path)
            except Exception as e:
                self._log(f" * [ERR] Save Boundaries failed: {e}")
                return
            if ok:
                self._log(f" * [OK]  Saved boundaries -> {os.path.basename(path)}")
            else:
                self._log(" * [ERR] Save Boundaries failed (invalid geometry).")

        def _on_load_boundaries(self) -> None:
            path = filedialog.askopenfilename(
                parent=self.root, title="Load Boundaries",
                filetypes=[("Numpy archive", "*.npz"), ("All files", "*.*")])
            if not path:
                return
            bb, ok = import_boundaries(path)
            if not ok:
                self._log(" * [ERR] Load Boundaries failed — not a valid boundary archive.")
                return
            if self.geom_data.mpHexa is None:
                self._log(" * [WRN] No geometry loaded — boundaries not merged.")
                return
            if (len(self.geom_data.bndrsAttrbs) != len(bb["bndrsAttrbs"])
                    or len(self.geom_data.bndrsHandles) != len(bb["bndrsHandles"])):
                self._log(" * [ERR] Load Boundaries: side-count mismatch with loaded geometry.")
                return
            for i in range(len(self.geom_data.bndrsAttrbs)):
                self.geom_data.bndrsAttrbs[i].mag = bb["bndrsAttrbs"][i].mag
                self.geom_data.bndrsAttrbs[i].isDirichlet = bb["bndrsAttrbs"][i].isDirichlet
                self.geom_data.bndrsAttrbs[i].pt3D = bb["bndrsAttrbs"][i].pt3D
                self.geom_data.bndrsAttrbs[i].n3D = bb["bndrsAttrbs"][i].n3D
                self.geom_data.bndrsHandles[i].isSelected = bb["bndrsHandles"][i].isSelected
                self.geom_data.bndrsHandles[i].isPersistent = bb["bndrsHandles"][i].isPersistent
            self.geom_data.volSource = bb["volSource"]
            self._log(f" * [OK]  Loaded boundaries from {os.path.basename(path)}")
            self._apply_state()

        def _on_reset(self) -> None:
            self.geom_data = get_void_geom_data()
            self.side_picked = None
            for e in self.entries.values():
                e.delete(0, "end")
                e.insert(0, "0.00" if e is not self.entries.get("EIG")
                         else "1.0")
            self.check_vars["SideDirichlet"].set(False)
            self.checks["SideDirichlet"].config(state="disabled")
            self.texts["SideID"].config(text="Side ID: #")
            self._log(" * Reset")
            self._apply_state()

        def _on_export_solution(self) -> None:
            self._log(" * [WRN] Solution export not ported "
                      "(MATLAB .mat only)")

        def _on_pick_side(self) -> None:
            if not _TK_AVAILABLE:
                return
            if not self.geom_data.is_loaded():
                messagebox.showwarning(
                    "HexIGASim", "Load a geometry first.")
                return
            n = len(self.geom_data.bndrsAttrbs)
            ans = simpledialog.askinteger(
                "Pick Side",
                f"Enter side ID (1..{n}):",
                minvalue=1, maxvalue=n, parent=self.root)
            if ans is None:
                # No side picked -- reset side options (mirrors MATLAB
                # dcm.removeAllDataCursors branch).
                for h in self.geom_data.bndrsHandles:
                    h.isSelected = False
                self.side_picked = None
                self.checks["SideDirichlet"].config(state="disabled")
                self.btns["FlipDir"].config(state="disabled")
                self.entries["Mag"].config(state="disabled")
                self.texts["SideID"].config(text="Side ID: #")
                self._apply_state()
                return
            self.side_picked = int(ans)
            side = self.geom_data.bndrsAttrbs[self.side_picked - 1]
            self.texts["SideID"].config(
                text=f"Side ID: {self.side_picked}")
            self.checks["SideDirichlet"].config(state="normal")
            self.check_vars["SideDirichlet"].set(bool(side.isDirichlet))
            self.entries["Mag"].delete(0, "end")
            self.entries["Mag"].insert(0, f"{side.mag:.2f}")
            # Update isSelected flags (mirrors MATLAB loop).
            for i, h in enumerate(self.geom_data.bndrsHandles):
                if i + 1 == self.side_picked and not side.isDirichlet:
                    h.isSelected = True
                else:
                    h.isSelected = False
            self._apply_state()
            self._log(f" * Picked side {self.side_picked} "
                      f"(Dirichlet={side.isDirichlet}, mag={side.mag:.3f})")

        def _on_side_dirichlet_toggle(self) -> None:
            if self.side_picked is None:
                return
            side = self.geom_data.bndrsAttrbs[self.side_picked - 1]
            is_dirichlet = bool(self.check_vars["SideDirichlet"].get())
            side.isDirichlet = is_dirichlet
            if not is_dirichlet:
                self.geom_data.bndrsHandles[self.side_picked - 1].isSelected = True
            else:
                self.geom_data.bndrsHandles[self.side_picked - 1].isSelected = False
                side.mag = 0.0
                self.entries["Mag"].delete(0, "end")
                self.entries["Mag"].insert(0, f"{side.mag:.2f}")
            self._apply_state()

        def _on_flip_dir(self) -> None:
            if self.side_picked is None:
                return
            side = self.geom_data.bndrsAttrbs[self.side_picked - 1]
            if side.n3D is None:
                return
            side.n3D = -np.asarray(side.n3D)
            self._log(f" * Flipped normal on side {self.side_picked}")

        def _on_run_sim(self) -> None:
            if not _TK_AVAILABLE:
                return
            if not self.geom_data.is_loaded():
                messagebox.showwarning("HexIGASim",
                                       "Load a geometry first.")
                return
            # Update volSource from entries
            try:
                f = float(self.entries["fmag"].get() or 0.0)
                fx = float(self.entries["fx"].get() or 0.0)
                fy = float(self.entries["fy"].get() or 0.0)
                fz = float(self.entries["fz"].get() or 0.0)
            except ValueError:
                f = fx = fy = fz = 0.0
            from ..iga import VolumetricSource
            self.geom_data.volSource = VolumetricSource(
                f=f, fx=fx, fy=fy, fz=fz)

            kind = self._sim_var.get()
            self._log(f" * Running {kind} simulation...")
            try:
                geom, ok = run_sim(self.geom_data, kind)
            except Exception as e:
                self._log(f" * [ERR] Solver failed: {e}")
                return
            if ok:
                self._log(" * [OK] Simulation completed")
                self.btns["ExportSolution"].config(state="normal")
            else:
                messagebox.showwarning(
                    "HexIGASim", "Missing Valid Boundary Conditions!")
            self.geom_data = geom
            self._apply_state()

        def mainloop(self) -> None:
            if self._owns_root:
                self.root.mainloop()

def main() -> None:
    """Launch the HexIGASim GUI."""
    if not _TK_AVAILABLE:
        raise RuntimeError("tkinter is not available")
    app = HexIGASim()
    app.mainloop()

if __name__ == "__main__":
    main()
