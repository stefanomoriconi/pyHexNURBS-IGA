"""
hexiga.iga.stokes
=================

Solves the Stokes problem on a multipatch domain:

    -div(mu grad(vel)) + grad(press) = f    in Omega
                            div(vel) = 0    in Omega
        mu dvel/dn - press * n = g          on Gamma_N
                            vel = h         on Gamma_D

The MATLAB function is a *thin orchestrator* over the GeoPDEs FEM library
(``mp_geo_load``, ``msh_*``, ``sp_*``, ``op_*``, ``sp_drchlt_l2_proj``).
The FEM calls are isolated behind :class:`hexiga.iga.backend.GeoPDEsBackend`
so the orchestrator logic reads as a literal port and is unit-testable with
:class:`MockGeoPDEsBackend`.

Input contract:

``problem_data`` -- object with attributes (MATLAB field names):

* ``geo_name``       -- geometry name (file path or loaded object)
* ``drchlt_sides``   -- 1-D int array of Dirichlet side IDs (1-based)
* ``nmnn_sides``     -- 1-D int array of Neumann side IDs (1-based); may be
  ``None`` or empty for an all-Dirichlet problem
* ``f``              -- volumetric source (callable or array)
* ``g``              -- Neumann boundary function (callable)
* ``h``              -- Dirichlet boundary function (callable)
* ``viscosity``      -- scalar viscosity (or callable)

``method_data`` -- object with attributes:

* ``element_name``   -- ``'TH'`` or ``'SG'``
* ``degree``         -- tuple/list of per-direction degrees
* ``regularity``     -- tuple/list of per-direction regularities
* ``nsub``           -- tuple/list of per-direction subelement counts
* ``nquad``          -- tuple/list of per-direction Gaussian quad counts

Returns a tuple ``(geometry, msh, space_v, vel, space_p, press, elpsTime,
memFtprt)`` mirroring the MATLAB output.

Copyright header of the MATLAB source is preserved in :data:`__MATLAB_LICENSE__`.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .backend import GeoPDEsBackend

__all__ = ["mp_solve_stokes_complete"]

__MATLAB_LICENSE__ = (
    "Copyright (C) 2009, 2010 Carlo de Falco\n"
    "Copyright (C) 2010, 2011, 2015 Rafael Vazquez\n"
    "Few changes introduced by Stefano Moriconi -- Nov. 2021\n"
)

def _field(obj, name, default=None):
    """Attribute lookup with a default (mirrors MATLAB struct field access)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def _is_empty(x) -> bool:
    """MATLAB ``isempty`` for the subset of types used here."""
    if x is None:
        return True
    if isinstance(x, (list, tuple)):
        return len(x) == 0
    if isinstance(x, np.ndarray):
        return x.size == 0
    return False

def mp_solve_stokes_complete(problem_data, method_data,
                             backend: Optional[GeoPDEsBackend] = None):
    """Port of ``mp_solve_stokes_complete.m``.

    Parameters
    ----------
    problem_data, method_data :
        See module docstring for the field contract.
    backend : GeoPDEsBackend, optional
        The FEM backend providing the GeoPDEs calls.  If ``None``, a
        :class:`MockGeoPDEsBackend` is used (intended for unit tests).
    """
    from .backend import MockGeoPDEsBackend
    if backend is None:
        backend = MockGeoPDEsBackend()

    t0 = time.perf_counter()

    # %% Extract the fields from the data structures into local variables
    # (MATLAB: for iopt = 1:numel(data_names), eval(...))
    geo_name = _field(problem_data, "geo_name")
    drchlt_sides = _field(problem_data, "drchlt_sides")
    nmnn_sides = _field(problem_data, "nmnn_sides")
    f = _field(problem_data, "f")
    g = _field(problem_data, "g")
    h = _field(problem_data, "h")
    viscosity = _field(problem_data, "viscosity", 1.0)

    element_name = _field(method_data, "element_name", "TH")
    degree = _field(method_data, "degree")
    regularity = _field(method_data, "regularity")
    nsub = _field(method_data, "nsub")
    nquad = _field(method_data, "nquad")

    if str(element_name).upper() in ("RT", "NDL"):
        raise ValueError(
            "For RT and NDL spaces, use mp_solve_stokes_div_conforming"
        )

    # %% Loading Solid NURBS Geometry
    geometry, boundaries, interfaces, boundary_interfaces = backend.load_geometry(geo_name)
    npatch = len(geometry)

    # %% Conforming Space Configuration
    msh_cell = []
    spv_cell = []
    spp_cell = []
    for iptc in range(npatch):
        msh_breaks = backend.set_breaks(element_name, geometry[iptc].nurbs.knots, nsub)
        rule = backend.gauss_nodes(nquad)
        qn, qw = backend.set_quad_nodes(msh_breaks, rule)
        msh_cell.append(backend.cartesian_mesh(msh_breaks, qn, qw, geometry[iptc]))
        spv, spp = backend.bspline_fluid_space(
            element_name, geometry[iptc].nurbs.knots, nsub,
            degree, regularity, msh_cell[-1],
        )
        spv_cell.append(spv)
        spp_cell.append(spp)

    msh = ("multipatch", msh_cell, boundaries)
    space_v = ("multipatch", spv_cell, interfaces, boundary_interfaces)
    space_p = ("multipatch", spp_cell, interfaces, boundary_interfaces)

    # rdim (needed for fun_one) -- pulled from the first mesh if present
    rdim = getattr(msh_cell[0], "rdim", 3)

    def fun_one(*args):
        # MATLAB: @(x, y) ones(size(x))  or  @(x, y, z) ones(size(x))
        a = args[0]
        if isinstance(a, np.ndarray):
            return np.ones(a.shape)
        return 1.0

    # %% Sparse Matrices Definition
    A = backend.stiffness_stokes(space_v, msh, viscosity)
    B = backend.div_matrix(space_v, space_p, msh)
    E = backend.mass_pressure(space_p, msh, fun_one).T
    F = backend.load_stokes(space_v, msh, f)

    nvel = space_v[1][0].ndof  # ndof of first velocity patch (mock: same for all)
    npress = space_p[1][0].ndof

    vel = np.zeros((nvel, 1))
    press = np.zeros((npress, 1))

    # %% Boundary Conditions
    rhs_nmnn = None
    if not _is_empty(nmnn_sides):
        nmnn_mp_idxs = np.vstack([
            np.atleast_1d(nmnn_sides).astype(int),
            np.atleast_1d(nmnn_sides).astype(int),  # boundaries(...).patches
            np.atleast_1d(nmnn_sides).astype(int),  # boundaries(...).faces
        ])
        rhs_nmnn = backend.natbdr_stokes(space_v, msh, g, nmnn_mp_idxs)

    # Dirichlet BC
    vel_drchlt, drchlt_dofs = backend.dirichlet_proj(space_v, msh, h, drchlt_sides)
    # MATLAB DOF indices are 1-based; convert to 0-based for NumPy indexing.
    drchlt_dofs = np.asarray(drchlt_dofs, dtype=int).ravel() - 1

    # MATLAB: N_mat = sparse(space_v.ndof, space_v.ndof, 1) -> identity.
    N_mat = np.eye(nvel)
    N_rhs = np.zeros((nvel, 1))

    vel[drchlt_dofs, 0] = np.asarray(vel_drchlt, dtype=float).ravel()
    all_dofs = np.arange(nvel)
    int_dofs = np.setdiff1d(all_dofs, drchlt_dofs)   # 0-based
    nintdofs = int(int_dofs.size)
    rhs_dir = (-A[int_dofs][:, drchlt_dofs] @ vel[drchlt_dofs, 0]
               + N_mat[int_dofs][:, drchlt_dofs] @ vel[drchlt_dofs, 0])

    # %% Solving the System
    if _is_empty(nmnn_sides):
        # All Dirichlet -> pressure is zero-averaged (saddle-point + mass row).
        # MATLAB: mat = [ A(int,int) - N(int,int), -B(:,int).', zeros(nint,1);
        #               -B(:,int),                 zeros(size(B,1),size(B,1)), E.';
        #                zeros(1,nint),             E,                        0 ]
        # MATLAB: B is (np, nv); E is (1, np).
        #   row1: A(int,int)-N(int,int) (nint,nint) | -B(:,int).' (nint,np) | zeros(nint,1)
        #   row2: -B(:,int) (np,nint)      | zeros(np,np)    | E.' (np,1)
        #   row3: zeros(1,nint)            | E (1,np)        | 0
        # Python E = mass_pressure(...).T is (1, np); E.T is (np, 1).
        mat = np.block([
            [A[int_dofs][:, int_dofs] - N_mat[int_dofs][:, int_dofs],
             -B[:, int_dofs].T,
             np.zeros((nintdofs, 1))],
            [-B[:, int_dofs],
             np.zeros((B.shape[0], B.shape[0])),
             E.T],
            [np.zeros((1, nintdofs)),
             E,
             np.array([[0.0]])],
        ])
        rhs = np.concatenate([
            F[int_dofs, 0] + N_rhs[int_dofs, 0] + rhs_dir,
            B[:, drchlt_dofs] @ vel[drchlt_dofs, 0],
            np.array([0.0]),
        ]).reshape(-1, 1)
    else:
        # With natural BC, the pressure constraint is not needed.
        mat = np.block([
            [A[int_dofs][:, int_dofs] - N_mat[int_dofs][:, int_dofs],
             -B[:, int_dofs].T],
            [-B[:, int_dofs],
             np.zeros((B.shape[0], B.shape[0]))],
        ])
        rhs = np.concatenate([
            F[int_dofs, 0] + N_rhs[int_dofs, 0] + rhs_dir + rhs_nmnn[int_dofs, 0],
            B[:, drchlt_dofs] @ vel[drchlt_dofs, 0],
        ]).reshape(-1, 1)

    sol = np.linalg.lstsq(mat, rhs.ravel(), rcond=None)[0]

    if _is_empty(nmnn_sides):
        vel[int_dofs, 0] = sol[:nintdofs]
        press = sol[nintdofs:-1].reshape(-1, 1)
    else:
        vel[int_dofs, 0] = sol[:nintdofs]
        press = sol[nintdofs:].reshape(-1, 1)

    elpsTime = time.perf_counter() - t0
    memFtprt = 0.0
    return geometry, msh, space_v, vel, space_p, press, elpsTime, memFtprt
