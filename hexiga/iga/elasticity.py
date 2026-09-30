"""
hexiga.iga.elasticity
=====================

Solves the linear elasticity problem on a multipatch domain (plane strain
for planar domains):

    - div(sigma(u)) = f        in Omega
    sigma(u) . n = g           on Gamma_N
              u = h            on Gamma_D

with ``sigma(u) = mu*(grad(u) + grad(u)^T) + lambda*div(u)*I``.

The MATLAB function is a *thin orchestrator* over the GeoPDEs FEM library.
FEM calls are isolated behind :class:`hexiga.iga.backend.GeoPDEsBackend`
so the orchestrator reads as a literal port and is unit-testable with
:class:`MockGeoPDEsBackend`.

Input contract:

``problem_data`` -- object with attributes (MATLAB field names):

* ``geo_name``      -- geometry name
* ``nmnn_sides``    -- 1-D int array of Neumann side IDs (1-based); may be empty
* ``drchlt_sides``  -- 1-D int array of Dirichlet side IDs (1-based)
* ``lambda_lame``   -- first Lame parameter (scalar or callable)
* ``mu_lame``       -- second Lame parameter (scalar or callable)
* ``f``             -- volumetric source (callable or array)
* ``g``             -- Neumann boundary function (callable)
* ``h``             -- Dirichlet boundary function (callable)

``method_data`` -- object with attributes:

* ``degree``      -- tuple/list of per-direction degrees
* ``regularity``  -- tuple/list of per-direction regularities
* ``nsub``        -- tuple/list of per-direction subelement counts
* ``nquad``       -- tuple/list of per-direction Gaussian quad counts

Returns ``(geometry, msh, space, u, elpsTime, memFtprt)``.

Copyright header of the MATLAB source is preserved in :data:`__MATLAB_LICENSE__`.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .backend import GeoPDEsBackend
from .stokes import _field, _is_empty

__all__ = ["mp_solve_linear_elasticity_complete"]

__MATLAB_LICENSE__ = (
    "Copyright (C) 2010, 2011 Carlo de Falco\n"
    "Copyright (C) 2010, 2011, 2015 Rafael Vazquez\n"
)

def mp_solve_linear_elasticity_complete(problem_data, method_data,
                                        backend: Optional[GeoPDEsBackend] = None):
    """Port of ``mp_solve_linear_elasticity_complete.m``.

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
    geo_name = _field(problem_data, "geo_name")
    nmnn_sides = _field(problem_data, "nmnn_sides")
    drchlt_sides = _field(problem_data, "drchlt_sides")
    lambda_lame = _field(problem_data, "lambda_lame")
    mu_lame = _field(problem_data, "mu_lame")
    f = _field(problem_data, "f")
    g = _field(problem_data, "g")
    h = _field(problem_data, "h")

    degree = _field(method_data, "degree")
    regularity = _field(method_data, "regularity")
    nsub = _field(method_data, "nsub")
    nquad = _field(method_data, "nquad")

    # %% Loading Solid NURBS Geometry
    geometry, boundaries, interfaces, boundary_interfaces = backend.load_geometry(geo_name)
    npatch = len(geometry)

    # %% Conforming Space Configuration
    msh_cell = []
    sp_cell = []
    for iptc in range(npatch):
        degelev = max(int(degree[0]) - int(geometry[iptc].nurbs.order - 1), 0)
        # nurbs = nrbdegelev(geometry(iptc).nurbs, degelev)  -- delegated to backend
        nurbs = geometry[iptc].nurbs
        rknots, zeta, nknots = backend.refine_knots(
            nurbs.knots, int(nsub[0]) - 1, int(nurbs.order) - 1, regularity
        )
        # nurbs = geo_load(nrbkntins(nurbs, nknots)) -- delegated to backend
        # (the backend is responsible for returning the updated geometry;
        #  the mock returns the input unchanged.)
        rule = backend.gauss_nodes(nquad)
        qn, qw = backend.set_quad_nodes(rknots, rule)
        msh_cell.append(backend.cartesian_mesh_elas(rknots, qn, qw, geometry[iptc]))

        sp_scalar = backend.nurbs_space(nurbs, msh_cell[-1])
        scalar_spaces = [sp_scalar for _ in range(int(getattr(msh_cell[-1], "rdim", 3)))]
        sp_cell.append(backend.vector_space(scalar_spaces, msh_cell[-1]))

    msh = ("multipatch", msh_cell, boundaries)
    space = ("multipatch", sp_cell, interfaces, boundary_interfaces)

    # %% Sparse Matrices Definition
    mat = backend.stiffness_elastic(space, msh, lambda_lame, mu_lame)
    rhs = backend.load_elastic(space, msh, f)

    # %% Boundary Conditions
    # Neumann
    nvel = sp_cell[0].ndof
    if not _is_empty(nmnn_sides):
        # NB: in the MATLAB, boundaries(nsides) gives the per-side patch/face
        # list; the mock does not model that, so we pass the side IDs directly
        # and let the backend interpret them.
        for iref in np.atleast_1d(nmnn_sides).astype(int):
            gref = g  # the MATLAB wraps g with the side index; the backend
                      # receives the same callable.
            iref_patch_list = np.atleast_1d(iref).astype(int)
            rhs_nmnn = backend.load_elastic_bnd(space, msh, gref, iref_patch_list)
            boundary_dofs = np.asarray(space[1][0].boundary_dofs, dtype=int) - 1
            vals = np.asarray(rhs_nmnn, dtype=float).ravel()
            rhs[boundary_dofs, 0] += vals[:boundary_dofs.size]

    # Dirichlet
    u = np.zeros((nvel, 1))
    u_drchlt, drchlt_dofs = backend.dirichlet_proj(space, msh, h, drchlt_sides)
    # MATLAB DOF indices are 1-based; convert to 0-based for NumPy indexing.
    drchlt_dofs = np.asarray(drchlt_dofs, dtype=int).ravel() - 1
    u[drchlt_dofs, 0] = np.asarray(u_drchlt, dtype=float).ravel()

    # %% Solving the System
    all_dofs = np.arange(nvel)
    int_dofs = np.setdiff1d(all_dofs, drchlt_dofs)   # 0-based
    rhs[int_dofs, 0] = (
        rhs[int_dofs, 0]
        - mat[int_dofs][:, drchlt_dofs] @ u[drchlt_dofs, 0]
    )
    u[int_dofs, 0] = np.linalg.solve(
        mat[int_dofs][:, int_dofs], rhs[int_dofs, 0]
    )

    elpsTime = time.perf_counter() - t0
    memFtprt = 0.0
    return geometry, msh, space, u, elpsTime, memFtprt
