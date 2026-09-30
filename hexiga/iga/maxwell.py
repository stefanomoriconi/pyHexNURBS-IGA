"""
hexiga.iga.maxwell
==================

Solves the Maxwell eigenvalue problem on a multipatch domain
(H(curl) / conforming curl-preserving B-spline space):

    curl (1/mu(x) curl (u)) = lambda (epsilon(x) u)   in Omega
      (1/mu(x) curl(u)) x n = 0                       on Gamma_N
                      u x n = 0                       on Gamma_D

The MATLAB function is a *thin orchestrator* over the GeoPDEs FEM library
(``mp_geo_load``, ``kntrefine``, ``knt_derham``, ``msh_*``, ``sp_*``,
``op_curlu_curlv_mp``, ``op_u_v_mp``, ``eig``/``eigs``).  The FEM calls are
isolated behind :class:`hexiga.iga.backend.GeoPDEsBackend` so the
orchestrator reads as a literal port and is unit-testable with
:class:`MockGeoPDEsBackend`.

Input contract:

``problem_data`` -- object with attributes (MATLAB field names):

* ``geo_name``      -- geometry name
* ``drchlt_sides``  -- 1-D int array of Dirichlet side IDs (1-based)
* ``nmnn_sides``    -- 1-D int array of Neumann side IDs (1-based); may be
  ``None`` or empty
* ``c_elec_perm``   -- electric permittivity epsilon(x) (scalar or callable)
* ``c_magn_perm``   -- magnetic permeability mu(x) (scalar or callable)
* ``forceAll``      -- bool; when True solve the *full* eigenproblem,
  otherwise a limited ``maxNumEigs``-mode solve

``method_data`` -- object with attributes:

* ``degree``      -- tuple/list of per-direction degrees
* ``regularity``  -- tuple/list of per-direction regularities
* ``nsub``        -- tuple/list of per-direction subelement counts
* ``nquad``       -- tuple/list of per-direction Gaussian quad counts

Returns a tuple ``(geometry, msh, space, eigv, eigf, elpsTime, memFtprt)``
mirroring the MATLAB output.

Copyright header of the MATLAB source is preserved in :data:`__MATLAB_LICENSE__`.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .backend import GeoPDEsBackend, _multipatch_ndof
from .stokes import _field, _is_empty

__all__ = ["mp_solve_maxwell_eig_complete"]

__MATLAB_LICENSE__ = (
    "Copyright (C) 2010, 2011, 2015 Rafael Vazquez\n"
)

def mp_solve_maxwell_eig_complete(problem_data, method_data,
                                  backend: Optional[GeoPDEsBackend] = None):
    """Port of ``mp_solve_maxwell_eig_complete.m``.

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
    nmnn_sides = _field(problem_data, "nmnn_sides")
    drchlt_sides = _field(problem_data, "drchlt_sides")
    c_elec_perm = _field(problem_data, "c_elec_perm", 1.0)
    c_magn_perm = _field(problem_data, "c_magn_perm", 1.0)
    forceAll = bool(_field(problem_data, "forceAll", False))

    degree = _field(method_data, "degree")
    regularity = _field(method_data, "regularity")
    nsub = _field(method_data, "nsub")
    nquad = _field(method_data, "nquad")

    # %% Loading Solid NURBS Geometry
    geometry, boundaries, interfaces, boundary_interfaces = backend.load_geometry(geo_name)
    npatch = len(geometry)

    # %% Conforming Space Configuration (H(curl) / curl-preserving)
    msh_cell = []
    sp_cell = []
    for iptc in range(npatch):
        degelev = max(int(degree[0]) - int(geometry[iptc].nurbs.order - 1), 0)
        nurbs = geometry[iptc].nurbs
        rknots, zeta, nknots = backend.refine_knots(
            nurbs.knots, int(nsub[0]) - 1, int(nurbs.order) - 1, regularity
        )
        rule = backend.gauss_nodes(nquad)
        qn, qw = backend.set_quad_nodes(rknots, rule)
        msh_cell.append(backend.cartesian_mesh_elas(rknots, qn, qw, geometry[iptc]))

        # MATLAB: [knots_hcurl, degree_hcurl] = knt_derham (knots, degree, 'Hcurl')
        # then one H(curl) scalar space per parametric dimension:
        knots_hcurl, degree_hcurl = backend.derham(rknots, int(degree[0]), "Hcurl")

        scalar_spaces = []
        for idim in range(int(getattr(msh_cell[-1], "ndim", 3))):
            scalar_spaces.append(backend.hcurl_scalar_space(
                knots_hcurl[idim], int(degree_hcurl[idim]), msh_cell[-1]
            ))
        # MATLAB: sp_vector (scalar_spaces, msh, 'curl-preserving')
        sp_cell.append(backend.hcurl_vector_space(scalar_spaces, msh_cell[-1],
                                                  "curl-preserving"))

    msh = ("multipatch", msh_cell, boundaries)
    space = ("multipatch", sp_cell, interfaces, boundary_interfaces)

    # %% Sparse Matrices Definition
    # MATLAB:
    #   if (msh.rdim == 2)  invmu = @(x,y) 1./c_magn_perm (x,y);
    #   elseif (msh.rdim == 3) invmu = @(x,y,z) 1./c_magn_perm (x,y,z);
    #   stiff_mat = op_curlu_curlv_mp (space, space, msh, invmu);
    #   mass_mat  = op_u_v_mp (space, space, msh, c_elec_perm);
    rdim = int(getattr(msh_cell[0], "rdim", 3))
    if rdim in (2, 3):
        invmu = c_magn_perm  # the backend evaluates 1./c_magn_perm internally
    else:
        invmu = 1.0
    stiff_mat = backend.stiffness_maxwell(space, msh, invmu)
    mass_mat = backend.mass_maxwell(space, msh, c_elec_perm)

    # %% Boundary Conditions (homogeneous Dirichlet on drchlt_sides)
    # MATLAB:
    #   bnd_dofs = union of space.boundary.gnum{iref_patch_list} over the
    #   Dirichlet sides, then drchlt_dofs = space.boundary.dofs (bnd_dofs).
    # The per-side patch/face bookkeeping is delegated to the backend
    # (the mock resolves this to a concrete set of DOFs).
    drchlt_dofs = backend.maxwell_dirichlet_dofs(space, msh, drchlt_sides)
    # MATLAB DOF indices are 1-based; convert to 0-based for NumPy indexing.
    drchlt_dofs = np.asarray(drchlt_dofs, dtype=int).ravel() - 1
    all_dofs = np.arange(_multipatch_ndof(space))
    int_dofs = np.setdiff1d(all_dofs, drchlt_dofs)   # 0-based

    # %% Solving the System (eigenvalue problem)
    # MATLAB: forceAll branch -> eig(full(A), full(M), 'vector');
    #         otherwise        -> eigs(A', M', maxNumEigs, eigvTarget).
    maxNumEigs = 100
    if maxNumEigs > int_dofs.size:
        maxNumEigs = int_dofs.size - 2
    eigvTarget = 1e-1

    eigf, eigv = backend.maxwell_eigsolve(stiff_mat, mass_mat, int_dofs,
                                          forceAll, maxNumEigs, eigvTarget)

    elpsTime = time.perf_counter() - t0
    memFtprt = 0.0
    return geometry, msh, space, eigv, eigf, elpsTime, memFtprt
