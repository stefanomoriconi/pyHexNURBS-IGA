"""
hexiga.iga.backend
==================

Protocol + test backend for the FEM assembly pieces that in MATLAB are
provided by the external **GeoPDEs** library (see ``includeGeoPDEs.m``).

The HexIGA IGA layer (``mp_solve_stokes_complete``,
``mp_solve_linear_elasticity_complete``) is a *thin orchestrator*: it calls
into GeoPDEs for geometry loading, mesh/space construction, operator
assembly and Dirichlet projection.  Those pieces are **not part of HexIGA**;
they come from the GeoPDEs installation.  In the Python port we isolate that
boundary behind :class:`GeoPDEsBackend` so that:

* the HexIGA orchestrators can be ported literally (they only depend on the
  protocol methods below, mirroring the MATLAB call names);
* the unit tests can drive them with :class:`MockGeoPDEsBackend` (a tiny,
  deterministic stand-in that returns structurally-valid arrays so the
  orchestration, boundary-condition bookkeeping and ``numpy`` solve path
  are all exercised end-to-end without a real FEM library);
* a real backend (wrapping an actual GeoPDEs installation, e.g. via a
  MATLAB bridge or a future native port) can be dropped in by implementing
  the same protocol.

Protocol methods (1:1 with the MATLAB calls used by the solvers):

=============================  ==============================================
Method                         MATLAB (GeoPDEs) call
=============================  ==============================================
``load_geometry``              ``mp_geo_load``
``set_breaks``                 ``msh_set_breaks``
``gauss_nodes``                ``msh_gauss_nodes``
``set_quad_nodes``             ``msh_set_quad_nodes``
``cartesian_mesh``             ``msh_cartesian``
``bspline_fluid_space``        ``sp_bspline_fluid``
``cartesian_mesh_elas``        ``msh_cartesian`` (elasticity path)
``nurbs_space``                ``sp_nurbs``
``vector_space``               ``sp_vector``
``refine_knots``               ``kntrefine``
``stiffness_stokes``           ``op_gradu_gradv_mp``
``div_matrix``                 ``op_div_v_q_mp``
``mass_pressure``              ``op_f_v_mp(space_p, fun_one)``
``load_stokes``                ``op_f_v_mp(space_v, f)``
``natbdr_stokes``              ``mp_natbdrcondsStokes``
``dirichlet_proj``             ``sp_drchlt_l2_proj``
``stiffness_elastic``          ``op_su_ev_mp``
``load_elastic``               ``op_f_v_mp(space, f)``
``load_elastic_bnd``           ``op_f_v_mp(space.boundary, .., iref_patch_list)``
``derham``                     ``knt_derham(knots, degree, 'Hcurl')``
``hcurl_scalar_space``         ``sp_bspline(knots_hcurl{idim}, degree_hcurl{idim}, msh)``
``hcurl_vector_space``         ``sp_vector(scalar_spaces, msh, 'curl-preserving')``
``stiffness_maxwell``          ``op_curlu_curlv_mp``
``mass_maxwell``               ``op_u_v_mp``
``maxwell_dirichlet_dofs``     the "Apply homogeneous Dirichlet BC" DOF loop
                               (union of ``space.boundary.gnum`` over the
                               Dirichlet sides, mapped via ``boundary.dofs``)
=============================  ==============================================

All returned arrays are dense ``np.ndarray`` (the MATLAB solvers work with
sparse matrices; for the orchestrator the two are interchangeable via
indexing, and the mock backend keeps things simple by returning dense).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

try:
    import scipy.linalg
    import scipy.sparse.linalg
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is expected in the dev env
    _HAS_SCIPY = False

__all__ = ["GeoPDEsBackend", "MockGeoPDEsBackend"]

@runtime_checkable
class GeoPDEsBackend(Protocol):
    """Protocol the HexIGA IGA orchestrators depend on.

    See the module docstring for the mapping to MATLAB call names.
    """

    # -- geometry -----------------------------------------------------------
    def load_geometry(self, geo_name: object):
        """``[geometry, boundaries, interfaces, _, boundary_interfaces]``."""
        ...

    # -- mesh / space (Stokes path) ---------------------------------------
    def set_breaks(self, element_name: str, knots: object, nsub: object):
        ...

    def gauss_nodes(self, nquad: object):
        ...

    def set_quad_nodes(self, msh_breaks: object, rule: object):
        ...

    def cartesian_mesh(self, msh_breaks: object, qn: object, qw: object,
                       geometry_patch: object):
        ...

    def bspline_fluid_space(self, element_name: str, knots: object, nsub: object,
                            degree: object, regularity: object, msh: object):
        """Returns ``(space_v, space_p)`` for one patch."""
        ...

    # -- mesh / space (Elasticity path) ------------------------------------
    def refine_knots(self, knots: object, nsub_minus_1: int, order_minus_1: int,
                     regularity: object):
        """``[rknots, zeta, nknots]`` from ``kntrefine``."""
        ...

    def cartesian_mesh_elas(self, knots: object, qn: object, qw: object,
                            geometry_patch: object):
        ...

    def nurbs_space(self, nurbs: object, msh: object):
        ...

    def vector_space(self, scalar_spaces: object, msh: object):
        ...

    # -- operators (Stokes) ------------------------------------------------
    def stiffness_stokes(self, space_v: object, msh: object, viscosity: float):
        """``op_gradu_gradv_mp(space_v, space_v, msh, viscosity)``."""
        ...

    def div_matrix(self, space_v: object, space_p: object, msh: object):
        """``op_div_v_q_mp(space_v, space_p, msh)`` -- returns ``B``."""
        ...

    def mass_pressure(self, space_p: object, msh: object, fun_one: object):
        """``op_f_v_mp(space_p, msh, fun_one)`` -- returns ``E``."""
        ...

    def load_stokes(self, space_v: object, msh: object, f: object):
        """``op_f_v_mp(space_v, msh, f)`` -- returns ``F`` (rhs)."""
        ...

    def natbdr_stokes(self, space_v: object, msh: object, g: object,
                      nmnn_mp_idxs: object):
        """``mp_natbdrcondsStokes(space_v, msh, g, nmnn_mp_idxs)``."""
        ...

    def dirichlet_proj(self, space: object, msh: object, h: object,
                       drchlt_sides: object):
        """``sp_drchlt_l2_proj(space, msh, h, drchlt_sides)`` -- returns
        ``(values, dofs)`` with ``dofs`` a 1-D int array of DOF indices."""
        ...

    # -- operators (Elasticity) --------------------------------------------
    def stiffness_elastic(self, space: object, msh: object, lambda_lame: object,
                          mu_lame: object):
        """``op_su_ev_mp(space, space, msh, lambda_lame, mu_lame)``."""
        ...

    def load_elastic(self, space: object, msh: object, f: object):
        """``op_f_v_mp(space, msh, f)`` -- returns ``rhs``."""
        ...

    def load_elastic_bnd(self, space: object, msh: object, gref: object,
                         iref_patch_list: object):
        """``op_f_v_mp(space.boundary, msh.boundary, gref, iref_patch_list)``."""
        ...

    # -- H(curl) space + Maxwell (eigen) assembly ----------------------------
    def derham(self, knots: object, degree: int, mode: str):
        """``[knots_hcurl, degree_hcurl] = knt_derham (knots, degree, 'Hcurl')``.

        Returns ``(knots_hcurl, degree_hcurl)`` where each is indexable per
        parametric dimension (``[idim]``).
        """
        ...

    def hcurl_scalar_space(self, knots_d: object, degree_d: int, msh: object):
        """``sp_bspline (knots_d, degree_d, msh)`` -- one H(curl) scalar."""
        ...

    def hcurl_vector_space(self, scalar_spaces: object, msh: object, mode: str):
        """``sp_vector (scalar_spaces, msh, 'curl-preserving')``."""
        ...

    def stiffness_maxwell(self, space: object, msh: object, invmu: object):
        """``op_curlu_curlv_mp (space, space, msh, invmu)`` -- curl stiffness."""
        ...

    def mass_maxwell(self, space: object, msh: object, c_elec_perm: object):
        """``op_u_v_mp (space, space, msh, c_elec_perm)`` -- mass matrix."""
        ...

    def maxwell_dirichlet_dofs(self, space: object, msh: object,
                               drchlt_sides: object):
        """Homogeneous-Dirichlet DOF set (1-based, MATLAB convention)."""
        ...

    def maxwell_eigsolve(self, stiff_mat: object, mass_mat: object,
                         int_dofs: object, forceAll: bool, maxNumEigs: int,
                         eigvTarget: float):
        """Solve the generalized eigenproblem on interior DOFs (see
        ``mp_solve_maxwell_eig_complete`` for the MATLAB branches)."""
        ...

class _MockSpace:
    """Minimal stand-in for a GeoPDEs multipatch space.

    DOF indices are **1-based** (MATLAB convention); the HexIGA
    orchestrators convert to 0-based internally for NumPy indexing.
    """

    def __init__(self, ndof: int, rdim: int):
        self.ndof = int(ndof)
        self.rdim = int(rdim)
        self.boundary_dofs = np.arange(1, 1 + max(1, ndof // 2), dtype=int)
        self.boundary = self  # self-reference so space.boundary.dofs works

    @property
    def dofs(self):
        return np.arange(1, self.ndof + 1, dtype=int)

def _multipatch_ndof(space) -> int:
    """Extract ``ndof`` from either a bare space (``.ndof``) or a multipatch
    tuple ``('multipatch', cell_list, ...)`` (first cell's ``.ndof``)."""
    if hasattr(space, "ndof"):
        return int(space.ndof)
    # multipatch tuple form: (tag, cell_list, ...)
    cell_list = space[1]
    return int(cell_list[0].ndof)

class _MockNurbs:
    """Minimal stand-in for a GeoPDEs multipatch nurbs (knot vectors + order)."""

    def __init__(self, order: int = 4, dim: int = 3):
        self.order = int(order)
        self.knots = [np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
                      for _ in range(int(dim))]

class _MockPatch:
    """Minimal stand-in for a GeoPDEs multipatch element (a cell)."""

    def __init__(self, order: int = 4):
        self.nurbs = _MockNurbs(order=order, dim=3)

class MockGeoPDEsBackend:
    """Deterministic stand-in for the GeoPDEs backend.

    Returns structurally-valid, tiny arrays so the HexIGA orchestrators
    (``mp_solve_stokes_complete`` / ``mp_solve_linear_elasticity_complete``)
    can be exercised end-to-end in unit tests without a real FEM library:

    * geometry -- ``npatch`` cubic Bezier patches (1- or 3-D parametric)
    * stiffness -- identity (or ``viscosity * I`` for Stokes)
    * ``B`` (divergence) -- zeros, shape ``(space_p.ndof, space_v.ndof)``
    * ``E`` (pressure mass) -- ones, shape ``(space_p.ndof, 1)``
    * ``F`` / ``rhs`` -- ones vector
    * Dirichlet projection -- first half of DOFs, value ``1.0``

    The mock keeps the MATLAB call signatures verbatim so the orchestrator
    code reads as a literal port.
    """

    # -- configurable knobs -------------------------------------------------
    def __init__(self, npatch: int = 2, ndof_v: int = 10, ndof_p: int = 5,
                 ndof_e: int = 12, rdim: int = 3):
        self.npatch = int(npatch)
        self.ndof_v = int(ndof_v)
        self.ndof_p = int(ndof_p)
        self.ndof_e = int(ndof_e)
        self.rdim = int(rdim)

    # -- geometry -----------------------------------------------------------
    def load_geometry(self, geo_name):
        geometry = [_MockPatch(order=4) for _ in range(self.npatch)]
        boundaries = [object() for _ in range(self.npatch * 2)]
        interfaces = [object() for _ in range(max(0, self.npatch - 1))]
        boundary_interfaces = [object() for _ in range(max(0, self.npatch - 1))]
        return geometry, boundaries, interfaces, boundary_interfaces

    # -- Stokes path --------------------------------------------------------
    def set_breaks(self, element_name, knots, nsub):
        return ("breaks", element_name, nsub)

    def gauss_nodes(self, nquad):
        return ("gauss", nquad)

    def set_quad_nodes(self, msh_breaks, rule):
        return (np.linspace(0.0, 1.0, 2), np.array([0.5, 0.5]))

    def cartesian_mesh(self, msh_breaks, qn, qw, geometry_patch):
        m = _MockSpace(self.ndof_v, self.rdim)
        m.kind = "stokes"
        return m

    def bspline_fluid_space(self, element_name, knots, nsub, degree,
                            regularity, msh):
        spv = _MockSpace(self.ndof_v, self.rdim)
        spp = _MockSpace(self.ndof_p, self.rdim)
        return spv, spp

    # -- Elasticity path ----------------------------------------------------
    def refine_knots(self, knots, nsub_minus_1, order_minus_1, regularity):
        return (np.asarray(knots, dtype=float), np.array([0.5]), np.array([2]))

    def cartesian_mesh_elas(self, knots, qn, qw, geometry_patch):
        m = _MockSpace(self.ndof_e, self.rdim)
        m.kind = "elastic"
        return m

    def nurbs_space(self, nurbs, msh):
        return _MockSpace(self.ndof_e, self.rdim)

    def vector_space(self, scalar_spaces, msh):
        return _MockSpace(self.ndof_e, self.rdim)

    # -- Operators (Stokes) -------------------------------------------------
    def stiffness_stokes(self, space_v, msh, viscosity):
        # MATLAB passes viscosity as a function handle (@(x,y,z) ones(size(x)));
        # evaluate it to a scalar for the mock (isotropic, uniform = 1.0).
        if callable(viscosity):
            try:
                val = viscosity(1.0)
                vis = float(np.asarray(val).ravel()[0])
            except Exception:
                vis = 1.0
        else:
            vis = float(viscosity)
        return vis * np.eye(_multipatch_ndof(space_v))

    def div_matrix(self, space_v, space_p, msh):
        return np.zeros((_multipatch_ndof(space_p), _multipatch_ndof(space_v)))

    def mass_pressure(self, space_p, msh, fun_one):
        return np.ones((_multipatch_ndof(space_p), 1))

    def load_stokes(self, space_v, msh, f):
        return np.ones((_multipatch_ndof(space_v), 1))

    def natbdr_stokes(self, space_v, msh, g, nmnn_mp_idxs):
        return np.zeros((_multipatch_ndof(space_v), 1))

    def dirichlet_proj(self, space, msh, h, drchlt_sides):
        n = max(1, _multipatch_ndof(space) // 2)
        values = np.ones((n, 1))
        # DOF indices follow the MATLAB (1-based) convention; the HexIGA
        # orchestrators normalise to 0-based internally for NumPy indexing.
        dofs = np.arange(1, 1 + n, dtype=int)
        return values, dofs

    # -- Operators (Elasticity) ---------------------------------------------
    def stiffness_elastic(self, space, msh, lambda_lame, mu_lame):
        return np.eye(_multipatch_ndof(space))

    def load_elastic(self, space, msh, f):
        return np.ones((_multipatch_ndof(space), 1))

    def load_elastic_bnd(self, space, msh, gref, iref_patch_list):
        return np.zeros((_multipatch_ndof(space), 1))

    # -- H(curl) / Maxwell path ---------------------------------------------
    def derham(self, knots, degree, mode):
        n = len(knots) if isinstance(knots, (list, tuple)) else 3
        knots_hcurl = [np.asarray(knots[i % len(knots)], dtype=float)
                       for i in range(n)]
        degree_hcurl = [int(degree)] * n
        return knots_hcurl, degree_hcurl

    def hcurl_scalar_space(self, knots_d, degree_d, msh):
        return _MockSpace(self.ndof_e, self.rdim)

    def hcurl_vector_space(self, scalar_spaces, msh, mode):
        m = _MockSpace(self.ndof_e, self.rdim)
        m.kind = "maxwell"
        return m

    def stiffness_maxwell(self, space, msh, invmu):
        """Diagonal matrix diag(1..nd): with an identity mass matrix the
        generalized eigenvalues equal the diagonal entries, so hand-computed
        expectations are possible in tests."""
        nd = _multipatch_ndof(space)
        return np.diag(np.arange(1, nd + 1, dtype=float))

    def mass_maxwell(self, space, msh, c_elec_perm):
        return np.eye(_multipatch_ndof(space))

    def maxwell_dirichlet_dofs(self, space, msh, drchlt_sides):
        nd = _multipatch_ndof(space)
        # 1-based, first half of DOFs (MATLAB convention).
        return np.arange(1, 1 + max(1, nd // 2), dtype=int)

    def maxwell_eigsolve(self, stiff_mat, mass_mat, int_dofs, forceAll,
                         maxNumEigs, eigvTarget):
        A = np.asarray(stiff_mat, dtype=float)
        M = np.asarray(mass_mat, dtype=float)
        A_sub = A[np.ix_(int_dofs, int_dofs)]
        M_sub = M[np.ix_(int_dofs, int_dofs)]
        ndof = A.shape[0]

        if forceAll:
            # MATLAB: [eigf, eigv] = eig(full(A), full(M), 'vector')
            if not _HAS_SCIPY:  # pragma: no cover
                raise RuntimeError("scipy is required for the forceAll branch")
            w, v = scipy.linalg.eig(A_sub, M_sub)
            # 'vector': unit 2-norm eigenvectors
            norms = np.linalg.norm(v, axis=0)
            norms[norms == 0.0] = 1.0
            v = v / norms
        else:
            # MATLAB: eigs(A', M', maxNumEigs, eigvTarget) -- eigenvalues
            # closest to eigvTarget, then sorted ascending (MATLAB side).
            if not _HAS_SCIPY:  # pragma: no cover
                raise RuntimeError("scipy is required for the eigs branch")
            k = max(1, min(int(maxNumEigs), int_dofs.size))
            w, v = scipy.sparse.linalg.eigs(A_sub, k=k, M=M_sub, sigma=eigvTarget)
            order = np.argsort(w.real)
            w = w[order]
            v = v[:, order]

        eigv = np.asarray(w.real, dtype=float)
        eigf = np.zeros((ndof, eigv.size), dtype=float)
        eigf[int_dofs] = np.asarray(v.real, dtype=float)
        return eigf, eigv

    # -- H(curl) / Maxwell path ---------------------------------------------
    def derham(self, knots, degree, mode):
        n = len(knots) if isinstance(knots, (list, tuple)) else 3
        knots_hcurl = [np.asarray(knots[i % len(knots)], dtype=float)
                       for i in range(n)]
        degree_hcurl = [int(degree)] * n
        return knots_hcurl, degree_hcurl

    def hcurl_scalar_space(self, knots_d, degree_d, msh):
        return _MockSpace(self.ndof_e, self.rdim)

    def hcurl_vector_space(self, scalar_spaces, msh, mode):
        m = _MockSpace(self.ndof_e, self.rdim)
        m.kind = "maxwell"
        return m

    def stiffness_maxwell(self, space, msh, invmu):
        """Diagonal matrix diag(1..nd): with an identity mass matrix the
        generalized eigenvalues equal the diagonal entries, so hand-computed
        expectations are possible in tests."""
        nd = _multipatch_ndof(space)
        return np.diag(np.arange(1, nd + 1, dtype=float))

    def mass_maxwell(self, space, msh, c_elec_perm):
        return np.eye(_multipatch_ndof(space))

    def maxwell_dirichlet_dofs(self, space, msh, drchlt_sides):
        nd = _multipatch_ndof(space)
        # 1-based, first half of DOFs (MATLAB convention).
        return np.arange(1, 1 + max(1, nd // 2), dtype=int)

    def maxwell_eigsolve(self, stiff_mat, mass_mat, int_dofs, forceAll,
                         maxNumEigs, eigvTarget):
        A = np.asarray(stiff_mat, dtype=float)
        M = np.asarray(mass_mat, dtype=float)
        A_sub = A[np.ix_(int_dofs, int_dofs)]
        M_sub = M[np.ix_(int_dofs, int_dofs)]
        ndof = A.shape[0]

        if forceAll:
            # MATLAB: [eigf, eigv] = eig(full(A), full(M), 'vector')
            if not _HAS_SCIPY:  # pragma: no cover
                raise RuntimeError("scipy is required for the forceAll branch")
            w, v = scipy.linalg.eig(A_sub, M_sub)
            # 'vector': unit 2-norm eigenvectors
            norms = np.linalg.norm(v, axis=0)
            norms[norms == 0.0] = 1.0
            v = v / norms
        else:
            # MATLAB: eigs(A', M', maxNumEigs, eigvTarget) -- eigenvalues
            # closest to eigvTarget, then sorted ascending (MATLAB side).
            if not _HAS_SCIPY:  # pragma: no cover
                raise RuntimeError("scipy is required for the eigs branch")
            k = max(1, min(int(maxNumEigs), int_dofs.size))
            w, v = scipy.sparse.linalg.eigs(A_sub, k=k, M=M_sub, sigma=eigvTarget)
            order = np.argsort(w.real)
            w = w[order]
            v = v[:, order]

        eigv = np.asarray(w.real, dtype=float)
        eigf = np.zeros((ndof, eigv.size), dtype=float)
        eigf[int_dofs] = np.asarray(v.real, dtype=float)
        return eigf, eigv
