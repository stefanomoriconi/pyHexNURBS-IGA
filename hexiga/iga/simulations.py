"""
hexiga.iga.simulations
======================

Faithful Python ports of the three ``run*Sim_GeoPDEsAPI.m`` orchestration
wrappers that sit above the converters and solvers.

MATLAB ground-truth sources (``iga_utils/``):

* ``runFluidSim_GeoPDEsAPI.m``      -> :func:`run_fluid_simulation`
* ``runElastSim_GeoPDEsAPI.m``      -> :func:`run_elastic_simulation`
* ``runMaxwellEigSim_GeoPDEsAPI.m`` -> :func:`run_maxwell_simulation`

The MATLAB function names reference the GeoPDEs API surface; per project
policy the Python names do **not** reference GeoPDEs (the FEM backend is
isolated behind :class:`hexiga.iga.backend.GeoPDEsBackend`).

Each wrapper:

* builds ``problem_data`` / ``method_data`` with FIXED example parameters
  (literal port of the MATLAB constants);
* calls the matching ``mp_solve_*`` solver;
* post-processes (Maxwell: zero-eigenvalue fallback prompt);
* fills the DATA dataclass in-place and returns it.

MATLAB ``try/catch`` becomes ``try/except``: on failure the MATLAB code
prints ``FAILED!`` and returns the DATA unchanged; the Python port mirrors
that (prints the same banner, re-raises nothing, returns the input).

Copyright header of the MATLAB sources is preserved in
:data:`__MATLAB_LICENSE__`.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .dataclasses import ElasticData, FluidData, MaxwellData
from .stokes import mp_solve_stokes_complete
from .elasticity import mp_solve_linear_elasticity_complete
from .maxwell import mp_solve_maxwell_eig_complete
from .backend import GeoPDEsBackend

__all__ = [
    "run_fluid_simulation",
    "run_elastic_simulation",
    "run_maxwell_simulation",
    "mp_stats",
]

__MATLAB_LICENSE__ = (
    "Copyright (C) 2009, 2010 Carlo de Falco\n"
    "Copyright (C) 2010, 2011, 2015 Rafael Vazquez\n"
    "Few changes introduced by Stefano Moriconi -- Nov. 2021\n"
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ids_count(row) -> int:
    """MATLAB ``length(row.IDs)`` -- 0 when ``None``/empty."""
    if row is None or getattr(row, "IDs", None) is None:
        return 0
    ids = np.atleast_1d(np.asarray(row.IDs)).ravel()
    return int(ids.size)

def _make_f(VolumetricSource) -> Callable:
    """MATLAB: ``problem_data.f = @(x,y,z) cat(1, fx(x,y,z), fy(x,y,z), fz(x,y,z))``
    with ``fx = f .* fx_const .* ones(size(x))`` and analogous for y, z.
    """
    if VolumetricSource is None:
        fx0 = fy0 = fz0 = f0 = 0.0
    else:
        f0 = float(getattr(VolumetricSource, "f", 0.0))
        fx0 = float(getattr(VolumetricSource, "fx", 0.0))
        fy0 = float(getattr(VolumetricSource, "fy", 0.0))
        fz0 = float(getattr(VolumetricSource, "fz", 0.0))

    vx = f0 * fx0
    vy = f0 * fy0
    vz = f0 * fz0

    def _f(x, y=None, z=None):
        shape = np.shape(x) if x is not None else ()
        out = np.zeros((3,) + shape, dtype=float)
        out[0] = vx
        out[1] = vy
        out[2] = vz
        return out

    return _f

def _build_bc_closures(BoundaryConditions, drchlt_sides, nmnn_sides):
    """Port of the MATLAB ``hx / hy / hz`` and ``gx / gy / gz`` struct-of-closures
    loops, returning ``(h, g)`` callables where ``h(x,y,z,iside)`` and
    ``g(x,y,z,iside)`` return 3-component fields.

    MATLAB indexing semantics preserved:

    * ``drchlt_sides`` -- ``[dirichlet.IDs, leading.IDs]`` concatenated.
      For positions ``ds <= length(dirichlet.IDs)`` the value is ZERO;
      for positions ``ds >  length(dirichlet.IDs)`` the value is
      ``leading.m(ds-ofst) .* leading.m{ x, y, z }(ds-ofst)`` with
      ``ofst = length(dirichlet.IDs)``.
    * ``nmnn_sides`` -- exactly ``leading.IDs`` (1-based).  For the ``ns``-th
      leading side (``ns`` 1-based), the value is
      ``leading.m(ns) .* leading.m{ x, y, z }(ns)``.

    The closures are **constant in (x, y, z)** (the MATLAB code multiplies
    a scalar by ``ones(size(x))``), matching the mock backend's expected
    behaviour.
    """
    n_dir = _ids_count(getattr(BoundaryConditions, "dirichlet", None))
    n_lead = _ids_count(getattr(BoundaryConditions, "leading", None))

    dr_ids = np.atleast_1d(np.asarray(drchlt_sides, dtype=int)).ravel() \
        if drchlt_sides is not None else np.empty(0, dtype=int)
    nm_ids = np.atleast_1d(np.asarray(nmnn_sides, dtype=int)).ravel() \
        if nmnn_sides is not None else np.empty(0, dtype=int)

    # Precompute per-side values (Python 1-based side id -> (vx, vy, vz)).
    h_values = {}
    lead = getattr(BoundaryConditions, "leading", None)
    for i, ds in enumerate(dr_ids):
        if i < n_dir:
            h_values[int(ds)] = (0.0, 0.0, 0.0)
        else:
            ofst = i - n_dir  # 0-based index into leading row
            m = _lead_val(lead, "m", ofst)
            mx = _lead_val(lead, "mx", ofst)
            my = _lead_val(lead, "my", ofst)
            mz = _lead_val(lead, "mz", ofst)
            h_values[int(ds)] = (m * mx, m * my, m * mz)

    g_values = {}
    for ns1, iside in enumerate(nm_ids):  # ns1 0-based == MATLAB ns 1-based - 1
        m = _lead_val(lead, "m", ns1)
        mx = _lead_val(lead, "mx", ns1)
        my = _lead_val(lead, "my", ns1)
        mz = _lead_val(lead, "mz", ns1)
        g_values[int(iside)] = (m * mx, m * my, m * mz)

    def _h(x, y=None, z=None, iside=None):
        out = np.zeros((3,) + (np.shape(x) if x is not None else ()), dtype=float)
        if iside is None:
            return out
        v = h_values.get(int(iside), (0.0, 0.0, 0.0))
        out[0] = v[0]; out[1] = v[1]; out[2] = v[2]
        return out

    def _g(x, y=None, z=None, iside=None):
        out = np.zeros((3,) + (np.shape(x) if x is not None else ()), dtype=float)
        if iside is None:
            return out
        v = g_values.get(int(iside), (0.0, 0.0, 0.0))
        out[0] = v[0]; out[1] = v[1]; out[2] = v[2]
        return out

    return _h, _g

def _lead_val(row, name: str, idx: int) -> float:
    """Read row.<name>[idx+1] (MATLAB 1-based) as a scalar; 0.0 if absent."""
    if row is None:
        return 0.0
    arr = getattr(row, name, None)
    if arr is None:
        return 0.0
    a = np.atleast_1d(np.asarray(arr, dtype=float)).ravel()
    if a.size == 0:
        return 0.0
    i = int(idx)
    if i < 0 or i >= a.size:
        return 0.0
    return float(a[i])

def mp_stats(geometry):
    """Port of the MATLAB ``[numH, numP] = mpStats([geometry(:).nurbs])``.

    Counts control-point count (``numH``) and element count (``numP``)
    across the multipatch geometry.  The exact GeoPDEs semantics are
    implementation-specific; this helper returns ``0, 0`` for ``None``
    geometry and inspects ``geometry`` for a ``.nurbs`` field when present.
    """
    if geometry is None:
        return 0, 0
    nurbs = getattr(geometry, "nurbs", None)
    if nurbs is None:
        # geometry may itself be a list/tuple of patch objects (the mock
        # returns a 2-tuple ("multipatch", cells, ...)).
        if isinstance(geometry, (list, tuple)) and len(geometry) >= 2:
            nurbs = geometry[1]
        else:
            return 0, 0
    try:
        cells = np.atleast_1d(np.asarray(nurbs)).ravel()
    except Exception:
        return 0, 0
    numH = numP = 0
    for c in cells:
        n_cpts = getattr(c, "ncp", None)
        n_el = getattr(c, "nel", None)
        if n_cpts is not None:
            numH += int(n_cpts)
        if n_el is not None:
            numP += int(n_el)
    return numH, numP

# ---------------------------------------------------------------------------
# Fluid (Stokes) wrapper
# ---------------------------------------------------------------------------

def run_fluid_simulation(FluidDATA: FluidDATA,
                         backend: Optional[GeoPDEsBackend] = None) -> FluidData:
    """Port of ``runFluidSim_GeoPDEsAPI.m``.

    See the MATLAB source for the exact input/output field contract.  The
    MATLAB wrapper's ``try/catch`` becomes ``try/except``: on failure the
    ``FAILED!`` banner is printed and ``FluidDATA`` is returned unchanged
    (the MATLAB code does NOT re-raise; it just ``catch``es).
    """
    try:
        print(" *** Stokes Fluid Simulation: API (GeoPDEs) ***")

        # --- 1) PHYSICAL DATA OF THE PROBLEM -------------------------------
        class P:
            pass

        problem_data = P()
        problem_data.geo_name = FluidDATA.GeomFileName

        B = FluidDATA.BoundaryConditions
        n_dir = _ids_count(getattr(B, "dirichlet", None)) if B is not None else 0
        n_lead = _ids_count(getattr(B, "leading", None)) if B is not None else 0

        dir_ids = np.atleast_1d(np.asarray(
            getattr(B, "dirichlet", None).IDs if B is not None else None,
            dtype=int)).ravel() if n_dir else np.empty(0, dtype=int)
        lead_ids = np.atleast_1d(np.asarray(
            getattr(B, "leading", None).IDs if B is not None else None,
            dtype=int)).ravel() if n_lead else np.empty(0, dtype=int)

        # MATLAB: drchlt = [dirichlet.IDs, leading.IDs]; nmnn = leading.IDs.
        problem_data.drchlt_sides = np.concatenate([dir_ids, lead_ids]) \
            if (dir_ids.size + lead_ids.size) else np.empty(0, dtype=int)
        problem_data.nmnn_sides = lead_ids

        # MATLAB: viscosity = @(x,y,z) ones(size(x));
        def _viscosity(x, y=None, z=None):
            shape = np.shape(x) if x is not None else ()
            return np.ones(shape, dtype=float)

        problem_data.viscosity = _viscosity

        # MATLAB: f = cat(1, f.*fx.*ones, f.*fy.*ones, f.*fz.*ones)
        problem_data.f = _make_f(FluidDATA.VolumetricSource)

        # MATLAB: h/g closures (Dirichlet ZERO + LEADING, Neumann LEADING).
        h, g = _build_bc_closures(B, problem_data.drchlt_sides, problem_data.nmnn_sides)
        problem_data.h = h
        problem_data.g = g

        # --- 2) DISCRETIZATION ---------------------------------------------
        class M:
            pass

        method_data = M()
        method_data.element_name = "TH"
        method_data.degree = [2, 2, 2]
        method_data.regularity = [1, 1, 1]
        method_data.nsub = [1, 1, 1]
        method_data.nquad = [4, 4, 4]

        # --- 3) SOLVER -------------------------------------------------------
        geometry, _msh, space_v, vel, space_p, press, elpsTime, memFtprt = \
            mp_solve_stokes_complete(problem_data, method_data, backend=backend)

        numH, numP = mp_stats(geometry)

        print(" *** Stokes Fluid Simulation: Completed! ***")
        print("     Elements: " + str(numH))
        print("     Ctrl-Pts: " + str(numP))
        print("     Time: %.2f s " % elpsTime)
        print("     Memory: %.2f MB " % memFtprt)

        # --- 4) RETURN -------------------------------------------------------
        FluidDATA.geometry = geometry
        FluidDATA.space_v = space_v
        FluidDATA.vel = vel
        FluidDATA.space_p = space_p
        FluidDATA.press = press
        FluidDATA.elpsTime = elpsTime
        FluidDATA.memFtprt = memFtprt

    except Exception:
        print(" *** Stokes Fluid Simulation: FAILED! -- Please Check Manually! ***")

    return FluidDATA

# ---------------------------------------------------------------------------
# Elasticity wrapper
# ---------------------------------------------------------------------------

def run_elastic_simulation(ElasticDATA: ElasticData,
                           backend: Optional[GeoPDEsBackend] = None) -> ElasticData:
    """Port of ``runElastSim_GeoPDEsAPI.m``.

    Note the MATLAB quirk: if ``dirichlet.IDs`` is empty, ``drchlt_sides``
    falls back to ``leading.IDs`` (so the *only* Dirichlet sides are the
    leading ones, treated as non-zero first-type BCs); otherwise
    ``drchlt_sides = dirichlet.IDs`` only (leading sides appear in
    ``nmnn_sides`` only).
    """
    try:
        print(" *** Linear Elasticity Simulation: API (GeoPDEs) ***")

        class P:
            pass

        problem_data = P()
        problem_data.geo_name = ElasticDATA.GeomFileName

        B = ElasticDATA.BoundaryConditions
        n_dir = _ids_count(getattr(B, "dirichlet", None)) if B is not None else 0
        n_lead = _ids_count(getattr(B, "leading", None)) if B is not None else 0

        dir_ids = np.atleast_1d(np.asarray(
            getattr(B, "dirichlet", None).IDs if B is not None else None,
            dtype=int)).ravel() if n_dir else np.empty(0, dtype=int)
        lead_ids = np.atleast_1d(np.asarray(
            getattr(B, "leading", None).IDs if B is not None else None,
            dtype=int)).ravel() if n_lead else np.empty(0, dtype=int)

        # MATLAB: if isempty(dirichlet.IDs) drchlt = leading.IDs
        #         else                      drchlt = dirichlet.IDs
        if n_dir == 0:
            problem_data.drchlt_sides = lead_ids
        else:
            problem_data.drchlt_sides = dir_ids
        problem_data.nmnn_sides = lead_ids

        # MATLAB: E = 1; nu = 0.3; lambda = nu*E/((1+nu)*(1-2*nu));
        #                          mu     = E/(2*(1+nu));
        E = 1.0
        nu = 0.3
        lam = nu * E / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        def _lam_fn(x, y=None, z=None):
            shape = np.shape(x) if x is not None else ()
            return np.full(shape, lam, dtype=float)

        def _mu_fn(x, y=None, z=None):
            shape = np.shape(x) if x is not None else ()
            return np.full(shape, mu, dtype=float)

        problem_data.lambda_lame = _lam_fn
        problem_data.mu_lame = _mu_fn

        problem_data.f = _make_f(ElasticDATA.VolumetricSource)

        h, g = _build_bc_closures(B, problem_data.drchlt_sides, problem_data.nmnn_sides)
        problem_data.h = h
        problem_data.g = g

        # --- 2) DISCRETIZATION ---------------------------------------------
        class M:
            pass

        method_data = M()
        # MATLAB has element_name commented out for elasticity; kept as None.
        method_data.element_name = None
        method_data.degree = [3, 3, 3]
        method_data.regularity = [2, 2, 2]
        method_data.nsub = [1, 1, 1]
        method_data.nquad = [4, 4, 4]

        # --- 3) SOLVER -------------------------------------------------------
        geometry, _msh, space, u, elpsTime, memFtprt = \
            mp_solve_linear_elasticity_complete(problem_data, method_data,
                                                 backend=backend)

        numH, numP = mp_stats(geometry)

        print(" *** Linear Elasticity Simulation: Completed! ***")
        print("     Elements: " + str(numH))
        print("     Ctrl-Pts: " + str(numP))
        print("     Time: %.2f s " % elpsTime)
        print("     Memory: %.2f MB " % memFtprt)

        # --- 4) RETURN -------------------------------------------------------
        ElasticDATA.geometry = geometry
        ElasticDATA.space = space
        ElasticDATA.u = u
        ElasticDATA.lambda_lame = problem_data.lambda_lame
        ElasticDATA.mu_lame = problem_data.mu_lame
        ElasticDATA.elpsTime = elpsTime
        ElasticDATA.memFtprt = memFtprt

    except Exception:
        print(" *** Linear Elasticity Simulation: FAILED! -- Please Check Manually! ***")

    return ElasticDATA

# ---------------------------------------------------------------------------
# Maxwell eigen wrapper
# ---------------------------------------------------------------------------

def run_maxwell_simulation(MaxwellDATA: MaxwellData,
                           backend: Optional[GeoPDEsBackend] = None,
                           force_all_prompt: Optional[Callable[[str], str]] = None
                           ) -> MaxwellData:
    """Port of ``runMaxwellEigSim_GeoPDEsAPI.m``.

    The MATLAB wrapper ends with a ``questdlg`` prompt when no non-zero
    eigenvalue was found.  Because the Python port is meant to run
    headless, the interactive prompt is abstracted behind the
    ``force_all_prompt`` callable: ``prompt(message) -> 'Yes' | 'No'``.
    Default: always answer ``'Yes'`` (so the forced re-solve happens
    automatically, matching the MATLAB ``Yes`` branch).  Pass a custom
    callable (e.g. ``input``-based) to make the prompt interactive.
    """
    try:
        print(" *** Maxwell EigenFunction Simulation: API (GeoPDEs) ***")

        class P:
            pass

        problem_data = P()
        problem_data.geo_name = MaxwellDATA.GeomFileName

        B = MaxwellDATA.BoundaryConditions
        # MATLAB: drchlt = 1 : ( #dirichlet + #leading + #floating ); nmnn = [].
        # "ALL sides Dirichlet" -- the MATLAB range is over the *count* of
        # sides, so we reproduce with the same count.
        n_total = 0
        if B is not None:
            n_total = (_ids_count(getattr(B, "dirichlet", None))
                       + _ids_count(getattr(B, "leading", None))
                       + _ids_count(getattr(B, "floating", None)))
        problem_data.drchlt_sides = np.arange(1, n_total + 1, dtype=int) \
            if n_total else np.empty(0, dtype=int)
        problem_data.nmnn_sides = np.empty(0, dtype=int)

        # MATLAB: c_elec_perm = c_magn_perm = @(x,y,z) ones(size(x));
        def _ones(x, y=None, z=None):
            shape = np.shape(x) if x is not None else ()
            return np.ones(shape, dtype=float)

        problem_data.c_elec_perm = _ones
        problem_data.c_magn_perm = _ones

        # MATLAB: forceAll = false (initially).
        problem_data.forceAll = False

        # --- 2) DISCRETIZATION ---------------------------------------------
        class M:
            pass

        method_data = M()
        # MATLAB has element_name commented out for Maxwell.
        method_data.element_name = None
        method_data.degree = [2, 2, 2]
        method_data.regularity = [1, 1, 1]
        method_data.nsub = [1, 1, 1]
        method_data.nquad = [3, 3, 3]

        # --- 3) SOLVER -------------------------------------------------------
        geometry, _msh, space, eigv, eigf, elpsTime, memFtprt = \
            mp_solve_maxwell_eig_complete(problem_data, method_data,
                                          backend=backend)

        # --- 4) POSTPROCESSING ---------------------------------------------
        temp_eigv = np.sort(np.asarray(eigv, dtype=float))
        nzeros = int(np.sum(np.asarray(temp_eigv < 1e-10, dtype=int)))

        if (temp_eigv.size - nzeros) < 1:
            msg = "Do You Want to Force All Solutions [This may be long]?"
            if force_all_prompt is None:
                answer = "Yes"
            else:
                answer = force_all_prompt(msg)
            if str(answer) == "Yes":
                problem_data.forceAll = True
                geometry, _msh, space, eigv, eigf, elpsTime, memFtprt = \
                    mp_solve_maxwell_eig_complete(problem_data, method_data,
                                                   backend=backend)

        numH, numP = mp_stats(geometry)

        print(" *** Maxwell Eigenfunction Simulation: Completed! ***")
        print("     Elements: " + str(numH))
        print("     Ctrl-Pts: " + str(numP))
        print("     Time: %.2f s " % elpsTime)
        print("     Memory: %.2f MB " % memFtprt)

        # --- 5) RETURN -------------------------------------------------------
        MaxwellDATA.geometry = geometry
        MaxwellDATA.space = space
        MaxwellDATA.eigv = eigv
        MaxwellDATA.eigf = eigf
        MaxwellDATA.elpsTime = elpsTime
        MaxwellDATA.memFtprt = memFtprt

    except Exception:
        print(" *** Maxwell Eigenfunction Simulation: FAILED! -- Please Check Manually! ***")

    return MaxwellDATA
