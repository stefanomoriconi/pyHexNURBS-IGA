"""``hexiga.smth`` -- multipatch NURBS hexa smoothing.

Public API:
    * :func:`mp_turbo_smooth` -- top-level smoothing driver (port of
      ``mpTurboSmooth.m``).
    * :func:`smth_ctrl_pts_topology_atrbs` -- one smoothing iteration
      (port of ``smthCtrlPtsTopologyAttribs.m``).
    * :func:`pushback_ctrl_pts_topology_to_mp` -- port of
      ``pushbackCtrlPtsTopologyToMultiPatchHexa.m``.
    * :func:`refine_term_knts_mp` -- port of
      ``refineTermKntsMPHexa.m``.
    * :func:`extract_ctrl_pts_topology_from_multipatch_hexa` -- port of
      ``extractCtrlPtsTopologyFromMultiPatchHexa.m``.
    * :func:`append_smth_atrbs_new` -- port of
      ``appendSmthAtrbs_new.m``.
    * :func:`aggregate_multi_patch_hexa` -- port of
      ``aggregateMultiPatchHexa.m``.
    * :class:`CPT`, :class:`CPTAttr`, :class:`Cuboid` -- dataclasses from
      :mod:`hexiga.smth.cpt`.
"""
from __future__ import annotations

from .cpt import CPT, CPTAttr, Cuboid
from .find_ctrl_pt import find_ctrl_pt
from .cuboid import (
    gen_cuboid_grid,
    compute_cuboid_grid_shells,
)
from .aggregate import aggregate_multi_patch_hexa
from .extract_cpt_topology_mp import extract_ctrl_pts_topology_from_multipatch_hexa
from .append_smth_atrbs_new import append_smth_atrbs_new
from .smth_cpt_topology_atrbs import smth_ctrl_pts_topology_atrbs
from .pushback import pushback_ctrl_pts_topology_to_mp
from .refine_term_knts_mp import refine_term_knts_mp
from .mp_turbo_smooth import mp_turbo_smooth, set_progress_hook

__all__ = [
    "CPT",
    "CPTAttr",
    "Cuboid",
    "find_ctrl_pt",
    "gen_cuboid_grid",
    "compute_cuboid_grid_shells",
    "aggregate_multi_patch_hexa",
    "extract_ctrl_pts_topology_from_multipatch_hexa",
    "append_smth_atrbs_new",
    "smth_ctrl_pts_topology_atrbs",
    "pushback_ctrl_pts_topology_to_mp",
    "refine_term_knts_mp",
    "mp_turbo_smooth",
    "set_progress_hook",
]
