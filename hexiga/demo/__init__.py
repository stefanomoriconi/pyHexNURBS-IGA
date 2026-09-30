"""
hexiga.demo
===========

Python port of HexIGA's ``demo_utils`` -- the synthetic (``Synthetic``)
and CAD (``CAD``) demo geometries used to exercise the IGA solver stack.
The CAD demos (milestone A3, e.g. ``demoSocketCAD``) build linear
hexahedral multi-patch assemblies and degree-elevate them to cubic.

The public surface is intentionally minimal: each ``demo*`` function
returns a list of :class:`hexiga.nurbs.nrb.Nrb` patches (a multi-patch
geometry), and the ``smooth=True`` path (which requires the
``mpTurboSmooth`` core, ported under milestone A4) is preserved as an
explicit option that raises :class:`NotImplementedError` until A4 lands.
"""

from .make_cuboid import makeCuboid
from .demo_torus import demoTorus
from .demo_quadball import demoQuadball
from .demo_egg_lw import demoEggLW
from .demo_frame import demoFrame
from .demo_cross6 import demoCross6
from .demo_twist import demoTwist
from .demo_pinocchio import demoPinocchio
from .demo_socket import demoSocketCAD, genSocket
from .demo_hooks import demoHooksCAD, genHooks
from .demo_stent import demoStentCAD, genStent
from .demo_gear import demoGearCAD, genGear
from .demo_turbine import demoTurbineCAD, genTurbineDomains
from .demo_plate import demoPlateCAD, genPlate
from .demo_tpipe import demoTPipeCAD, genTPipe

__all__ = [
    "makeCuboid",
    "demoTorus",
    "demoQuadball",
    "demoEggLW",
    "demoFrame",
    "demoCross6",
    "demoTwist",
    "demoPinocchio",
    "demoSocketCAD",
    "genSocket",
    "demoHooksCAD",
    "genHooks",
    "demoStentCAD",
    "genStent",
    "demoGearCAD",
    "genGear",
    "demoTurbineCAD",
    "genTurbineDomains",
    "demoPlateCAD",
    "genPlate",
    "demoTPipeCAD",
    "genTPipe",
]
