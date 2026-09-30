"""
hexiga.gui
==========

Tkinter-based GUI applications that mirror the MATLAB ``MAIN_gui`` apps.

Each app class is **constructable without showing** (call :meth:`build`)
so that headless smoke tests can instantiate the widgets and exercise the
state machine without invoking :meth:`mainloop`.

Apps
----
* :class:`app_demo.DemoSynthCAD`  ←  ``MAIN_gui/DemoSynthCAD.m``
* :class:`app_njunction.NJunction`  ←  ``MAIN_gui/NJunction.m``
* :class:`app_sim.HexIGASim`  ←  ``MAIN_gui/HexIGASim.m``

Launch from the command line::

    python -m hexiga.gui.app_demo
    python -m hexiga.gui.app_njunction
    python -m hexiga.gui.app_sim

The Tk app classes are exposed lazily (via :data:`__getattr__`) so that the
package stays importable on headless systems without a Tk runtime; the
headless core functions (import / attribute / validation helpers) are always
importable.
"""

__all__ = [
    "GeomDATA",
    "gen_boundaries_attributes",
    "gen_arrow_boundary_handles",
    "chk_valid_bc_vol_source",
    "import_geometry",
    "save_boundaries",
    "import_boundaries",
    "run_sim",
    "DemoSynthCAD",
    "NJunction",
    "HexIGASim",
]

# Every public symbol is exposed **lazily** (via :data:`__getattr__`) so that
# importing the ``hexiga.gui`` package never eagerly imports a single app
# module. This matters for ``python -m hexiga.gui.app_sim``: runpy imports the
# parent package first; if the package ``__init__`` had already imported
# ``app_sim``, runpy would find it in ``sys.modules`` before executing it as
# ``__main__`` and emit a ``RuntimeWarning`` about unpredictable behaviour.
# Deferring all app imports to first attribute access avoids that entirely.
#
# The headless core of ``app_sim`` (always importable, no Tk required) is the
# only group of *function* symbols; the rest are the optional Tk-backed app
# classes.
_HEADLESS_SIM = (
    "GeomDATA",
    "gen_boundaries_attributes",
    "gen_arrow_boundary_handles",
    "chk_valid_bc_vol_source",
    "import_geometry",
    "save_boundaries",
    "import_boundaries",
    "run_sim",
)

def __getattr__(name):
    # Lazy access to the (optionally Tk-backed) app classes so that importing
    # ``hexiga.gui`` never hard-requires a Tk runtime.
    if name in _HEADLESS_SIM:
        from . import app_sim
        return getattr(app_sim, name)
    if name == "HexIGASim":
        from .app_sim import HexIGASim
        return HexIGASim
    if name == "NJunction":
        from .app_njunction import NJunction
        return NJunction
    if name == "DemoSynthCAD":
        from .app_demo import DemoSynthCAD
        return DemoSynthCAD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | set(__all__))
