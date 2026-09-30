"""
hexiga.gui.app_demo
===================

Python port of ``MAIN_gui/DemoSynthCAD.m`` -- the synthetic/CAD demo
launch pad.

Faithful mapping of the MATLAB app:

======================  ======================================================
MATLAB                  Python
======================  ======================================================
``popupmenu_Demos``     :class:`ttk.Combobox` with the 14 demo entries
``radiobutton_None/``   :class:`ttk.Radiobutton` trio; Stent & Plate have
``Smooth/TurboSmooth``  TurboSmooth disabled (MATLAB has no TURBOSMOOTH
                        branch for cases 10/12)
``pushbutton_Run``      :meth:`DemoSynthCAD.run` -- dispatches the demo
                        driver with ``('SMOOTH', False|True, 'VISUALISE',
                        False)`` or ``('TURBOSMOOTH', True, 'VISUALISE',
                        False)`` exactly as in ``runDemo``
``pushbutton_Reset``    :meth:`DemoSynthCAD.reset` -- confirms, restores
                        default widget values, re-enables config
``pushbutton_ExportNURBS``  :meth:`DemoSynthCAD.export_nurbs` --
                        ``nrbsave`` (single file, or ``_L`` / ``_W`` split
                        when ``idxComposite`` is set)
``pushbutton_ExportSTL``    :meth:`DemoSynthCAD.export_stl` --
                        ``nrb_export_stl`` (same L/W split convention)
viz radiobuttons        5 modes calling the :mod:`hexiga.viz` plot
                        functions in a live (``off_screen=False``) window
license gate            omitted (MATLAB ``chkLicense`` is not ported)
======================  ======================================================

The class is constructable without showing: :meth:`DemoSynthCAD.build`
creates all widgets; :meth:`DemoSynthCAD.mainloop` enters the event loop.
Headless smoke tests call ``build()`` only.

Launch with::

    python -m hexiga.gui.app_demo
"""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
import tkinter.ttk as ttk
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox
from typing import Any, Callable, List, Optional, Sequence, Tuple

from ..nurbs import Nrb, nrbmultipatch, nrbsave
from ..export import nrb_export_stl
from ..demo import (
    demoCross6,
    demoEggLW,
    demoFrame,
    demoPinocchio,
    demoQuadball,
    demoTorus,
    demoTwist,
    demoGearCAD,
    demoHooksCAD,
    demoPlateCAD,
    demoSocketCAD,
    demoStentCAD,
    demoTPipeCAD,
    demoTurbineCAD,
)
from ..viz import (
    plot_solid_domain,
    plot_ctrl_pts,
    plot_exploded_param,
    plot_reflection_lines,
    plot_trabecular,
)
from ..smth import set_progress_hook

__all__ = ["DemoData", "DemoSynthCAD", "DEMO_NAMES", "main"]

# ---------------------------------------------------------------------------
# Demo catalogue (1-based in MATLAB -> 0-based entries here)
# ---------------------------------------------------------------------------
#: (label, driver module path, driver name, returns_tuple, has_turbo, merge)
DEMO_NAMES: List[str] = [
    "Cross",      #  1  Synthetic
    "Egg",        #  2  Synthetic  (idxComposite + GhostFLAG [-1,-1])
    "Frame",      #  3  Synthetic
    "Pinocchio",  #  4  Synthetic
    "Quadball",   #  5  Synthetic
    "Torus",      #  6  Synthetic  (MERGE option)
    "Twist",      #  7  Synthetic
    "Gear",       #  8  CAD
    "Hooks",      #  9  CAD
    "Plate",      # 10  CAD        (no TURBOSMOOTH branch in MATLAB)
    "Socket",     # 11  CAD
    "Stent",      # 12  CAD        (no TURBOSMOOTH branch in MATLAB)
    "T-Pipe",     # 13  CAD        (MERGE option)
    "Turbine",    # 14  CAD        (idxComposite + GhostFLAG [0,1])
]

#: Driver registry, 1-based index == MATLAB ``selValue``.
#: ``fn``        -- the :mod:`hexiga.demo` driver (callable).
#: ``merge``     -- driver takes a ``('MERGE', ...)`` option (Torus, T-Pipe).
#: ``turbo``     -- MATLAB has no TURBOSMOOTH branch for Plate / Stent.
#: ``returns2``  -- driver returns ``(partA, partB)`` (Egg, Turbine).
_DEMOS = {
    1:  {"label": "Cross",     "fn": demoCross6,
         "merge": False, "turbo": True},
    2:  {"label": "Egg",       "fn": demoEggLW,
         "merge": False, "turbo": True, "returns2": True},
    3:  {"label": "Frame",     "fn": demoFrame,
         "merge": False, "turbo": True},
    4:  {"label": "Pinocchio", "fn": demoPinocchio,
         "merge": False, "turbo": True},
    5:  {"label": "Quadball",  "fn": demoQuadball,
         "merge": False, "turbo": True},
    6:  {"label": "Torus",     "fn": demoTorus,
         "merge": True,  "turbo": True},
    7:  {"label": "Twist",     "fn": demoTwist,
         "merge": False, "turbo": True},
    8:  {"label": "Gear",      "fn": demoGearCAD,
         "merge": False, "turbo": True},
    9:  {"label": "Hooks",     "fn": demoHooksCAD,
         "merge": False, "turbo": True},
    10: {"label": "Plate",     "fn": demoPlateCAD,
         "merge": False, "turbo": False},
    11: {"label": "Socket",    "fn": demoSocketCAD,
         "merge": False, "turbo": True},
    12: {"label": "Stent",     "fn": demoStentCAD,
         "merge": False, "turbo": False},
    13: {"label": "T-Pipe",    "fn": demoTPipeCAD,
         "merge": True,  "turbo": True},
    14: {"label": "Turbine",   "fn": demoTurbineCAD,
         "merge": False, "turbo": True, "returns2": True},
}

#: Visualisation modes (order == MATLAB radiobuttons).
VIZ_MODES = [
    "Solid Domain",
    "Control Points",
    "Exploded Parametric",
    "Reflection Lines",
    "Trabecular",
]

# ---------------------------------------------------------------------------
# DATA container  (mirrors MATLAB ``initDemoDATA``)
# ---------------------------------------------------------------------------
@dataclass
class DemoData:
    """Mirror of the MATLAB ``DemoDATA`` struct."""

    Name: str = ""
    mpHexa: List[Nrb] = field(default_factory=list)
    idxComposite: Optional[int] = None
    Except: Any = None
    GhostFLAG: List[int] = field(default_factory=list)
    SmthLevel: Optional[int] = None  # 0=None  1=Smooth  2=TurboSmooth

    def is_empty(self) -> bool:
        return (
            not self.Name
            and len(self.mpHexa) == 0
            and self.idxComposite is None
            and self.Except is None
            and len(self.GhostFLAG) == 0
            and self.SmthLevel is None
        )

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class DemoSynthCAD:
    """Tkinter port of ``MAIN_gui/DemoSynthCAD.m``.

    The widget layout mirrors the MATLAB figure: a config panel (demo
    selection + smoothing), a run/reset control row, an export panel, and a
    visualisation panel.  The state machine mirrors the MATLAB
    ``disable_Config`` / ``enable_Export`` / ``enable_VisualisationGroup``
    helpers: after a successful :meth:`run`, config is disabled and export
    is enabled; the visualisation group is enabled only when
    ``DemoDATA.Except`` is empty; :meth:`reset` restores everything.
    """

    def __init__(self, root: Optional[tk.Tk] = None, own_root: bool = True) -> None:
        self.own_root = own_root
        if root is not None:
            self.root = root
        else:
            self.root = tk.Tk()
            self.root.title("HexIGA -- DemoSynthCAD (Python)")
            self.root.geometry("560x520")

        self.DemoDATA = DemoData()
        self._viz_plotter = None  # live pyvista window of the last plot

        self.smth_var = tk.IntVar(value=0)
        self.demo_var = tk.StringVar(value=DEMO_NAMES[0])
        self.viz_var = tk.StringVar(value=VIZ_MODES[0])

        self.config_widgets: List[Any] = []
        self.export_widgets: List[Any] = []
        self.viz_widgets: List[Any] = []

        # Thread-safe hand-off from the worker thread back to the Tk main
        # loop.  Tk is not thread-safe, so the worker enqueues result
        # payloads and the main thread drains the queue via ``_poll_results``.
        self._result_queue: "queue.Queue" = queue.Queue()

    # -- construction -------------------------------------------------------
    def build(self) -> "DemoSynthCAD":
        """Create all widgets (no event loop).  Idempotent-ish: safe to call
        once per construction for headless smoke tests."""
        base = tk.Frame(self.root)
        base.pack(fill="both", expand=True, padx=10, pady=10)

        # -- Config group ---------------------------------------------------
        cfg = tk.LabelFrame(base, text="Demo Configuration")
        cfg.pack(fill="x", pady=(0, 8))

        row1 = tk.Frame(cfg)
        row1.pack(fill="x", pady=4)
        tk.Label(row1, text="Demo:").pack(side="left")
        cb = ttk.Combobox(row1, textvariable=self.demo_var,
                          values=DEMO_NAMES, state="readonly", width=18)
        cb.pack(side="left", padx=6)
        self.demo_combo = cb
        self.config_widgets.append(cb)

        row2 = tk.Frame(cfg)
        row2.pack(fill="x", pady=4)
        self.rb_none = tk.Radiobutton(row2, text="None", value=0,
                                      variable=self.smth_var,
                                      command=self._on_smth_change)
        self.rb_smooth = tk.Radiobutton(row2, text="Smooth", value=1,
                                        variable=self.smth_var,
                                        command=self._on_smth_change)
        self.rb_turbo = tk.Radiobutton(row2, text="TurboSmooth", value=2,
                                       variable=self.smth_var,
                                       command=self._on_smth_change)
        self.rb_none.pack(side="left", padx=(0, 12))
        self.rb_smooth.pack(side="left", padx=(0, 12))
        self.rb_turbo.pack(side="left")
        self.config_widgets.extend([self.rb_none, self.rb_smooth, self.rb_turbo])

        self._on_smth_change()

        # -- Run / Reset row ------------------------------------------------
        runrow = tk.Frame(base)
        runrow.pack(fill="x", pady=8)
        self.btn_run = tk.Button(runrow, text="Run", command=self.run)
        self.btn_reset = tk.Button(runrow, text="Reset", command=self.reset)
        self.btn_run.pack(side="left", padx=(0, 8))
        self.btn_reset.pack(side="left")
        # NOTE: btn_reset is intentionally NOT added to config_widgets so that
        # it stays enabled at all times -- the user must always be able to
        # start fresh, even after a successful Run.  Only the Run button and
        # the configuration widgets get disabled once a geometry exists.
        self.config_widgets.append(self.btn_run)
        self._running = False  # guard against double-launch while a thread runs

        # -- Export group ---------------------------------------------------
        exp = tk.LabelFrame(base, text="Export")
        exp.pack(fill="x", pady=(0, 8))
        self.btn_exp_nurbs = tk.Button(exp, text="Export NURBS (.txt)",
                                       command=self.export_nurbs,
                                       state="disabled")
        self.btn_exp_stl = tk.Button(exp, text="Export STL (.stl)",
                                     command=self.export_stl,
                                     state="disabled")
        self.btn_exp_nurbs.pack(side="left", padx=(0, 8))
        self.btn_exp_stl.pack(side="left")
        self.export_widgets = [self.btn_exp_nurbs, self.btn_exp_stl]

        # -- Visualisation group ---------------------------------------------
        viz = tk.LabelFrame(base, text="Visualisation")
        viz.pack(fill="x", pady=(0, 8))
        for i, name in enumerate(VIZ_MODES):
            rb = tk.Radiobutton(viz, text=name, value=name,
                                variable=self.viz_var,
                                command=self.plot_current,
                                state="disabled")
            rb.pack(anchor="w")
            self.viz_widgets.append(rb)

        # -- Status line -----------------------------------------------------
        self.status = tk.Label(base, text="Ready.", anchor="w",
                               fg="#005500")
        self.status.pack(fill="x")

        # -- Smoothing progress bar ------------------------------------------
        # Temporary 0->MAXITER progress bar shown while a Run is in flight.
        # Driven from the smoothing driver's progress hook (worker thread)
        # via a queue, marshalled to the Tk main thread in _poll_results.
        progrow = tk.Frame(base)
        progrow.pack(fill="x", pady=(4, 0))
        self.prog_label = tk.Label(progrow, text="", anchor="e", width=24)
        self.prog_label.pack(side="right")
        self.progress = ttk.Progressbar(progrow, mode="determinate",
                                        maximum=100)
        self.progress.pack(side="left", fill="x", expand=True)
        self._progress_queue: "queue.Queue" = queue.Queue()

        # Start the main-thread result poller (drains worker-thread output).
        self._start_result_polling()

        return self

    def _start_result_polling(self) -> None:
        """Re-arm the main-thread result poller every 50 ms.

        ``after`` is only valid from the Tk main thread, so this is called
        from :meth:`build` (main thread) and re-scheduled by
        :meth:`_poll_results`.
        """
        try:
            self.root.after(50, self._poll_results)
        except tk.TclError:
            pass

    # -- helpers ------------------------------------------------------------
    def _sel_value(self) -> int:
        """1-based MATLAB ``selValue`` from the combobox."""
        name = self.demo_var.get()
        return DEMO_NAMES.index(name) + 1

    def _on_smth_change(self) -> None:
        """Disable TurboSmooth for Stent (12) and Plate (10), matching the
        MATLAB GUI which has no TURBOSMOOTH branch for those demos."""
        sel = self._sel_value()
        has_turbo = bool(_DEMOS[sel]["turbo"])
        if not has_turbo and self.smth_var.get() == 2:
            self.smth_var.set(0)
        try:
            self.rb_turbo.config(state="normal" if has_turbo else "disabled")
        except tk.TclError:
            pass

    def _set_config_state(self, state: str) -> None:
        for w in self.config_widgets:
            try:
                w.config(state=state)
            except tk.TclError:
                pass
        # btn_reset is deliberately excluded -- it must remain clickable.

    def _set_export_state(self, state: str) -> None:
        for w in self.export_widgets:
            try:
                w.config(state=state)
            except tk.TclError:
                pass

    def _set_viz_state(self, state: str) -> None:
        for w in self.viz_widgets:
            try:
                w.config(state=state)
            except tk.TclError:
                pass

    def _status(self, msg: str) -> None:
        self.status.config(text=msg)

    # -- smoothing progress -------------------------------------------------
    def _on_smth_progress(self, current: int, total: int) -> None:
        """Smoothing-driver progress hook (runs on the WORKER thread).

        Only enqueues a value -- Tk widget access is NOT thread-safe, so the
        actual :class:`ttk.Progressbar` update happens on the main thread in
        :meth:`_poll_results`.
        """
        try:
            self._progress_queue.put((int(current), int(total)))
        except (RuntimeError, AttributeError):
            pass  # widget / queue destroyed; ignore

    # -- RUN ----------------------------------------------------------------
    def run(self) -> bool:
        """Port of ``runDemo`` + the Run pushbutton callback.

        Runs the demo driver on a background thread so the Tk main loop
        stays responsive during (potentially slow) geometry generation and
        smoothing.  All Tk widget updates are marshalled back to the main
        thread via :meth:`tkinter.Tk.after`.

        Returns ``True`` immediately (the job has been dispatched),
        ``False`` if a run is already in progress or the core fails before
        it can be threaded (e.g. invalid selection).
        """
        if self._running:
            self._status("A run is already in progress -- please wait.")
            return False

        # Capture inputs on the main thread (Tk vars are not thread-safe).
        try:
            selValue = self._sel_value()
            SmthLevel = self.smth_var.get()
            spec = _DEMOS[selValue]
        except (KeyError, ValueError, IndexError) as exc:
            self._status(f"Invalid demo selection: {exc}")
            messagebox.showerror("DemoSynthCAD",
                                 f"Invalid demo selection: {exc}")
            return False

        if SmthLevel == 2 and not spec["turbo"]:
            self._status(f"{spec['label']} does not support TURBOSMOOTH.")
            messagebox.showerror(
                "DemoSynthCAD",
                f"{spec['label']} does not support TURBOSMOOTH.")
            return False

        self._running = True
        self._status(f"Running {spec['label']}...")
        self.btn_run.config(state="disabled")
        # Reset the smoothing progress bar for this run.
        try:
            self.progress.config(value=0, maximum=100)
            self.prog_label.config(text="")
        except tk.TclError:
            pass
        # Register the smoothing-driver progress hook (worker thread).  The
        # hook enqueues (current, total); the main-thread poller applies it.
        self._progress_queue = queue.Queue()
        set_progress_hook(self._on_smth_progress)
        self.root.update_idletasks()

        def _worker():
            try:
                ok, msg = self._run_core(selValue, SmthLevel, spec)
            except Exception as exc:  # defensive: _run_core already catches
                ok, msg = False, f"{type(exc).__name__}: {exc}"
            # Tk is not thread-safe: hand the result to the main thread via
            # a queue, drained by _poll_results() (scheduled with after()).
            self._result_queue.put((ok, msg))

        threading.Thread(target=_worker, daemon=True).start()
        return True

    def _poll_results(self) -> None:
        """Drain the worker-result queue on the Tk main thread.

        Invoked (re-armed) every 50 ms from :meth:`_start_result_polling`
        so that every worker completion is applied on the main thread,
        where Tk widget calls and message boxes are legal.
        """
        # Re-arm first so a busy handler cannot starve subsequent results.
        self._start_result_polling()

        # Apply any smoothing-iteration progress enqueued by the worker
        # thread (thread-safe: we are on the Tk main thread here).
        try:
            while True:
                current, total = self._progress_queue.get_nowait()
                if total > 0:
                    self.progress.config(maximum=total)
                self.progress.config(value=current)
                self.prog_label.config(
                    text=f"smoothing: {current}/{total}")
        except queue.Empty:
            pass
        except (tk.TclError, AttributeError):
            return  # root already destroyed

        while True:
            try:
                ok, msg = self._result_queue.get_nowait()
            except queue.Empty:
                break
            except tk.TclError:
                return  # root already destroyed
            self._finish_run(ok, msg)

    def _finish_run(self, ok: bool, msg: str) -> None:
        """Apply the post-run state machine (main thread only).

        ``self._running`` is cleared last, so callers polling it (and the
        reset guard) only observe the completed state.
        """
        try:
            self.btn_run.config(state="normal")
        except tk.TclError:
            pass
        # Clear the smoothing progress hook (no more worker callbacks).
        set_progress_hook(None)
        # Reset / hide the progress bar now that the run is done.
        try:
            self.progress.config(value=0)
            self.prog_label.config(text="done" if ok else "")
        except tk.TclError:
            pass
        if ok:
            # State machine (MATLAB: disable_Config, enable_Export,
            # enable_VisualisationGroup when Except empty).
            # NOTE: btn_reset is deliberately excluded from
            # _set_config_state so it stays enabled at all times.
            self._set_config_state("disabled")
            self._set_export_state("normal")
            self._set_viz_state("normal")
            self._status(
                f"{self.DemoDATA.Name}: "
                f"{len(self.DemoDATA.mpHexa)} patches "
                f"(SmthLevel={self.DemoDATA.SmthLevel}) -- "
                f"pick a visualisation mode."
            )
        else:
            self._status(f"Error: {msg}")
            messagebox.showerror(
                "DemoSynthCAD",
                "Errors occurred while Running Demo! " + str(msg))
        # Clear the running flag last so observers see the finished state.
        self._running = False

    def _run_core(self, selValue: int, SmthLevel: int,
                  spec: dict) -> Tuple[bool, str]:
        """Blocking demo-driver execution (called on a worker thread).

        Returns ``(True, "")`` on success or ``(False, message)`` on
        failure.  All side effects on :class:`DemoDATA` happen here;
        no Tk calls are made from this method.
        """
        try:
            # MATLAB's runDemo (re)initialises DemoDATA at the start of
            # every run -- clear composite state so a previous Egg /
            # Turbine run cannot leak into a plain demo's export.
            self.DemoDATA.idxComposite = None
            self.DemoDATA.GhostFLAG = []

            fn = spec["fn"]
            if fn is None:
                raise NotImplementedError(
                    f"{spec['label']} is not implemented in the Python "
                    "demo catalogue.")

            if SmthLevel == 0:
                opts: List[Any] = ["SMOOTH", False, "VISUALISE", False]
            elif SmthLevel == 1:
                opts = ["SMOOTH", True, "VISUALISE", False]
            else:
                opts = ["TURBOSMOOTH", True, "VISUALISE", False]

            # NB: MATLAB's runDemo never passes a 'MERGE' option -- the
            # Torus / T-Pipe drivers default to merge=True, so we omit it
            # too (literal fidelity to the ground-truth GUI).
            res = fn(*opts)

            if spec.get("returns2"):
                partA, partB = res
                mpHexa = list(partA) + list(partB)
                self.DemoDATA.idxComposite = len(partA)
                if selValue == 2:   # Egg  -> GhostFLAG [-1,-1]
                    self.DemoDATA.GhostFLAG = [-1, -1]
                elif selValue == 14:  # Turbine -> GhostFLAG [0,1]
                    self.DemoDATA.GhostFLAG = [0, 1]
            else:
                mpHexa = list(res)

            self.DemoDATA.Name = spec["label"]
            self.DemoDATA.mpHexa = mpHexa
            self.DemoDATA.SmthLevel = SmthLevel
            self.DemoDATA.Except = None
            return True, ""
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    # -- RESET --------------------------------------------------------------
    def reset(self) -> None:
        """Port of the Reset pushbutton callback (questdlg + resetRoutine).

        The button is always enabled.  If a run is still in flight on its
        background thread, we wait for it to settle first so it cannot
        overwrite the freshly-reset ``DemoDATA``.
        """
        if messagebox.askyesno(
            "DemoSynthCAD", "Do You Really Want to Reset?"
        ):
            # Let any in-flight worker finish before we clear state.
            while self._running:
                self.root.update()
                time.sleep(0.05)
            self._close_viz()
            self.DemoDATA.reset()
            self.demo_var.set(DEMO_NAMES[0])
            self.smth_var.set(0)
            self.viz_var.set(VIZ_MODES[0])
            self._set_config_state("normal")
            self._set_export_state("disabled")
            self._set_viz_state("disabled")
            self._on_smth_change()
            self._status("Ready.")

    # -- EXPORT -------------------------------------------------------------
    def export_nurbs(self) -> Optional[str]:
        """MATLAB ``pushbutton_ExportNURBS``: save the multi-patch as a
        HexIGA ``.txt`` (v2.1) geometry file via :func:`hexiga.nurbs.nrbsave`.
        """
        initial = (self.DemoDATA.Name or "mpHexa") + ".txt"
        path = filedialog.asksaveasfilename(
            title="Export NURBS",
            defaultextension=".txt",
            initialfile=initial,
            filetypes=[("HexIGA NURBS", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return None

        def _w(patches: List[Nrb], fname: str) -> None:
            ifaces, bnds = nrbmultipatch(patches)
            # nrbmultipatch returns one entry per side; nrbsave expects a
            # named boundary group -- aggregate (default group 'G0').
            if bnds:
                boundary = {
                    "name": "G0",
                    "nsides": len(bnds),
                    "patches": [b["patches"] for b in bnds],
                    "faces": [b["faces"] for b in bnds],
                }
            else:
                boundary = None
            nrbsave(fname, patches, interfaces=ifaces,
                    boundaries=[boundary] if boundary else None,
                    header_comments=[f"HexIGA Python export -- {self.DemoDATA.Name}"])

        try:
            folder = os.path.dirname(path) + os.sep
            base = os.path.basename(path)
            exported: List[str] = []
            if self.DemoDATA.idxComposite:
                idxC = self.DemoDATA.idxComposite
                stem = base[:-4] if base.endswith(".txt") else base
                pL = folder + stem + "_L.txt"
                pW = folder + stem + "_W.txt"
                _w(list(self.DemoDATA.mpHexa[:idxC]), pL)
                _w(list(self.DemoDATA.mpHexa[idxC:]), pW)
                exported = [pL, pW]
            else:
                _w(list(self.DemoDATA.mpHexa), path)
                exported = [path]
            self._status("NURBS exported.")
            messagebox.showinfo("DemoSynthCAD",
                                "NURBS exported to:\n" + "\n".join(exported))
        except Exception as exc:
            self._status(f"Export NURBS failed: {exc}")
            messagebox.showerror("DemoSynthCAD",
                                 f"NURBS export failed: {exc}")
        return path

    def export_stl(self) -> Optional[str]:
        """MATLAB ``pushbutton_ExportSTL`` via :func:`hexiga.export.nrb_export_stl`."""
        initial = (self.DemoDATA.Name or "mpHexa") + ".stl"
        path = filedialog.asksaveasfilename(
            title="Export STL",
            defaultextension=".stl",
            initialfile=initial,
            filetypes=[("STL mesh", "*.stl"), ("All files", "*.*")],
        )
        if not path:
            return None

        def _w(patches: List[Nrb], fname: str) -> None:
            nrb_export_stl(patches,
                           "FOLDERPATH", os.path.dirname(fname) + os.sep,
                           "FILENAME", os.path.basename(fname),
                           "MPINTERFACES", True, "MPBOUNDARIES", True)

        try:
            folder = os.path.dirname(path) + os.sep
            base = os.path.basename(path)
            stem = base[:-4] if base.lower().endswith(".stl") else base
            exported: List[str] = []
            if self.DemoDATA.idxComposite:
                idxC = self.DemoDATA.idxComposite
                pL = folder + stem + "_L.stl"
                pW = folder + stem + "_W.stl"
                _w(list(self.DemoDATA.mpHexa[:idxC]), pL)
                _w(list(self.DemoDATA.mpHexa[idxC:]), pW)
                exported = [pL, pW]
            else:
                _w(list(self.DemoDATA.mpHexa), path)
                exported = [path]
            self._status("STL exported.")
            messagebox.showinfo("DemoSynthCAD",
                                "STL exported to:\n" + "\n".join(exported))
        except Exception as exc:
            self._status(f"Export STL failed: {exc}")
            messagebox.showerror("DemoSynthCAD", f"STL export failed: {exc}")
        return path

    # -- VISUALISATION ------------------------------------------------------
    def _close_viz(self) -> None:
        if self._viz_plotter is not None:
            try:
                self._viz_plotter.close()
            except Exception:
                pass
            self._viz_plotter = None

    def plot_current(self) -> Optional[Any]:
        """Port of ``plotDemoDATA``: dispatch to the selected visualisation
        mode and open a live pyvista window (replacing any previous one).

        Returns the :class:`pyvista.Plotter` for testing, or ``None`` on
        error.
        """
        mpHexa = self.DemoDATA.mpHexa
        if not mpHexa:
            messagebox.showwarning("DemoSynthCAD", "Nothing to plot -- Run a demo first.")
            return None

        mode = self.viz_var.get()
        ghostflag = (list(self.DemoDATA.GhostFLAG)
                     if self.DemoDATA.GhostFLAG else None)
        idxC = self.DemoDATA.idxComposite

        try:
            self._close_viz()
            if mode == VIZ_MODES[0]:
                p = plot_solid_domain(mpHexa, ghostflag=ghostflag,
                                      idx_composite=idxC, off_screen=False)
            elif mode == VIZ_MODES[1]:
                p = plot_ctrl_pts(mpHexa, off_screen=False)
            elif mode == VIZ_MODES[2]:
                p = plot_exploded_param(mpHexa, off_screen=False)
            elif mode == VIZ_MODES[3]:
                p = plot_reflection_lines(mpHexa, off_screen=False)
            else:
                p = plot_trabecular(
                    mpHexa,
                    hexaids=list(range(1, len(mpHexa) + 1, 2)),
                    ghost_all=True,
                    off_screen=False,
                )
            self._viz_plotter = p
            p.show()  # blocks until the window is closed
        except Exception as exc:
            self._status(f"Plot failed: {exc}")
            messagebox.showerror("DemoSynthCAD",
                                 f"Errors occurred while plotting! {exc}")
            return None
        return self._viz_plotter

    def render_offscreen(self) -> Any:
        """Build the current visualisation mode headlessly (for tests)."""
        mpHexa = self.DemoDATA.mpHexa
        mode = self.viz_var.get()
        ghostflag = (list(self.DemoDATA.GhostFLAG)
                     if self.DemoDATA.GhostFLAG else None)
        idxC = self.DemoDATA.idxComposite
        if mode == VIZ_MODES[0]:
            return plot_solid_domain(mpHexa, ghostflag=ghostflag,
                                     idx_composite=idxC)
        if mode == VIZ_MODES[1]:
            return plot_ctrl_pts(mpHexa)
        if mode == VIZ_MODES[2]:
            return plot_exploded_param(mpHexa)
        if mode == VIZ_MODES[3]:
            return plot_reflection_lines(mpHexa)
        return plot_trabecular(
            mpHexa,
            hexaids=list(range(1, len(mpHexa) + 1, 2)),
            ghost_all=True,
        )

    # -- event loop ---------------------------------------------------------
    def mainloop(self) -> None:
        self.root.mainloop()

    def destroy(self) -> None:
        self._close_viz()
        if self.own_root:
            self.root.destroy()

def main() -> None:
    app = DemoSynthCAD().build()
    app.mainloop()

if __name__ == "__main__":
    main()
