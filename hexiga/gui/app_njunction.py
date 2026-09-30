"""

A headless-friendly Tk GUI for generating a NURBS junction:

1. Generate a random 3-tube (or N-tube) direction set (``genDirTubes``).
2. Optionally regularise the directions (``regulariseDirTubes``).
3. Build the base QF fork simplex (``getBaseQForkSimplex``) and split it
   (``sdvBaseQForkSimplex``).
4. Extrude + bevel the branches (``getExtrudeBevelQFS``) to obtain
   lumen / wall QF scaffolds.
5. Build NURBS hexa lumen (``makeQFSs2nrbHexa``) and, if the Wall option is
   on, NURBS hexa wall + cap patches (``makeQFSs2nrbWallHexa``).
6. Optionally smooth via ``mp_turbo_smooth`` (two presets).
7. Export to NURBS (.txt) and STL.
8. Visualise in 6 ways (JuncSimplex, CtrlPts, SolidDomain, ExplodedParam,
   ReflectionLines, Trabecular).

No license gate (OMIT per project directive).  No blocking ``plt.show()``:
every visualisation returns a ``pyvista.Plotter`` handle which the GUI
shows via ``.show()`` only when the user clicks "Open window".  Off-screen
renders (for tests / screenshots) use ``.write_png``.
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..junc import (
    genDirTubes,
    regulariseDirTubes,
    getBaseQForkSimplex,
    sdvBaseQForkSimplex,
    getExtrudeBevelQFS,
    makeQFSs2nrbHexa,
    makeQFSs2nrbWallHexa,
)
from ..nurbs import nrbdegelev, nrbkntins
from ..smth import mp_turbo_smooth, set_progress_hook
from ..viz import (
    visualise_individual_qfs,
    plot_solid_domain,
    plot_ctrl_pts,
    plot_exploded_param,
    plot_reflection_lines,
    plot_trabecular,
)

# --- export (lazy imports to keep Tk off the import path) -----------------

def _do_export_nurbs(
    mpHexa: list,
    folder: str,
    fname: str,
    numTubes: Optional[int] = None,
    wall: bool = False,
) -> List[str]:
    """Faithful port of the NURBS export block in ``NJunction.m``."""
    from ..nurbs import nrbsave

    files: List[str] = []
    stem = os.path.splitext(fname)[0]
    if wall and numTubes is not None:
        L = list(mpHexa[: numTubes * 4])
        W = list(mpHexa[numTubes * 4:])
        fL = os.path.join(folder, stem + "_L.txt")
        fW = os.path.join(folder, stem + "_W.txt")
        nrbsave(fL, L, header_comments=["HexIGA Python port of NJunction (lumen)"])
        nrbsave(fW, W, header_comments=["HexIGA Python port of NJunction (wall)"])
        files += [fL, fW]
    else:
        f = os.path.join(folder, fname)
        nrbsave(f, list(mpHexa), header_comments=["HexIGA Python port of NJunction"])
        files.append(f)
    return files

def _do_export_stl(
    mpHexa: list,
    folder: str,
    fname: str,
    numTubes: Optional[int] = None,
    wall: bool = False,
) -> List[str]:
    """Faithful port of the STL export block in ``NJunction.m``."""
    from ..export import nrb_export_stl

    files: List[str] = []
    stem = os.path.splitext(fname)[0]
    if wall and numTubes is not None:
        L = list(mpHexa[: numTubes * 4])
        W = list(mpHexa[numTubes * 4:])
        fL = os.path.join(folder, stem + "_L.stl")
        fW = os.path.join(folder, stem + "_W.stl")
        nrb_export_stl(L, "FOLDERPATH", folder, "FILENAME", stem + "_L")
        nrb_export_stl(W, "FOLDERPATH", folder, "FILENAME", stem + "_W")
        files += [fL, fW]
    else:
        f = os.path.join(folder, stem + ".stl")
        nrb_export_stl(list(mpHexa), "FOLDERPATH", folder, "FILENAME", stem)
        files.append(f)
    return files

# --- data container -------------------------------------------------------

@dataclass
class NJuncDATA:
    """Port of the ``NJuncDATA`` struct in ``NJunction.m``."""

    numTubes: int = 3
    mixIOFLAG: bool = False
    extrdLen: float = 1.0
    bevelDeg: float = 0.0
    randomLen: bool = True
    WallFLAG: bool = False
    regDirTubesFLAG: bool = True

    QFSdata: Optional[Dict[str, Any]] = None
    ioLetsHexa: Optional[np.ndarray] = None
    mpHexa: Optional[List[Any]] = None
    hZ: Any = None

    def is_empty(self) -> bool:
        return self.QFSdata is None and self.mpHexa is None and self.hZ is None

# --- pure generation functions (headless testable) ------------------------

def run_generate(njuncDATA: NJuncDATA) -> Tuple[Optional[Dict[str, Any]], bool]:
    """"""
    try:
        dirTubes = genDirTubes(int(njuncDATA.numTubes))
        if njuncDATA.regDirTubesFLAG:
            dirTubes = regulariseDirTubes(dirTubes)
        baseQFSfcs, baseQFSpts, baseidx, baselblCIS = getBaseQForkSimplex(dirTubes)
        QFSfcs, QFSpts, idx, *_ = sdvBaseQForkSimplex(
            dirTubes, baseQFSfcs, baseQFSpts, baseidx, baselblCIS
        )

        ioLets = (
            np.ones(idx.shape, dtype=bool)
            if not njuncDATA.mixIOFLAG
            else (np.random.rand(*idx.shape) > 0.15)
        )
        shrtFct = np.ones(ioLets.shape)
        shrtFct[~ioLets] = 2.0 / 3.0
        bvlFct = np.ones(ioLets.shape)
        bvlFct[~ioLets] = 1.15

        extrLen = (
            (2.0 + ((njuncDATA.extrdLen * ioLets) * (njuncDATA.randomLen * np.random.rand(*idx.shape))))
            * shrtFct
        )
        bevelF = (
            (1.0 + ((njuncDATA.bevelDeg * ioLets) * (njuncDATA.randomLen * np.random.randn(*idx.shape))))
            * bvlFct
        )
        dirMag = (1.0 + 0.1 * np.random.rand(*idx.shape)) * shrtFct

        # Lumen extrude + bevel (MATLAB line 629-635)
        ebQFSfcs, ebQFSpts, KAe, idT, edT = getExtrudeBevelQFS(
            dirTubes, QFSfcs, QFSpts, idx, extrLen, bevelF,
            isflat=ioLets, dirMag=dirMag,
        )
        # Wall extrude + bevel (MATLAB line 641-647)
        ebQFSfcs2, ebQFSpts2, KAe2, idT2, edC = getExtrudeBevelQFS(
            dirTubes, QFSfcs, QFSpts, idx, extrLen, bevelF + 0.2,
            isflat=ioLets, dirMag=dirMag, wall=True,
        )
        # MATLAB line 653-667: pack into struct
        QFSdata: Dict[str, Any] = {
            "dirTubes": dirTubes,
            "baseQFSfcs": baseQFSfcs,
            "baseQFSpts": baseQFSpts,
            "baseidx": baseidx,
            "QFSfcs": QFSfcs,
            "QFSpts": QFSpts,
            "idx": idx,
            "ebQFSfcs": ebQFSfcs,
            "ebQFSpts": ebQFSpts,
            "idT": idT,
            "edT": edT,
            "ioLets": ioLets,
            "ebQFSfcs2": ebQFSfcs2,
            "ebQFSpts2": ebQFSpts2,
            "edC": edC,
        }
        return QFSdata, True
    except Exception as exc:
        print(f"<!> Errors occurred while generating NJunction! {exc}")
        return None, False

def run_generate_raw_scaff(
    njuncDATA: NJuncDATA,
) -> Tuple[Optional[List[Any]], Optional[np.ndarray], bool]:
    """"""
    QFSdata = njuncDATA.QFSdata
    if QFSdata is None:
        return None, None, False
    try:
        QD = QFSdata
        # Lumen hexa (MATLAB line 689)
        H = makeQFSs2nrbHexa(
            QD["QFSfcs"], QD["QFSpts"], QD["idT"],
            QD["ebQFSfcs"], QD["ebQFSpts"],
            QD["edT"],
            QD["idx"],
        )
        # Wall hexa (MATLAB lines 691-702)
        if njuncDATA.WallFLAG:
            Hw, CapFlag = makeQFSs2nrbWallHexa(
                QD["ebQFSfcs"], QD["ebQFSpts"],
                QD["QFSfcs"], QD["QFSpts"],
                QD["idT"],
                QD["ebQFSfcs2"], QD["ebQFSpts2"],
                QD["QFSfcs"], QD["QFSpts"] * 1.2,
                QD["edT"], QD["edC"],
                QD["idx"], QD["ioLets"],
            )
            ioLetsWall = ~np.asarray(CapFlag, dtype=bool)
        else:
            Hw = []
            ioLetsWall = np.empty(0, dtype=bool)

        # ioLetsLumen (MATLAB lines 708-711): 4 entries per ioLets entry
        ioLetsLumen = np.repeat(QD["ioLets"], 4)
        ioLetsHexa = np.concatenate([ioLetsLumen, ioLetsWall]) if njuncDATA.WallFLAG else ioLetsLumen

        mpHexaLumen = list(H)
        if njuncDATA.WallFLAG:
            mpHexaWall = list(Hw)
        else:
            mpHexaWall = []

        # Degree elevation + knot insertion (MATLAB lines 714-723)
        for h in range(len(mpHexaLumen)):
            mpHexaLumen[h] = nrbdegelev(mpHexaLumen[h], [2, 2, 2])
            mpHexaLumen[h] = nrbkntins(mpHexaLumen[h], ([], [], [0.25, 0.5, 0.75]))
        if njuncDATA.WallFLAG:
            for h in range(len(mpHexaWall)):
                mpHexaWall[h] = nrbdegelev(mpHexaWall[h], [2, 2, 2])
                if ioLetsWall[h]:
                    mpHexaWall[h] = nrbkntins(mpHexaWall[h], ([], [], [0.25, 0.5, 0.75]))

        mpHexa = mpHexaLumen + mpHexaWall if njuncDATA.WallFLAG else mpHexaLumen
        return mpHexa, ioLetsHexa, True
    except Exception as exc:
        print(f"<!> Errors occurred while constructing Raw Scaffolding! {exc}")
        return None, None, False

def run_smooth_raw_scaff(
    njuncDATA: NJuncDATA,
) -> Tuple[Optional[List[Any]], bool]:
    """"""
    if njuncDATA.mpHexa is None:
        return None, False
    try:
        except_patches = np.nonzero(~njuncDATA.ioLetsHexa)[0] + 1
        mpHexa = mp_turbo_smooth(
            njuncDATA.mpHexa,
            "INOUTLETSSIDES", [5, 6],
            "EXEPTINOUTLETSPATCHES", except_patches,
        )
        return mpHexa, True
    except Exception as exc:
        print(f"<!> Errors occurred during Raw Scaffolding smoothing! {exc}")
        return None, False

def run_turbo_smooth_raw_scaff(
    njuncDATA: NJuncDATA,
) -> Tuple[Optional[List[Any]], bool]:
    """"""
    if njuncDATA.mpHexa is None:
        return None, False
    try:
        except_patches = np.nonzero(~njuncDATA.ioLetsHexa)[0] + 1
        mpHexa = mp_turbo_smooth(
            njuncDATA.mpHexa,
            "INOUTLETSSIDES", [5, 6],
            "EXEPTINOUTLETSPATCHES", except_patches,
            "FREEZEIOSIDES", True,
            "TURBO", True,
            "TURBOCYCLES", 3,
            "SKIP", True,
        )
        return mpHexa, True
    except Exception as exc:
        print(f"<!> Errors occurred during Turbo smoothing! {exc}")
        return None, False

# --- GUI ------------------------------------------------------------------

class NJunction:
    """Tk GUI for the NJunction workflow (port of ``NJunction.m``)."""

    VIZ_MODES = ["JuncSimplex", "CtrlPts", "SolidDomain", "ExplodedParam", "ReflectionLines", "Trabecular"]

    def __init__(self, root: Optional[tk.Tk] = None) -> None:
        self.root = root or tk.Tk()
        self.root.title("HexIGA Python - NJunction (port of MAIN_gui/NJunction.m)")
        self.njuncDATA = NJuncDATA()
        self.viz_handle = None
        self.plotter_handles: List[Any] = []
        self.entries: Dict[str, Any] = {}
        self.checks: Dict[str, tk.BooleanVar] = {}
        # A single shared variable backs all visualisation radiobuttons so the
        # group is mutually exclusive (faithful to ``app_demo.viz_var``).
        self.viz_var: Optional[tk.StringVar] = None
        self.viz_mode = None
        self.btns: Dict[str, Any] = {}
        # Sequential smoothing state (see _apply_state): Smooth runs once,
        # then Turbo Smooth unlocks and runs once (per user spec).
        self._smooth_done = False
        self._turbo_done = False
        self._running = False
        # Worker-thread -> main-thread marshalling (progress + result).
        self._progress_queue: "queue.Queue" = queue.Queue()
        self._result_queue: "queue.Queue" = queue.Queue()
        self._build()
        self._start_result_polling()
        self._apply_state()
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)

    # --- widget tree -------------------------------------------------------

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        frame = ttk.Frame(self.root, padding=8)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        # config row
        cfg = ttk.LabelFrame(frame, text="Configuration", padding=6)
        cfg.grid(row=0, column=0, sticky="ew", pady=4)
        self.entries["numBranches"] = ttk.Entry(cfg, width=5)
        self.entries["numBranches"].insert(0, "3")
        ttk.Label(cfg, text="Branches (3-10):").grid(row=0, column=0, sticky="w")
        self.entries["numBranches"].grid(row=0, column=1, sticky="w")
        self.checks["MixedIO"] = tk.BooleanVar(value=False)
        self.checks["Wall"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(cfg, text="Mixed IO (15% random)", variable=self.checks["MixedIO"]).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(cfg, text="Include wall (outer shell)", variable=self.checks["Wall"]).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(cfg, text="extrdLen:").grid(row=3, column=0, sticky="e")
        self.entries["extrdLen"] = ttk.Entry(cfg, width=8)
        self.entries["extrdLen"].insert(0, "1.0")
        self.entries["extrdLen"].grid(row=3, column=1, sticky="w")
        ttk.Label(cfg, text="bevelDeg:").grid(row=4, column=0, sticky="e")
        self.entries["bevelDeg"] = ttk.Entry(cfg, width=8)
        self.entries["bevelDeg"].insert(0, "0.0")
        self.entries["bevelDeg"].grid(row=4, column=1, sticky="w")

        # actions
        actions = ttk.Frame(frame)
        actions.grid(row=1, column=0, sticky="ew", pady=4)
        self.btns["Generate"] = ttk.Button(actions, text="Generate", command=self._on_generate)
        self.btns["Generate"].pack(side="left", padx=4)
        self.btns["Reset"] = ttk.Button(actions, text="Reset", command=self._on_reset)
        self.btns["Reset"].pack(side="left", padx=4)

        # tools
        tools = ttk.LabelFrame(frame, text="Scaffolding", padding=6)
        tools.grid(row=2, column=0, sticky="ew", pady=4)
        self.btns["rawScaff"] = ttk.Button(tools, text="Build raw scaffolding (hexa)", command=self._on_raw_scaff, state="disabled")
        self.btns["rawScaff"].pack(side="left", padx=4)
        self.btns["smoothScaff"] = ttk.Button(tools, text="Smooth", command=self._on_smooth, state="disabled")
        self.btns["smoothScaff"].pack(side="left", padx=4)
        self.btns["turboSmoothScaff"] = ttk.Button(tools, text="Turbo Smooth", command=self._on_turbo_smooth, state="disabled")
        self.btns["turboSmoothScaff"].pack(side="left", padx=4)

        # export
        export = ttk.LabelFrame(frame, text="Export", padding=6)
        export.grid(row=3, column=0, sticky="ew", pady=4)
        self.btns["exportNURBS"] = ttk.Button(export, text="Export NURBS (.txt)", command=self._on_export_nurbs, state="disabled")
        self.btns["exportNURBS"].pack(side="left", padx=4)
        self.btns["exportSTL"] = ttk.Button(export, text="Export STL", command=self._on_export_stl, state="disabled")
        self.btns["exportSTL"].pack(side="left", padx=4)

        # viz
        viz = ttk.LabelFrame(frame, text="Visualisation", padding=6)
        viz.grid(row=4, column=0, sticky="ew", pady=4)
        # One shared variable across the whole group => mutually exclusive.
        self.viz_var = tk.StringVar(value=self.VIZ_MODES[0])
        self.viz_mode = self.VIZ_MODES[0]
        for i, m in enumerate(self.VIZ_MODES):
            ttk.Radiobutton(viz, text=m, value=m, variable=self.viz_var,
                            command=self._on_viz_mode).grid(row=0, column=i, sticky="w", padx=4)
        self.btns["openWindow"] = ttk.Button(viz, text="Open window", command=self._on_open_window, state="disabled")
        self.btns["openWindow"].grid(row=1, column=0, columnspan=len(self.VIZ_MODES), sticky="ew", pady=4)

        # smoothing progress bar (driven by mp_turbo_smooth's progress hook)
        prog = ttk.Frame(frame)
        prog.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        self.prog_label = ttk.Label(prog, text="", anchor="e")
        self.prog_label.pack(side="right")
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True)

        # status
        self.status_var = tk.StringVar(value="Ready. Enter branch count and click Generate.")
        ttk.Label(frame, textvariable=self.status_var, anchor="w", foreground="#555").grid(row=6, column=0, sticky="ew", pady=(8, 0))

    # --- state ------------------------------------------------------------

    def _apply_state(self) -> None:
        if self.njuncDATA.mpHexa is None and self.njuncDATA.QFSdata is None:
            state_map = {
                "rawScaff": "disabled",
                "smoothScaff": "disabled",
                "turboSmoothScaff": "disabled",
                "exportNURBS": "disabled",
                "exportSTL": "disabled",
                "openWindow": "disabled",
            }
        elif self.njuncDATA.mpHexa is None:
            state_map = {
                "rawScaff": "normal",
                "smoothScaff": "disabled",
                "turboSmoothScaff": "disabled",
                "exportNURBS": "disabled",
                "exportSTL": "disabled",
                "openWindow": "normal",
            }
        else:
            state_map = {
                "rawScaff": "disabled",
                # Sequential smoothing: Smooth enabled until it has run once;
                # Turbo Smooth enabled only after Smooth has run, and only
                # until it has itself run once (per user spec).
                "smoothScaff": "disabled" if self._smooth_done else "normal",
                "turboSmoothScaff": "normal"
                if (self._smooth_done and not self._turbo_done)
                else "disabled",
                "exportNURBS": "normal",
                "exportSTL": "normal",
                "openWindow": "normal",
            }
        for name, st in state_map.items():
            self.btns[name].configure(state=st)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    # --- handlers ---------------------------------------------------------

    def _on_generate(self) -> None:
        num_raw = self.entries["numBranches"].get()
        try:
            val = float(num_raw)
            if not np.isfinite(val) or not (2.0 < val < 11.0):
                from tkinter import messagebox
                messagebox.showwarning("Invalid input", "2 < branches < 11")
                self.entries["numBranches"].delete(0, "end")
                self.entries["numBranches"].insert(0, "3")
                self._set_status("Branch count out of range — reverting to 3.")
                return
        except ValueError:
            from tkinter import messagebox
            messagebox.showwarning("Invalid input", "2 < branches < 11")
            self.entries["numBranches"].delete(0, "end")
            self.entries["numBranches"].insert(0, "3")
            self._set_status("Invalid branch count — reverting to 3.")
            return

        self.njuncDATA.numTubes = int(round(val))
        self.njuncDATA.mixIOFLAG = bool(self.checks["MixedIO"].get())
        self.njuncDATA.WallFLAG = bool(self.checks["Wall"].get())
        try:
            self.njuncDATA.extrdLen = float(self.entries["extrdLen"].get())
        except ValueError:
            self.njuncDATA.extrdLen = 1.0
        try:
            self.njuncDATA.bevelDeg = float(self.entries["bevelDeg"].get())
        except ValueError:
            self.njuncDATA.bevelDeg = 0.0

        self._set_status("Generating NJunction...")
        self.root.update_idletasks()
        QFSdata, ok = run_generate(self.njuncDATA)
        if not ok:
            self._set_status("Generate failed.")
            return
        self.njuncDATA.QFSdata = QFSdata
        self.njuncDATA.mpHexa = None
        self.njuncDATA.ioLetsHexa = None
        self.njuncDATA.hZ = None
        self._smooth_done = False
        self._turbo_done = False
        # auto-preview
        try:
            self._render_viz_current(off_screen=False)
        except Exception:
            pass
        self._apply_state()
        self._set_status(f"Generated {self.njuncDATA.numTubes}-branch NJunction.")

    def _on_raw_scaff(self) -> None:
        self._set_status("Building raw scaffolding...")
        self.root.update_idletasks()
        mpHexa, ioLetsHexa, ok = run_generate_raw_scaff(self.njuncDATA)
        if not ok:
            self._set_status("Raw scaffolding failed.")
            return
        self.njuncDATA.mpHexa = mpHexa
        self.njuncDATA.ioLetsHexa = ioLetsHexa
        self._smooth_done = False
        self._turbo_done = False
        self._apply_state()
        self._set_status(f"Raw scaffolding: {len(mpHexa)} hexa.")

    def _start_result_polling(self) -> None:
        """Re-arm the main-thread result/progress poller every 50 ms.

        ``after`` is only valid from the Tk main thread, so this is called
        from ``__init__`` and re-scheduled by :meth:`_poll_results`.
        """
        try:
            self.root.after(50, self._poll_results)
        except tk.TclError:
            pass

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

    def _run_smooth_worker(self) -> None:
        """Worker-thread body for a single smoothing pass."""
        try:
            if self._smooth_done:
                # Turbo pass (only reachable after Smooth has run once).
                mpHexa, ok = run_turbo_smooth_raw_scaff(self.njuncDATA)
                label = "Turbo smoothing"
            else:
                mpHexa, ok = run_smooth_raw_scaff(self.njuncDATA)
                label = "Smoothing"
        except Exception as exc:  # defensive: the run_* helpers already catch
            mpHexa, ok, label = None, False, "Smoothing"
            print(f"<!> Errors occurred during {label}! {exc}")
        finally:
            set_progress_hook(None)
        # Propagate BOTH ok flag AND the (possibly) smoothed mpHexa so the
        # main thread can commit it into self.njuncDATA (the old synchronous
        # handlers did this inline; the threaded path needs an explicit
        # hand-off).
        self._result_queue.put((ok, mpHexa))

    def _on_smooth(self) -> None:
        """Launch the first smoothing pass on a worker thread.

        Sequential per user spec: Smooth may run only once; after it succeeds,
        Turbo Smooth unlocks (enforced by :meth:`_apply_state`) and Smooth
        stays disabled.
        """
        if self._running:
            self._set_status("A run is already in progress -- please wait.")
            return
        if self._smooth_done:
            self._set_status("Smooth already applied. Use Turbo Smooth.")
            return
        self._running = True
        self._set_status("Smoothing...")
        for name in ("smoothScaff", "turboSmoothScaff", "rawScaff"):
            self.btns[name].configure(state="disabled")
        try:
            self.progress.config(value=0, maximum=100)
            self.prog_label.config(text="")
        except tk.TclError:
            pass
        self._progress_queue = queue.Queue()
        self._result_queue = queue.Queue()
        set_progress_hook(self._on_smth_progress)
        self.root.update_idletasks()
        threading.Thread(target=self._run_smooth_worker, daemon=True).start()

    def _on_turbo_smooth(self) -> None:
        """Launch the Turbo pass on a worker thread (after one Smooth)."""
        if self._running:
            self._set_status("A run is already in progress -- please wait.")
            return
        if not self._smooth_done:
            self._set_status("Run Smooth first, then Turbo Smooth.")
            return
        if self._turbo_done:
            self._set_status("Turbo Smooth already applied.")
            return
        self._running = True
        self._set_status("Turbo smoothing...")
        for name in ("smoothScaff", "turboSmoothScaff", "rawScaff"):
            self.btns[name].configure(state="disabled")
        try:
            self.progress.config(value=0, maximum=100)
            self.prog_label.config(text="")
        except tk.TclError:
            pass
        self._progress_queue = queue.Queue()
        self._result_queue = queue.Queue()
        set_progress_hook(self._on_smth_progress)
        self.root.update_idletasks()
        threading.Thread(target=self._run_smooth_worker, daemon=True).start()

    def _poll_results(self) -> None:
        """Drain the worker progress/result queues on the Tk main thread."""
        self._start_result_polling()
        try:
            while True:
                current, total = self._progress_queue.get_nowait()
                if total > 0:
                    self.progress.config(maximum=total)
                self.progress.config(value=current)
                self.prog_label.config(text=f"smoothing: {current}/{total}")
        except queue.Empty:
            pass
        except (tk.TclError, AttributeError):
            return
        while True:
            try:
                ok, mpHexa = self._result_queue.get_nowait()
            except queue.Empty:
                break
            except tk.TclError:
                return
            self._finish_smooth(ok, mpHexa)

    def _finish_smooth(self, ok: bool, mpHexa) -> None:
        """Apply the post-smooth state on the main thread.

        Commits the smoothed scaffolding back into ``self.njuncDATA`` (so
        Export / Viz / a subsequent Turbo pass all see the result), updates the
        sequential-enable bookkeeping, and refreshes the button states.
        """
        set_progress_hook(None)
        try:
            self.progress.config(value=0)
            self.prog_label.config(text="done" if ok else "")
        except tk.TclError:
            pass
        self._running = False
        if ok and mpHexa is not None:
            # Commit the smoothed scaffolding into the shared data object.
            self.njuncDATA.mpHexa = mpHexa
            # Sequential-enable bookkeeping: Turbo only becomes available
            # AFTER a Smooth pass; Turbo can only run once; once Turbo has
            # run, both are disabled (MATLAB ``disableSmoothScaffPushbuttons``
            # equivalent).
            if self._smooth_done and not self._turbo_done:
                self._turbo_done = True
            else:
                self._smooth_done = True
            self._apply_state()
            self._set_status(f"Smoothed: {len(mpHexa)} hexa.")
        else:
            # On failure, re-enable buttons so the user can retry (flags
            # unchanged => _apply_state restores the pre-run state).
            self._apply_state()
            self._set_status("Smoothing failed.")

    def _on_export_nurbs(self) -> None:
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("NURBS", "*.txt"), ("All files", "*.*")],
            initialfile="njunc.txt",
        )
        if not path:
            return
        folder, fname = os.path.split(path)
        try:
            files = _do_export_nurbs(
                self.njuncDATA.mpHexa, folder, fname,
                numTubes=self.njuncDATA.numTubes, wall=self.njuncDATA.WallFLAG,
            )
            self._set_status(f"Exported NURBS: {', '.join(os.path.basename(f) for f in files)}")
        except Exception as exc:
            self._set_status(f"NURBS export failed: {exc}")

    def _on_export_stl(self) -> None:
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".stl",
            filetypes=[("STL", "*.stl"), ("All files", "*.*")],
            initialfile="njunc.stl",
        )
        if not path:
            return
        folder, fname = os.path.split(path)
        try:
            files = _do_export_stl(
                self.njuncDATA.mpHexa, folder, fname,
                numTubes=self.njuncDATA.numTubes, wall=self.njuncDATA.WallFLAG,
            )
            self._set_status(f"Exported STL: {', '.join(os.path.basename(f) for f in files)}")
        except Exception as exc:
            self._set_status(f"STL export failed: {exc}")

    def _on_viz_mode(self) -> None:
        self.viz_mode = self.viz_var.get()
        self._set_status(f"Visualisation mode: {self.viz_mode}")

    def _on_open_window(self) -> None:
        try:
            self._render_viz_current(off_screen=False)
            self._set_status(f"Opened {self.viz_mode} view.")
        except Exception as exc:
            self._set_status(f"Visualisation failed: {exc}")

    def _on_reset(self) -> None:
        from tkinter import messagebox
        if messagebox.askyesno("Resetting...", "Do You Really Want to Reset?"):
            self.njuncDATA = NJuncDATA()
            self.entries["numBranches"].delete(0, "end")
            self.entries["numBranches"].insert(0, "3")
            self.entries["extrdLen"].delete(0, "end")
            self.entries["extrdLen"].insert(0, "1.0")
            self.entries["bevelDeg"].delete(0, "end")
            self.entries["bevelDeg"].insert(0, "0.0")
            self.checks["MixedIO"].set(False)
            self.checks["Wall"].set(False)
            self._smooth_done = False
            self._turbo_done = False
            self.viz_var.set("JuncSimplex")
            self.viz_mode = "JuncSimplex"
            self._apply_state()
            self._set_status("Reset.")

    # --- rendering --------------------------------------------------------

    def _render_viz_current(self, off_screen: bool) -> None:
        """Render the currently selected visualisation mode.

        Returns the ``pyvista.Plotter`` handle (or ``None`` if no data).
        """
        import pyvista as pv

        QFSdata = self.njuncDATA.QFSdata
        mpHexa = self.njuncDATA.mpHexa
        if QFSdata is None:
            return None

        mode = self.viz_mode or "JuncSimplex"
        QD = QFSdata
        N = int(self.njuncDATA.numTubes)

        if mode == "JuncSimplex":
            # MATLAB plotJuncSimplex: visualiseIndividualQFSs + nrbmultipatch ghost
            p = visualise_individual_qfs(
                QD["dirTubes"], QD["QFSfcs"], QD["QFSpts"], QD["idx"],
                EBQFSFCS=QD["ebQFSfcs"],
                EBQFSPTS=QD["ebQFSpts"],
                KAE=None,
                SHOWNODES=True,
                CRVSIMPLEX=True,
                ARCLINEWIDTH=3,
                QUIVERSCALE=1,
                QUIVERLINEWIDTH=3,
                PRINCDIRSFLAG=True,
                VDIR=QD["baseQFSpts"][:, 0] if np.asarray(QD["baseQFSpts"]).ndim == 2 else QD["baseQFSpts"],
                NODESCOLORS=np.ones((1, np.asarray(QD["QFSpts"]).shape[1])),
                OFFSCREEN=off_screen,
            )
            self._attach_plotter(p)
            # add edT black dots (MATLAB plot3)
            try:
                edT = np.asarray(QD["edT"])
                if edT.ndim == 2 and edT.shape[0] >= 3:
                    p.add_points(pv.PolyData(edT[:3].T, verts=[]), color="k", point_size=5)
            except Exception:
                pass
            if not off_screen:
                p.show()
            return p

        if mpHexa is None:
            return None

        if mode == "CtrlPts":
            p = plot_ctrl_pts(mpHexa, off_screen=off_screen)
            self._attach_plotter(p)
            if not off_screen:
                p.show()
            return p

        if mode == "SolidDomain":
            p = plot_solid_domain(mpHexa, off_screen=off_screen)
            self._attach_plotter(p)
            if not off_screen:
                p.show()
            return p

        if mode == "ExplodedParam":
            p = plot_exploded_param(mpHexa, off_screen=off_screen)
            self._attach_plotter(p)
            if not off_screen:
                p.show()
            return p

        if mode == "ReflectionLines":
            p = plot_reflection_lines(mpHexa, off_screen=off_screen)
            self._attach_plotter(p)
            if not off_screen:
                p.show()
            return p

        if mode == "Trabecular":
            lk = np.linspace(0.0, 1.0, 5)
            p = plot_trabecular(
                mpHexa,
                uknots=lk, vknots=lk, wknots=lk,
                ghost_all=True,
                off_screen=off_screen,
            )
            self._attach_plotter(p)
            if not off_screen:
                p.show()
            return p

        raise ValueError(f"Unknown viz mode: {mode}")

    def _attach_plotter(self, p) -> None:
        self.plotter_handles.append(p)
        if len(self.plotter_handles) > 4:
            old = self.plotter_handles.pop(0)
            try:
                old.close()
            except Exception:
                pass

    # --- lifecycle --------------------------------------------------------

    def close_viz(self) -> None:
        for p in self.plotter_handles:
            try:
                p.close()
            except Exception:
                pass
        self.plotter_handles = []

    def destroy(self) -> None:
        self.close_viz()
        try:
            self.root.destroy()
        except Exception:
            pass

    def mainloop(self) -> None:
        self.root.mainloop()

def main() -> None:
    NJunction().mainloop()

if __name__ == "__main__":
    main()
