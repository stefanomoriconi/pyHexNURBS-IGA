"""Port of ``mpTurboSmooth.m``.

Top-level smoothing driver:

1. ``extractCtrlPtsTopologyFromMultiPatchHexa`` -> ``[mpCPT, mpCPTatrb, mpCuboid]``
2. ``appendSmthAtrbs_new(mpCPT, mpCPTatrb, ...)`` -> ``mpCPTSini``
3. ``iterativeSmoothing`` -> ``mpCPTSfin``
4. ``pushbackCtrlPtsTopologyToMultiPatchHexa`` -> ``mpHexaSmth``

Optional Turbo mode: repeat ``refineTermKntsMPHexa`` + smoothing cycle
``TurboCycles`` times.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional, Tuple

import numpy as np

from ..nurbs.nrb import Nrb
from .append_smth_atrbs_new import append_smth_atrbs_new
from .extract_cpt_topology_mp import extract_ctrl_pts_topology_from_multipatch_hexa
from .pushback import pushback_ctrl_pts_topology_to_mp
from .refine_term_knts_mp import refine_term_knts_mp
from .smth_cpt_topology_atrbs import smth_ctrl_pts_topology_atrbs

# ---------------------------------------------------------------------------
# Optional, thread-safe progress hook.
#
# This is a *Python-only* convenience that does NOT alter the MATLAB opt
# surface or any smoothing math.  A GUI (or any caller) may register a
# callback ``cb(current, total)`` to observe the 0->MAXITER smoothing
# progress of the *currently running* smoothing driver instance.  It is
# invoked from whichever thread is running the smoothing, so the callback
# is responsible for marshalling back to its own UI thread.
# ---------------------------------------------------------------------------
_progress_hook: Optional[Callable[[int, int], None]] = None
_progress_hook_lock = threading.Lock()

def set_progress_hook(
    cb: Optional[Callable[[int, int], None]],
) -> None:
    """Register (``cb``) or clear (``None``) the smoothing progress hook.

    ``cb`` is called as ``cb(current_iteration, max_iteration)`` after each
    smoothing iteration, from the thread that is running the smoothing.
    """
    global _progress_hook
    with _progress_hook_lock:
        _progress_hook = cb

def _emit_progress(current: int, total: int) -> None:
    """Fire the progress hook, if one is registered, swallowing errors."""
    with _progress_hook_lock:
        cb = _progress_hook
    if cb is None:
        return
    try:
        cb(current, total)
    except Exception:
        # Progress reporting must never break the smoothing pipeline.
        pass

def _get_inputs(pairs):
    # MATLAB matches keys case-insensitively via `switch upper(Inputs{jj})`.
    # We store every key in UPPER-CASE and look it up in UPPER-CASE too, so
    # that `mp_turbo_smooth(mp, "SKIP", True)` and `("Skip", True)` both work.
    OPTs: dict = {}
    it = iter(pairs)
    for key in it:
        OPTs[str(key).upper()] = next(it)
    OPTs.setdefault("INOUTLETSSIDES", [])
    OPTs.setdefault("EXEPTINOUTLETSPATCHES", [])
    OPTs.setdefault("EXEPTPATCHSIDEPAIRS", [])
    OPTs.setdefault("SHELLS", [])
    # MATLAB's mpTurboSmooth exposes FREEZEIOSIDES; the cycle forwards it to
    # appendSmthAtrbs_new as FREEZESHELLBOUNDARY.
    OPTs.setdefault("FREEZEIOSIDES", False)
    OPTs.setdefault("TURBO", False)
    OPTs.setdefault("TURBOCYCLES", 2)
    OPTs.setdefault("TURBOREFINESTEPS", [])
    OPTs.setdefault("SKIP", False)
    OPTs.setdefault("MAXITER", 100)
    return OPTs

def _iterative_smoothing(mpCPTSini, OPTs):
    """Port of ``iterativeSmoothing``."""
    maxItr = int(OPTs.get("MAXITER", 100))
    converged = False
    itr = 0
    mpCPTS = mpCPTSini
    while (not converged) and itr < maxItr:
        itr += 1
        mpCPTS = smth_ctrl_pts_topology_atrbs(mpCPTS)
        converged = all(c.isSmth for c in mpCPTS)
        _emit_progress(itr, maxItr)
        if itr >= maxItr:
            print(" * mpTurboSmooth: Smoothing Terminated by maxItr!")
    return mpCPTS

def _mp_turbo_smooth_cycle(mpHexa: list, OPTs: dict):
    """Port of ``mpTurboSmoothCycle``."""
    # extract accepts either the camelCase MATLAB keys or the uppercase
    # name/value spellings.
    (mpCPT, mpCPTatrb, mpCuboid) = extract_ctrl_pts_topology_from_multipatch_hexa(
        mpHexa,
        "INOUTLETSSIDES", OPTs.get("INOUTLETSSIDES", []),
        "EXEPTINOUTLETSPATCHES", OPTs.get("EXEPTINOUTLETSPATCHES", []),
        "EXEPTPATCHSIDEPAIRS", OPTs.get("EXEPTPATCHSIDEPAIRS", []),
    )
    # MATLAB passes OPTs.FreezeIOSides through as 'FREEZESHELLBOUNDARY'
    freezeShellBoundary = bool(OPTs.get("FREEZEIOSIDES", False))
    mpCPTSini = append_smth_atrbs_new(
        mpCPT, mpCPTatrb,
        "SHELLS", OPTs.get("SHELLS", []),
        "FREEZESHELLBOUNDARY", freezeShellBoundary,
    )
    mpCPTSfin = _iterative_smoothing(mpCPTSini, OPTs)
    mpHexaSmth = pushback_ctrl_pts_topology_to_mp(mpCPTSfin, mpCPTatrb)
    return mpHexaSmth

def mp_turbo_smooth(mpHexa: list, *pairs) -> list:
    """Port of ``mpTurboSmooth``.  Returns a list of ``Nrb`` patches."""
    OPTs = _get_inputs(pairs)

    # MATLAB: Skip ONLY skips the initial (default) smoothing cycle.
    # The Turbo loop below still runs whenever OPTs.Turbo is true,
    # even when Skip=true. (Faithful port of mpTurboSmooth.m.)
    if OPTs.get("SKIP", False):
        mpHexaSmth = list(mpHexa)
    else:
        mpHexaSmth = _mp_turbo_smooth_cycle(mpHexa, OPTs)

    if bool(OPTs.get("TURBO", False)):
        TurboCycles = int(OPTs.get("TURBOCYCLES", 2))
        steps = OPTs.get("TURBOREFINESTEPS", [])
        for cc in range(1, TurboCycles + 1):
            if not steps:
                mpHexaSmth = refine_term_knts_mp(mpHexaSmth)
            else:
                steps = np.atleast_1d(steps)
                uRef = (cc % int(steps[0])) == 0
                vRef = (cc % int(steps[1])) == 0 if len(steps) > 1 else True
                wRef = (cc % int(steps[2])) == 0 if len(steps) > 2 else True
                mpHexaSmth = refine_term_knts_mp(
                    mpHexaSmth,
                    "UREF", uRef, "VREF", vRef, "WREF", wRef,
                )
            mpHexaSmth = _mp_turbo_smooth_cycle(mpHexaSmth, OPTs)

    return mpHexaSmth
