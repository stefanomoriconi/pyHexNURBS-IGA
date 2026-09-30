"""
hexiga.scaff.mpstats
====================

Report the number of
hexahedral elements and the total number of control points in a
multi-patch geometry.

Signature: ``[numH, numP] = mpStats(mpHexa, 'VERBOSE', true)``.

Notes (literal port, quirks preserved):

* ``mpHexa`` is a sequence of ``Nrb`` patches (or any objects carrying a
  ``.number`` tuple); ``numH`` is its length and ``numP`` is the sum of the
  per-patch control-point counts.
* On any failure MATLAB returns empty (``[]``) results and prints an error
  line; the port returns ``(None, None)`` in that case.
* The MATLAB "unrecognised parameter" branch literally prints
  ``Inputs{jj+1}`` (the *value* following the flag, not the flag name).
  That quirk is preserved.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..nurbs.nrb import Nrb

__all__ = ["mpStats"]

def _get_inputs(inputs: Sequence):
    """Parse the trailing (key, value) options, mirroring
    ``mpStats.m`` ``getInputs`` (step-2 loop over 0-based positions).
    """

    class OPTs:
        verbose: bool = False

    opts = OPTs()
    n = len(inputs)
    jj = 0
    while jj < n:
        key = inputs[jj]
        if isinstance(key, str) and key.upper() == "VERBOSE":
            if jj + 1 < n:
                val = inputs[jj + 1]
                if isinstance(val, (bool, int)):
                    opts.verbose = bool(val)
                else:
                    # MATLAB: logical(Inputs{jj+1}(1)) -> take first char.
                    s = str(val)
                    opts.verbose = len(s) > 0 and s[0].lower() in ("t", "true")
            else:
                opts.verbose = False
        else:
            # Literal quirk: MATLAB prints Inputs{jj+1} (the value), not the
            # flag name, in this branch.
            value = inputs[jj + 1] if jj + 1 < n else ""
            print(
                " * mpStats: Unrecognised Parsed Parameter: "
                f"{value} - Default Applied."
            )
        jj += 2
    return opts

def mpStats(
    mpHexa: Sequence[Nrb],
    *args,
    **kwargs,
) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(numH, numP)`` for a multi-patch geometry.

    Parameters
    ----------
    mpHexa
        Sequence of NURBS patches (each exposing ``.number``).
    *args
        Optional ``('VERBOSE', True)`` pair (MATLAB-style options).
    **kwargs
        Optional ``verbose=True`` keyword.

    Returns
    -------
    (numH, numP)
        Element count and total control-point count.  On failure,
        ``(None, None)`` (MATLAB's empty ``[]``).
    """
    # Allow both MATLAB-style positional options and a Python keyword.
    inputs: List = list(args)
    if "verbose" in kwargs:
        inputs = ["VERBOSE", kwargs.pop("verbose")] + inputs
    if kwargs:
        raise TypeError(f"unrecognized keyword arguments: {list(kwargs)}")

    opts = _get_inputs(inputs)

    try:
        if mpHexa is None:
            raise ValueError("mpHexa is empty")
        numH = len(mpHexa)
        numP = 0
        for hh in range(numH):
            number = mpHexa[hh].number
            prod = 1
            for v in number:
                prod *= int(v)
            numP += prod
        if opts.verbose:
            print(f" * multi-patch Stats: {numH} Element(s) ; {numP} Ctrl-pts")
        return numH, numP
    except Exception:
        print(" <!> mpStats: Incompatible Input Data - Please check Manually!")
        return None, None
