"""Shared helpers for the MEEP scripts. Length unit a = 1 mm, so the MEEP frequency is f*a/c.

Material mapping. This package uses eps = eps' - j eps'' with exp(+j w t). MEEP uses exp(-i w t) and
a "D-conductivity" sigma_D with eps(w) = eps' (1 + i sigma_D / w)  [MEEP docs, "Materials":
conductivity and complex permittivity]. So at the design frequency f_m (in MEEP units)
    sigma_D = 2 pi f_m eps''/eps'
is exact at f_m. MEEP fields are the complex conjugates of this package's fields, so compare
conj(meep) with this package's results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # lets scripts import tissue_slab_modes from this checkout

C0 = 299792458.0
A = 1e-3                               # MEEP length unit = 1 mm


def fmeep(f_hz: float) -> float:
    return f_hz * A / C0


def medium(eps_c: complex, f_hz: float):
    import meep as mp
    er, ei = float(np.real(eps_c)), float(-np.imag(eps_c))
    if ei <= 0:
        return mp.Medium(epsilon=er)
    return mp.Medium(epsilon=er, D_conductivity=2 * np.pi * fmeep(f_hz) * ei / er)


def layer_medium(L, f_hz: float):
    """`medium(L.eps, f_hz)`, or a perfect electric conductor if L.pec -- see Layer.pec's
    docstring (used for e.g. a copper-plate control, where the real copper conductivity is both
    unresolvable at mm grid spacing, skin depth ~1.3 um at 2.4 GHz, and numerically unstable as
    a D_conductivity medium: that huge a conductivity puts a near-pole of eps(w) = eps'(1 + i
    sigma_D/w) inside the Gaussian source pulse's spectrum. PEC is the correct and standard
    microwave-engineering limit at this resolution, and is handled by MEEP as an exact
    E_tangential = 0 boundary rather than a lossy-dielectric ODE, so it is unconditionally
    stable)."""
    import meep as mp
    return mp.metal if getattr(L, "pec", False) else medium(L.eps, f_hz)


def bot_medium(stack, f_hz: float):
    """`medium(stack.eps_bot, f_hz)`, or a perfect electric conductor if stack.bot_pec -- see
    layer_medium's docstring."""
    import meep as mp
    return mp.metal if getattr(stack, "bot_pec", False) else medium(stack.eps_bot, f_hz)


def layered_geometry(stack, depth_mm: float, x_extent=None, threeD: bool = False):
    """Blocks for a planar stack whose skin surface lies at y = 0 (2-D) or z = 0 (3-D), with
    tissue below. The bottom half-space fills down to -depth_mm (into the PML)."""
    import meep as mp
    geom = []
    top = 0.0
    inf = mp.inf
    for L in stack.layers:
        t = L.thickness / A
        c = top - t / 2
        if threeD:
            geom.append(mp.Block(size=mp.Vector3(inf, inf, t), center=mp.Vector3(0, 0, c),
                                 material=layer_medium(L, stack.freq)))
        else:
            geom.append(mp.Block(size=mp.Vector3(inf, t, inf), center=mp.Vector3(0, c),
                                 material=layer_medium(L, stack.freq)))
        top -= t
    t = depth_mm + top
    c = top - t / 2
    if threeD:
        geom.append(mp.Block(size=mp.Vector3(inf, inf, t), center=mp.Vector3(0, 0, c),
                             material=bot_medium(stack, stack.freq)))
    else:
        geom.append(mp.Block(size=mp.Vector3(inf, t, inf), center=mp.Vector3(0, c),
                             material=bot_medium(stack, stack.freq)))
    return geom


def eps_from_sigma(er: float, sigma: float, f_hz: float) -> complex:
    """Complex eps' - j eps'' from a real part and a conductivity in S/m (this package's
    exp(+jwt) convention)."""
    w = 2 * np.pi * f_hz
    return er - 1j * sigma / (w * 8.8541878128e-12)


def eps_from_tand(er: float, tand: float) -> complex:
    return er * (1 - 1j * tand)


def run_to_steady_dft(sim, monitors, probe_point, comp, tol=1e-7):
    """Run a Gaussian-pulse simulation until the DFT monitors converge, using
    stop_when_dft_decayed if this MEEP version has it, else field decay at a probe point."""
    import meep as mp
    if hasattr(mp, "stop_when_dft_decayed"):
        sim.run(until_after_sources=mp.stop_when_dft_decayed(tol))
    else:
        sim.run(until_after_sources=mp.stop_when_fields_decayed(200, comp, probe_point, tol))
