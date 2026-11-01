"""High-level mode search: APM on the uniformised t-plane + sheet classification."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .apm import APMLog, Rect, find_roots, make_polisher
from .dispersion import TPlane, classify, dispersion_mp, dispersion_scaled
from .stack import Stack

NP_TO_DB = 20 / np.log(10)


@dataclass
class Mode:
    pol: str
    z: complex
    t: complex
    n_eff: complex
    gamma_top: complex
    gamma_bot: complex
    top: str
    bot: str
    loss_dB_per_cm: float
    residual: float

    @property
    def sheet(self):
        return f"{self.top[0].upper()}/{self.bot[0].upper()}"

    @property
    def is_bound(self):
        return self.top == "proper" and self.bot == "proper"

    def __str__(self):
        return (f"{self.pol} n_eff = {self.n_eff.real:8.4f} {self.n_eff.imag:+8.4f}j   "
                f"loss = {self.loss_dB_per_cm:8.3f} dB/cm   sheet(top/bot) = {self.sheet}")


def find_modes(stack: Stack, pol="TM", zr=(-1.6, 1.6), zi=(-np.pi, np.pi), log: APMLog | None = None,
               max_abs_n=None):
    """All roots of the dispersion function in the rectangle zr x zi of z = log t."""
    tp = TPlane(stack)
    f = lambda z: dispersion_scaled(z, stack, pol, tp)
    polish = make_polisher(lambda z: dispersion_mp(z, stack, pol))
    # small asymmetric offsets keep roots on the symmetry lines (|Im z| = pi, Im z = 0)
    # off the contour, as the notebook's `assym` did
    a = np.pi / 1000
    rect = Rect(zr[0] - 0.0031, zr[1] + 0.0047, zi[0] - 2 * a, zi[1] + a)
    raw = find_roots(f, rect, polish=polish, log=log)
    modes = []
    k0cm = stack.k0 / 100
    for z, _, _ in raw:
        t = np.exp(z)
        c = classify(t, tp)
        F, S = f(np.array([z]))
        n = c["n_eff"]
        if max_abs_n and abs(n) > max_abs_n:
            continue
        modes.append(Mode(pol, z, t, n, c["gamma_top"], c["gamma_bot"], c["top"], c["bot"],
                          loss_dB_per_cm=-NP_TO_DB * n.imag * k0cm, residual=float(abs(F[0]))))
    # de-duplicate (roots found twice from overlapping nudged boxes)
    uniq = []
    for m in sorted(modes, key=lambda m: (m.loss_dB_per_cm, m.n_eff.real)):
        if all(abs(m.t - u.t) > 1e-7 * (1 + abs(u.t)) for u in uniq):
            uniq.append(m)
    return uniq


def bound_modes(modes):
    return [m for m in modes if m.is_bound]


def mode_kind(stack: Stack, m: Mode) -> str:
    """'tissue'  : bound (P/P) with < 50 % of S_z in the top medium (e.g. skull-guided);
    'surface' : bound with > 50 % in air (Zenneck-type surface pole);
    'leaky-top' / 'leaky-bot' / 'improper' : improper on the top / bottom / both sides."""
    from .fields import mode_profile, power_fractions
    if not m.is_bound:
        if m.top == "improper" and m.bot == "improper":
            return "improper"
        return "leaky-top" if m.top == "improper" else "leaky-bot"
    pr = mode_profile(stack, m.n_eff, m.gamma_top, m.gamma_bot, m.pol)
    return "surface" if power_fractions(stack, pr).get(stack.top_name, 0) > 0.5 else "tissue"


def pick(stack: Stack, modes, kind: str, min_re_n=0.0):
    sel = [m for m in modes if m.loss_dB_per_cm > 0 and m.n_eff.real >= min_re_n and mode_kind(stack, m) == kind]
    return min(sel, key=lambda m: m.loss_dB_per_cm) if sel else None
