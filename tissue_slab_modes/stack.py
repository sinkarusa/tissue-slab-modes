"""Planar layered stacks (top half-space / finite layers / bottom half-space)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C0 = 299_792_458.0
EPS0 = 8.8541878128e-12
MU0 = 4e-7 * np.pi
ETA0 = np.sqrt(MU0 / EPS0)


@dataclass
class Layer:
    name: str
    eps: complex          # complex relative permittivity, Im < 0 for loss (exp(+j w t))
    thickness: float      # metres
    pec: bool = False     # MEEP-only: force a perfect-electric-conductor block, ignoring `eps`
                          # (used for e.g. Asan et al. 2018's copper-plate control -- a real
                          # copper sigma is both unresolvable at mm grid spacing, skin depth
                          # ~1.3 um at 2.4 GHz, and numerically unstable as a D_conductivity
                          # medium; PEC is the correct and standard microwave-engineering limit).
                          # The solver side (find_modes etc.) never reads this field.

@dataclass
class Stack:
    """Top half-space, layers (listed from the top interface downward), bottom half-space."""

    freq: float
    top_name: str
    eps_top: complex
    layers: list[Layer]
    bot_name: str
    eps_bot: complex
    bot_pec: bool = False  # MEEP-only, see Layer.pec
    label: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def k0(self) -> float:
        return 2 * np.pi * self.freq / C0

    @property
    def lam0(self) -> float:
        return C0 / self.freq

    def layers_norm(self):
        """[(eps, k0*d), ...] from the top interface downward."""
        return [(complex(L.eps), self.k0 * L.thickness) for L in self.layers]

    @property
    def total_thickness(self) -> float:
        return sum(L.thickness for L in self.layers)

    def describe(self) -> str:
        rows = [f"{self.label}  (f = {self.freq/1e9:.3f} GHz)",
                f"  top    : {self.top_name:<18s} eps = {complex(self.eps_top):.3f}"]
        for L in self.layers:
            rows.append(f"  layer  : {L.name:<18s} eps = {complex(L.eps):.3f}   d = {L.thickness*1e3:.2f} mm")
        rows.append(f"  bottom : {self.bot_name:<18s} eps = {complex(self.eps_bot):.3f}")
        return "\n".join(rows)

    def flipped(self) -> "Stack":
        return Stack(self.freq, self.bot_name, self.eps_bot, list(reversed(self.layers)),
                     self.top_name, self.eps_top, label=self.label + " (flipped)")
