"""Transverse field profiles of a mode of a planar stack.

TM (exp(j(wt - beta z)), x normal to the layers, normalised xi = k0 x):
    H_y(xi) from the transfer matrix,
    eta0 E_x = (n_eff / eps) H_y            (normal E; jumps at interfaces),
    eta0 E_z = j (1/eps) dH_y/dxi = j u1    (tangential E; continuous),
    S_z  ~ 1/2 Re(E_x H_y^*) = Re(n_eff/eps) |H_y|^2 / (2 eta0).
[e.g. Chew, Waves and Fields in Inhomogeneous Media, sec. 2.1; Kocabas et al.,
 Phys. Rev. B 79, 035120 (2009), eqs. for the MIM profiles]
TE: E_y from the transfer matrix (u = [E_y, dE_y/dxi]).
"""
from __future__ import annotations

import numpy as np

from .stack import Stack


def mode_profile(stack: Stack, n_eff: complex, gamma_top: complex, gamma_bot: complex, pol="TM",
                 above=None, below=None, npts=1200):
    """Return dict with x (m, 0 at the top interface, negative into the body), the
    main field component (H_y for TM, E_y for TE), E_x, E_z (TM only) and S_z,
    normalised so that max|field| = 1. Uses the root's own sheet (gamma signs)."""
    k0 = stack.k0
    D = stack.total_thickness
    above = above if above is not None else max(0.5 * D, 0.02 * stack.lam0)
    below = below if below is not None else max(0.5 * D, 0.02 * stack.lam0)
    n2 = n_eff ** 2
    p = (lambda e: e) if pol == "TM" else (lambda e: 1.0)

    # interface positions from the top (x = 0) downward
    edges = [0.0]
    for L in stack.layers:
        edges.append(edges[-1] - L.thickness)
    xb = edges[-1]
    xs = np.linspace(xb - below, above, npts)
    H = np.zeros_like(xs, dtype=complex)
    U1 = np.zeros_like(xs, dtype=complex)
    EPS = np.zeros_like(xs, dtype=complex)

    # bottom half-space: H = exp(gamma_b * k0 (x - xb))
    m = xs <= xb
    H[m] = np.exp(gamma_bot * k0 * (xs[m] - xb))
    U1[m] = gamma_bot / p(stack.eps_bot) * H[m]
    EPS[m] = stack.eps_bot
    u0, u1 = 1.0 + 0j, gamma_bot / p(stack.eps_bot)
    # layers, bottom-most first
    for i in range(len(stack.layers) - 1, -1, -1):
        L = stack.layers[i]
        x_lo, x_hi = edges[i + 1], edges[i]
        g = np.sqrt(n2 - L.eps + 0j)
        sel = (xs > x_lo) & (xs <= x_hi)
        s = k0 * (xs[sel] - x_lo)
        ch, sh = np.cosh(g * s), np.sinh(g * s)
        pe = p(L.eps)
        H[sel] = ch * u0 + pe * sh / g * u1
        U1[sel] = g * sh / pe * u0 + ch * u1
        EPS[sel] = L.eps
        kd = k0 * L.thickness
        ch, sh = np.cosh(g * kd), np.sinh(g * kd)
        u0, u1 = ch * u0 + pe * sh / g * u1, g * sh / pe * u0 + ch * u1
    # top half-space: H = u0 exp(-gamma_t k0 x)
    m = xs > 0
    H[m] = u0 * np.exp(-gamma_top * k0 * xs[m])
    U1[m] = -gamma_top / p(stack.eps_top) * H[m]
    EPS[m] = stack.eps_top

    inside = (xs <= 0) & (xs >= xb)          # normalise to the maximum inside the layers, so
    scale = H[inside][np.argmax(np.abs(H[inside]))]   # leaky (growing) tails do not hide the guide
    H, U1 = H / scale, U1 / scale
    out = dict(x=xs, edges=np.array(edges), eps=EPS, pol=pol, n_eff=n_eff,
               gamma_top=gamma_top, gamma_bot=gamma_bot, H_top=u0 / scale, H_bot=1.0 / scale, k0=k0)
    if pol == "TM":
        Ex = n_eff / EPS * H
        Ez = 1j * U1
        Sz = np.real(Ex * np.conj(H))
        nE, nS = np.max(np.abs(Ex[inside])), np.max(np.abs(Sz[inside]))
        out.update(Hy=H, Ex=Ex / nE, Ez=Ez / nE, Sz=Sz / nS)
    else:
        Sz = np.real(n_eff) * np.abs(H) ** 2
        out.update(Ey=H, Sz=Sz / np.max(np.abs(Sz[inside])))
    return out


def power_fractions(stack: Stack, prof) -> dict:
    """Fraction of |S_z| carried by each region. Finite layers are integrated on a fine
    grid; the two half-spaces analytically, int_0^inf |H|^2 e^{-2 k0 Re(gamma) x} dx =
    |H_interface|^2 / (2 k0 Re gamma), which is infinite on an improper side (the field
    grows away from the guide), in which case that side gets fraction 1."""
    k0, n = prof["k0"], prof["n_eff"]
    pt = stack.eps_top if prof["pol"] == "TM" else 1.0
    pb = stack.eps_bot if prof["pol"] == "TM" else 1.0
    gt, gb = prof["gamma_top"], prof["gamma_bot"]
    w = lambda p: abs(np.real(n / p)) if prof["pol"] == "TM" else abs(np.real(n))
    contrib = {}
    contrib[stack.top_name] = (w(pt) * abs(prof["H_top"]) ** 2 / (2 * k0 * gt.real)) if gt.real > 0 else np.inf
    # finite layers: re-sample each layer finely with the transfer matrix
    edges = prof["edges"]
    u0, u1 = prof["H_bot"], prof["H_bot"] * gb / pb
    for i in range(len(stack.layers) - 1, -1, -1):
        L = stack.layers[i]
        g = np.sqrt(n ** 2 - L.eps + 0j)
        pe = L.eps if prof["pol"] == "TM" else 1.0
        s = np.linspace(0, k0 * L.thickness, 400)
        Hs = np.cosh(g * s) * u0 + pe * np.sinh(g * s) / g * u1
        val = np.trapezoid(w(pe) * np.abs(Hs) ** 2, s / k0)
        contrib[L.name] = contrib.get(L.name, 0.0) + val
        kd = k0 * L.thickness
        u0, u1 = np.cosh(g * kd) * u0 + pe * np.sinh(g * kd) / g * u1, g * np.sinh(g * kd) / pe * u0 + np.cosh(g * kd) * u1
    contrib[stack.bot_name] = contrib.get(stack.bot_name, 0.0) + (
        (w(pb) * abs(prof["H_bot"]) ** 2 / (2 * k0 * gb.real)) if gb.real > 0 else np.inf)
    if any(np.isinf(v) for v in contrib.values()):
        return {k: (1.0 if np.isinf(v) else 0.0) for k, v in contrib.items()}
    tot = sum(contrib.values())
    return {k: v / tot for k, v in contrib.items()}
