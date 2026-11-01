"""Dispersion function of a planar multilayer between two different half-spaces.

Geometry and conventions
------------------------
* Layers are stacked along x. A semi-infinite *top* medium (usually air) lies above,
  a semi-infinite *bottom* medium (e.g. muscle or brain) lies below. Propagation is
  along z with fields ~ exp(j(omega t - beta z))   (engineering convention:
  lossy media have Im(eps) < 0, as in the original Mathematica APM notebook).
* Everything is normalised to the free-space wavenumber k0:
      n      = beta / k0                (effective index)
      gamma_j = sqrt(n**2 - eps_j)      (transverse decay constant / k0)
      xi     = k0 * x
* TM state vector  u = [H_y, (1/eps) dH_y/dxi]  (both continuous at interfaces);
  TE state vector  u = [E_y, dE_y/dxi].
  A homogeneous layer of normalised thickness k0*d maps u from its bottom to its top by
      M = [[cosh(phi),            p * k0 d * shc(phi)],
           [gamma^2/p * k0 d * shc(phi), cosh(phi)  ]],   phi = gamma k0 d,
  with p = eps (TM) or 1 (TE) and shc(phi) = sinh(phi)/phi.  Every entry is an even
  function of gamma, so interior layers introduce **no branch cuts**
  [Chew, *Waves and Fields in Inhomogeneous Media*, ch. 2; Anemogiannis & Glytsis,
  J. Lightwave Technol. 10, 1344 (1992)].
* In the bottom half-space the field is exp(+gamma_b xi) (decays downward when
  Re gamma_b > 0, i.e. "proper"), so u_bottom = [1, gamma_b / p_b].
  Above the top interface the field is exp(-gamma_t xi) (proper if Re gamma_t > 0).
  Matching gives the dispersion function
      F = (gamma_t / p_t) * u_top[0] + u_top[1] = 0.

Four-sheeted Riemann surface and its uniformisation
---------------------------------------------------
F depends on the two square roots gamma_t and gamma_b, so as a function of n it lives
on a four-sheeted Riemann surface [Smith, Houde-Walter & Forbes, IEEE J. Quantum
Electron. 28, 1520 (1992)]. Because gamma_t**2 - gamma_b**2 = eps_b - eps_t = C is a
constant, the substitution
      gamma_t = (sqrt(C)/2) (t + 1/t),     gamma_b = (sqrt(C)/2) (t - 1/t)
makes F a single-valued analytic function of t on C\\{0}: the four sheets
(proper/improper top x proper/improper bottom) map to the four regions related by
t -> -t, 1/t, -1/t. We search in z = log t (|Im z| <= pi) exactly like the original
notebook searched in z = log(kappa_m) for the symmetric MIM guide [Kocabas, Veronis,
Miller & Fan, Phys. Rev. B 79, 035120 (2009)].

Overflow safety
---------------
cosh/sinh of thick lossy layers overflow double precision. Each layer matrix is
multiplied by exp(-|Re phi|); the product is returned together with the accumulated
log-scale S. Because the scale is a *positive real* factor, arg(F) (all the
argument-principle count needs) is unchanged, and log F = log(F_scaled) + S is exact.
"""
from __future__ import annotations

import numpy as np
import mpmath as mp

from .stack import Stack


# ----------------------------------------------------------------------------- helpers
def _shc_scaled(phi, r):
    """sinh(phi)/phi * exp(-r), stable for small and large |phi| (r = |Re phi|)."""
    phi = np.asarray(phi, dtype=complex)
    small = np.abs(phi) < 1e-3
    out = np.empty_like(phi)
    ps = phi[~small]
    out[~small] = (np.exp(ps - r[~small]) - np.exp(-ps - r[~small])) / (2 * ps)
    p2 = phi[small] ** 2
    out[small] = (1 + p2 / 6 + p2 ** 2 / 120) * np.exp(-r[small])
    return out


def _cosh_scaled(phi, r):
    return (np.exp(phi - r) + np.exp(-phi - r)) / 2


def layer_matrix_scaled(gamma2, eps, kd, pol="TM"):
    """Scaled transfer matrix entries (a, b, c, d) * exp(-r) and r = |Re phi|.

    gamma2: array of gamma**2;  kd = k0 * thickness.
    """
    gamma = np.sqrt(np.asarray(gamma2, dtype=complex))  # branch irrelevant (even in gamma)
    phi = gamma * kd
    r = np.abs(phi.real)
    ch = _cosh_scaled(phi, r)
    sh = kd * _shc_scaled(phi, r)  # = sinh(phi)/gamma * exp(-r)
    p = eps if pol == "TM" else 1.0
    return ch, p * sh, gamma2 * sh / p, ch, r


def top_admittance_scaled(n2, stack: Stack, gamma_b, pol="TM"):
    """Propagate u from the bottom half-space up to the top interface.

    Returns (u0, u1, S): scaled state at the top interface and log-scale S.
    """
    n2 = np.asarray(n2, dtype=complex)
    pb = stack.eps_bot if pol == "TM" else 1.0
    u0 = np.ones_like(n2)
    u1 = np.asarray(gamma_b, dtype=complex) / pb * np.ones_like(n2)
    S = np.zeros(n2.shape)
    for eps, kd in reversed(stack.layers_norm()):  # bottom-most layer first
        a, b, c, d, r = layer_matrix_scaled(n2 - eps, eps, kd, pol)
        u0, u1 = a * u0 + b * u1, c * u0 + d * u1
        # renormalise to keep the state O(1)
        m = np.maximum(np.abs(u0), np.abs(u1))
        m = np.where(m > 0, m, 1.0)
        u0, u1 = u0 / m, u1 / m
        S = S + r + np.log(m)
    return u0, u1, S


# ----------------------------------------------------------------------------- t-plane
class TPlane:
    """Uniformising map t <-> (gamma_top, gamma_bot, n) for a given stack."""

    def __init__(self, stack: Stack):
        self.stack = stack
        C = complex(stack.eps_bot - stack.eps_top)
        if abs(C) < 1e-12:
            raise ValueError("eps_top == eps_bot: use a symmetric (log gamma) mapping instead")
        self.s = np.sqrt(C) / 2

    def gammas(self, t):
        t = np.asarray(t, dtype=complex)
        return self.s * (t + 1 / t), self.s * (t - 1 / t)

    def n_eff(self, t):
        gt, _ = self.gammas(t)
        return np.sqrt(gt ** 2 + self.stack.eps_top)

    def t_from_gammas(self, gt, gb):
        return (gt + gb) / (2 * self.s)

    def t_from_n(self, n, top="proper", bot="proper"):
        """Point on the t-plane for effective index n on the requested sheet."""
        gt = np.sqrt(n ** 2 - self.stack.eps_top + 0j)
        gb = np.sqrt(n ** 2 - self.stack.eps_bot + 0j)
        gt = gt if (gt.real >= 0) == (top == "proper") else -gt
        gb = gb if (gb.real >= 0) == (bot == "proper") else -gb
        return self.t_from_gammas(gt, gb)


def dispersion_scaled(z, stack: Stack, pol="TM", tp: TPlane | None = None):
    """F(t = exp(z)) * t * exp(-S) and S, vectorised over z (numpy).

    Multiplying by t removes the simple pole of gamma_t, gamma_b at t = 0, so the
    function is analytic on the whole z strip. arg() of the returned value equals
    arg(F_true) exactly; log F_true = log(returned) + S.
    """
    tp = tp or TPlane(stack)
    z = np.asarray(z, dtype=complex)
    t = np.exp(z)
    gt, gb = tp.gammas(t)
    n2 = gt ** 2 + stack.eps_top
    u0, u1, S = top_admittance_scaled(n2, stack, gb, pol)
    pt = stack.eps_top if pol == "TM" else 1.0
    F = t * ((gt / pt) * u0 + u1)
    return F, S


# ----------------------------------------------------------------------------- notebook
def F_symmetric_slab_tm(beta, epsi, epsm, d):
    """fRaw1 of the original Mathematica APM notebook (symmetric slab, TM, both parities):
        ki [(epsm^2 ki^2 + epsi^2 km^2) sinh(2 pi ki d) + 2 epsi epsm ki km cosh(2 pi ki d)]
    with km = sqrt(beta^2 - epsm), ki = sqrt(beta^2 - epsi); d in free-space wavelengths.
    [cf. Kocabas, Veronis, Miller & Fan, Phys. Rev. B 79, 035120 (2009)]."""
    km = mp.sqrt(beta ** 2 - epsm)
    ki = mp.sqrt(km ** 2 + epsm - epsi)
    return ki * ((epsm ** 2 * ki ** 2 + epsi ** 2 * km ** 2) * mp.sinh(2 * mp.pi * ki * d)
                 + 2 * epsi * epsm * ki * km * mp.cosh(2 * mp.pi * ki * d))


# ----------------------------------------------------------------------------- mpmath
def dispersion_mp(z, stack: Stack, pol="TM"):
    """Unscaled analytic F(t = e^z) * t with mpmath (no overflow; used for polishing)."""
    z = mp.mpc(z)
    t = mp.exp(z)
    C = mp.mpc(stack.eps_bot - stack.eps_top)
    s = mp.sqrt(C) / 2
    gt, gb = s * (t + 1 / t), s * (t - 1 / t)
    n2 = gt ** 2 + stack.eps_top
    pb = stack.eps_bot if pol == "TM" else 1
    u0, u1 = mp.mpf(1), gb / pb
    for eps, kd in reversed(stack.layers_norm()):
        g2 = n2 - eps
        g = mp.sqrt(g2)
        phi = g * kd
        ch = mp.cosh(phi)
        sh = kd * (mp.sinh(phi) / phi if abs(phi) > 1e-30 else 1)
        p = eps if pol == "TM" else 1
        u0, u1 = ch * u0 + p * sh * u1, g2 * sh / p * u0 + ch * u1
    pt = stack.eps_top if pol == "TM" else 1
    return t * ((gt / pt) * u0 + u1)


def classify(t, tp: TPlane):
    """Sheet labels and physical quantities for a root at t."""
    gt, gb = tp.gammas(t)
    n = complex(np.sqrt(gt ** 2 + tp.stack.eps_top))
    if n.real < 0:  # choose the forward-phase representative
        n = -n
    return dict(
        t=complex(t),
        n_eff=n,
        gamma_top=complex(gt),
        gamma_bot=complex(gb),
        top="proper" if gt.real > 0 else "improper",
        bot="proper" if gb.real > 0 else "improper",
    )
