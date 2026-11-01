"""Loss guiding <-> gain guiding: symmetric-slab analysis in Siegman's parameters.

Physics
-------
A low-index core between higher-index claddings is *index-antiguided*: with real
permittivities it has no bound modes, only leaky ones. Siegman showed that a fibre core
with enough *gain* relative to its cladding nevertheless confines a mode (gain-guided,
index-antiguided, "GG-IAG") [Siegman, JOSA A 20, 1617 (2003); JOSA B 24, 1677 (2007)].
Tissue realises the mirror image: fat (or bone) is the low-index core, and the *cladding*
(skin, muscle, CSF) is far lossier - the core has relative gain.

For a symmetric slab of half-thickness a, normalise with k0 a and write
        V^2 = (k0 a)^2 (eps_core - eps_clad) = DeltaN + j G         (exp(+j w t))
so that DeltaN < 0 is index antiguiding and G > 0 means the core is less lossy (or has
more gain) than the cladding - Siegman's complex V^2 = DeltaN + jG with the same signs.

TE: exact equivalence. The TE dispersion relation depends on the permittivities only
through gamma_j^2 = n^2 - eps_j. Adding the same imaginary constant j c to every eps_j
and to n^2 leaves every gamma_j unchanged, so "loss added to the cladding" and "gain
added to the core" are *identical* problems; only V^2 matters. The even TE modes obey
        kappa tan(kappa) = gamma,   kappa^2 + gamma^2 = V^2        (normalised to k0 a)
and the mode is bound (proper) when Re gamma > 0. On the threshold gamma = j eta (eta real):
        kappa tan(kappa) = j eta,   V^2 = kappa^2 - eta^2,
a one-parameter curve G_th(DeltaN) - the slab analogue of Siegman's fibre threshold.
Small-V limit: kappa^2 ~ j eta  =>  DeltaN = -eta^2, G = eta, i.e. G_th = sqrt(-DeltaN).

TM: the eps_j also appear in the boundary conditions ((1/eps) dH/dx continuous), so only
approximately V^2-only; the threshold is found numerically by continuation in the
cladding loss (see continue_root / tm_threshold_scale).

Search variable: as in the original Mathematica APM notebook for the symmetric MIM guide
[Kocabas et al., PRB 79, 035120 (2009)], z = ln(gamma_clad) unfolds the single cladding
branch cut; proper <=> Re gamma_clad > 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mpmath as mp

from .apm import Rect, find_roots, make_polisher
from .stack import C0

NP_TO_DB = 20 / np.log(10)


# ----------------------------------------------------------------------------- symmetric slab
def _parts(pol, ec, ef):
    return (ec, ef) if pol == "TM" else (1.0, 1.0)


def F_sym_np(z, eps_core, eps_clad, kd, pol="TM"):
    """Scaled dispersion function of clad|core(k0 d = kd)|clad in z = ln gamma_clad.
    Returns (F_scaled, S) with log F = log F_scaled + S (see dispersion.py)."""
    g = np.exp(np.asarray(z, dtype=complex))
    n2 = g ** 2 + eps_clad
    gf = np.sqrt(n2 - eps_core)
    phi = gf * kd
    pc, pf = _parts(pol, eps_clad, eps_core)
    r = np.abs(phi.real)
    ch = (np.exp(phi - r) + np.exp(-phi - r)) / 2
    small = np.abs(phi) < 1e-6
    safe = np.where(small, 1.0, phi)
    sh = np.where(small, np.exp(-r), (np.exp(phi - r) - np.exp(-phi - r)) / (2 * safe)) * kd
    u0 = ch + pf * sh * (g / pc)
    u1 = gf ** 2 * sh / pf + ch * (g / pc)
    return g * ((g / pc) * u0 + u1), r


def F_sym_mp(z, eps_core, eps_clad, kd, pol="TM"):
    g = mp.exp(z)
    n2 = g ** 2 + eps_clad
    gf2 = n2 - eps_core
    phi = mp.sqrt(gf2) * kd
    pc, pf = _parts(pol, eps_clad, eps_core)
    ch = mp.cosh(phi)
    sh = (mp.sinh(phi) / phi if abs(phi) > 1e-25 else 1) * kd
    u0 = ch + pf * sh * (g / pc)
    u1 = gf2 * sh / pf + ch * (g / pc)
    return g * ((g / pc) * u0 + u1)


@dataclass
class SymRoot:
    z: complex
    gamma_clad: complex
    n_eff: complex
    loss_dB_per_cm: float

    @property
    def bound(self):
        return self.gamma_clad.real > 0


def _root(z, eps_clad, k0):
    g = complex(np.exp(z))
    n = complex(np.sqrt(g ** 2 + eps_clad))
    n = n if n.real > 0 else -n
    return SymRoot(complex(z), g, n, -NP_TO_DB * n.imag * k0 / 100)


def sym_modes(eps_core, eps_clad, d, f, pol="TM", zr=(-4.0, 3.2)):
    k0 = 2 * np.pi * f / C0
    kd = k0 * d
    R = Rect(zr[0] - 0.0031, zr[1] + 0.0047, -np.pi - 0.003, np.pi + 0.0021)
    raw = find_roots(lambda z: F_sym_np(z, eps_core, eps_clad, kd, pol), R,
                     polish=make_polisher(lambda z: F_sym_mp(z, eps_core, eps_clad, kd, pol)))
    return [_root(z, eps_clad, k0) for z, _, _ in raw]


# ----------------------------------------------------------------------------- multilayer core
def F_symml_np(z, core, eps_clad, pol="TM"):
    """clad | core layers [(eps, k0 d), ...] | clad, same cladding both sides, z = ln gamma_clad.
    Uses the scaled transfer matrices of dispersion.py; returns (F_scaled, S)."""
    from .dispersion import layer_matrix_scaled
    g = np.exp(np.asarray(z, dtype=complex))
    n2 = g ** 2 + eps_clad
    pc = eps_clad if pol == "TM" else 1.0
    u0 = np.ones_like(g)
    u1 = g / pc
    S = np.zeros(g.shape)
    for e, kd in reversed(core):
        a, b, c, d, r = layer_matrix_scaled(n2 - e, e, kd, pol)
        u0, u1 = a * u0 + b * u1, c * u0 + d * u1
        mm = np.maximum(np.abs(u0), np.abs(u1))
        mm = np.where(mm > 0, mm, 1.0)
        u0, u1, S = u0 / mm, u1 / mm, S + r + np.log(mm)
    return g * ((g / pc) * u0 + u1), S


def F_symml_mp(z, core, eps_clad, pol="TM"):
    g = mp.exp(z)
    n2 = g ** 2 + eps_clad
    pc = eps_clad if pol == "TM" else 1
    u0, u1 = mp.mpf(1), g / pc
    for e, kd in reversed(core):
        g2 = n2 - e
        phi = mp.sqrt(g2) * kd
        ch = mp.cosh(phi)
        sh = (mp.sinh(phi) / phi if abs(phi) > 1e-25 else 1) * kd
        p = e if pol == "TM" else 1
        u0, u1 = ch * u0 + p * sh * u1, g2 * sh / p * u0 + ch * u1
    return g * ((g / pc) * u0 + u1)


def symml_modes(core_layers, eps_clad, f, pol="TM", zr=(-4.0, 3.2)):
    """core_layers: [(eps, thickness_m), ...] between two identical cladding half-spaces."""
    k0 = 2 * np.pi * f / C0
    core = [(complex(e), k0 * d) for e, d in core_layers]
    R = Rect(zr[0] - 0.0031, zr[1] + 0.0047, -np.pi - 0.003, np.pi + 0.0021)
    raw = find_roots(lambda z: F_symml_np(z, core, eps_clad, pol), R,
                     polish=make_polisher(lambda z: F_symml_mp(z, core, eps_clad, pol)))
    return [_root(z, eps_clad, k0) for z, _, _ in raw]


# ----------------------------------------------------------------------------- continuation
def _newton(fun, z, maxstep=0.03, it=60, tol=None):
    tol = tol if tol is not None else 10.0 ** (-mp.mp.dps + 3)
    fz = fun(z)
    last = None
    for _ in range(it):
        d = mp.diff(fun, z)
        if d == 0:
            return None
        step = fz / d
        if abs(step) > maxstep:
            step = step / abs(step) * maxstep
        lam = 1.0
        while lam > 1e-4:
            zn = z - lam * step
            fn = fun(zn)
            if abs(fn) < abs(fz):
                break
            lam /= 2
        else:
            return None
        z, fz = zn, fn
        last = abs(lam * step)
        if last < tol * (1 + abs(z)):
            return z
    return z if (last is not None and last < 1e-9 * (1 + abs(z))) else None


def continue_root(fun_of_p, z0, p0, p1, dp0=0.005, jump=0.05):
    """Follow a root z(p) of fun_of_p(p)(z) from p0 to p1 (predictor-corrector,
    adaptive step). Returns [(p, z), ...]."""
    out = [(p0, complex(z0))]
    z, zprev, p = mp.mpc(z0), None, p0
    dp = dp0 * np.sign(p1 - p0)
    while (p1 - p) * np.sign(p1 - p0) > 1e-12:
        pn = p + dp if (p1 - (p + dp)) * np.sign(p1 - p0) > 0 else p1
        pred = z if zprev is None else z + (z - zprev) * (pn - p) / (p - out[-2][0] if len(out) > 1 else 1)
        zn = _newton(fun_of_p(pn), pred)
        if zn is None or abs(zn - pred) > jump:
            dp /= 2
            if abs(dp) < 1e-6:
                break
            continue
        zprev, z, p = z, zn, pn
        out.append((p, complex(z)))
        if abs(dp) < dp0:
            dp *= 1.5
    return out


def tm_threshold_scale(eps_core, eps_clad_real, eps_clad_imag, d, f, pol="TM", s_max=8.0):
    """Cladding-loss scale s_th at which the fundamental mode of clad|core|clad turns from
    leaky to bound, with eps_clad(s) = eps_clad_real + j s eps_clad_imag. Returns
    (s_th or None, path) where path = [(s, SymRoot)]."""
    k0 = 2 * np.pi * f / C0
    kd = k0 * d
    ec = lambda s: complex(eps_clad_real, s * eps_clad_imag)
    ms = sym_modes(eps_core, ec(1.0), d, f, pol)
    nmin = 0.5 * np.sqrt(abs(complex(eps_core)))
    cand = [m for m in ms if m.n_eff.real > nmin and m.n_eff.imag < 0]
    if not cand:
        return None, []
    m0 = min(cand, key=lambda m: m.loss_dB_per_cm)
    fun = lambda s: (lambda z: F_sym_mp(z, eps_core, ec(s), kd, pol))
    down = continue_root(fun, m0.z, 1.0, 0.0)
    up = continue_root(fun, m0.z, 1.0, s_max)
    path = [(s, _root(z, ec(s), k0)) for s, z in (down[::-1] + up[1:])]
    th = None
    for (s1, r1), (s2, r2) in zip(path[:-1], path[1:]):
        if (r1.gamma_clad.real > 0) != (r2.gamma_clad.real > 0):
            th = s1 + (s2 - s1) * (-r1.gamma_clad.real) / (r2.gamma_clad.real - r1.gamma_clad.real)
            break
    return th, path


def _crossing(path):
    for (s1, r1), (s2, r2) in zip(path[:-1], path[1:]):
        if (r1.gamma_clad.real > 0) != (r2.gamma_clad.real > 0):
            return s1 + (s2 - s1) * (-r1.gamma_clad.real) / (r2.gamma_clad.real - r1.gamma_clad.real)
    return None


def antiguide_thresholds(eps_core_r, eps_clad_r, L, d, f, pol="TM", s_max=12.0, s_ref=(1.0, 3.0, 8.0)):
    """Start from a LOSSLESS index-antiguide (eps_core_r < eps_clad_r, both real) and
    add the same imaginary contrast s*L in two ways (identical V^2 at every s):
        'loss' : cladding  eps_clad = eps_clad_r - j s L        (loss guiding, tissue)
        'gain' : core      eps_core = eps_core_r + j s L        (gain guiding, Siegman)
    Follow the fundamental mode and return the thresholds s_th and G_th = (k0 a)^2 s_th L.
    For TE the two are identical (exact); for TM they differ."""
    k0 = 2 * np.pi * f / C0
    kd, k0a2 = k0 * d, (k0 * d / 2) ** 2
    ecl = lambda s: complex(eps_clad_r, -s * L)
    eco = lambda s: complex(eps_core_r, s * L)
    # identify the fundamental bound mode on the loss path at a reference loss
    m0, sref = None, None
    for s0 in s_ref:
        ms = [m for m in sym_modes(eps_core_r, ecl(s0), d, f, pol)
              if m.bound and m.n_eff.imag < 0 and m.n_eff.real > 0.3 * np.sqrt(eps_core_r)]
        if ms:
            m0, sref = min(ms, key=lambda m: m.loss_dB_per_cm), s0
            break
    if m0 is None:
        return None
    fl = lambda s: (lambda z: F_sym_mp(z, eps_core_r, ecl(s), kd, pol))
    down = continue_root(fl, m0.z, sref, 0.0)
    z_lossless = down[-1][1]
    out = {"DeltaN": k0a2 * (eps_core_r - eps_clad_r), "paths": {}}
    for kind in ("loss", "gain"):
        fun = (fl if kind == "loss" else
               (lambda s: (lambda z: F_sym_mp(z, eco(s), complex(eps_clad_r), kd, pol))))
        p = continue_root(fun, z_lossless, 0.0, s_max)
        ec = (lambda s: ecl(s)) if kind == "loss" else (lambda s: complex(eps_clad_r))
        path = [(s, _root(z, ec(s), k0)) for s, z in p]
        sth = _crossing(path)
        out["paths"][kind] = path
        out[f"s_{kind}"] = sth
        out[f"G_{kind}"] = None if sth is None else k0a2 * sth * L
    return out


# ----------------------------------------------------------------------------- Siegman map
def siegman_params(eps_core, eps_clad, d, f):
    """(DeltaN, G) with V^2 = (k0 a)^2 (eps_core - eps_clad), a = d/2."""
    k0a = 2 * np.pi * f / C0 * d / 2
    v2 = k0a ** 2 * (complex(eps_core) - complex(eps_clad))
    return v2.real, v2.imag


def te_threshold_curve(eta_max=12.0, n=400):
    """Exact TE0 bound/leaky threshold of the symmetric slab in the (DeltaN, G) plane.
    On threshold gamma = j eta; solve kappa tan kappa = j eta along the branch that
    starts at kappa^2 ~ j eta; return arrays (DeltaN, G, eta)."""
    etas = np.linspace(1e-3, eta_max, n)
    kap = mp.sqrt(1j * etas[0])
    dn, gg = [], []
    for eta in etas:
        kap = mp.findroot(lambda k: k * mp.tan(k) - 1j * eta, kap)
        v2 = complex(kap ** 2 - eta ** 2)
        dn.append(v2.real)
        gg.append(v2.imag)
    return np.array(dn), np.array(gg), etas


def te_is_bound(DeltaN, G):
    """Classify the TE0 mode of a symmetric slab with V^2 = DeltaN + jG (exact, any loss
    split between core and cladding): find the root continuously connected to the
    lossless guided/leaky TE0 and test Re gamma > 0."""
    v2 = complex(DeltaN, G)
    # start from a small-|V| guess and continue along a straight path in V^2
    kap = mp.sqrt(v2 / 2) if abs(v2) > 0 else mp.mpf(0)
    for t in np.linspace(0.02, 1, 60):
        vt = t * v2
        kap = mp.findroot(lambda k: k * mp.tan(k) - mp.sqrt(vt - k ** 2), kap if t > 0.02 else mp.sqrt(vt / 2))
    gam = mp.sqrt(v2 - kap ** 2)
    return float(mp.re(gam)) > 0
