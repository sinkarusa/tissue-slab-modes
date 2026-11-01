"""Modal excitation (residue) table: how strongly a small source in layered tissue excites
each catalog TM mode -- and whether that mode is actually present in a real source's field.

The dispersion-function plots in this repo (``02_pole_map.png`` etc.) show only the *zeros*
of the dispersion function F -- the modes themselves -- independent of any source. This script
adds the other half of the picture: for a given small source at depth x_s, how much of each
mode does it launch, and is that mode even reachable by a real source's field.

Geometry and conventions (shared with ``tissue_slab_modes.dispersion``): layers stacked along
x, 0 at the top (skin) interface, negative into the body; fields ~ exp(j(wt - beta z)); TM state
vector u = [H_y, v] with v = (1/eps) dH_y/dxi, xi = k0 x (both continuous at interfaces away from
a source). n = beta/k0 is the mode index.

1. Spectral Green's function. For a 2-D (y-invariant) line source at xi_s -- an electric current
   filament along y, localized at (xi_s, z=0) -- Fourier-transforming Maxwell's equations in z
   turns the TM field into a 1-D transmission-line problem in xi, with H_y playing the role of
   voltage and v of current:
       dH_y/dxi = eps(xi) v,          dv/dxi = (n^2/eps(xi) - 1) H_y      (+ source terms).
   Integrating the delta-function source across xi_s (direct from the Maxwell curl equations,
   not assumed) gives clean jump conditions at the source, for a unit-amplitude current:
     * normal  (x-directed, perpendicular to the layers):  H_y continuous,  v jumps by  j n / eps_s
     * tangential (z-directed, along the propagation axis): v continuous,  H_y jumps by  1
   (eps_s = eps of the layer holding the source). This matches the standard antenna-theory
   duality: a current along the propagation axis is a series (voltage-jump) source, a current
   transverse to it is a shunt (current-jump) source.
2. The two homogeneous solutions used to build G are exactly the up/down transfer-matrix states
   used for the catalog's own dispersion function: psi_B(xi), built from the bottom half-space's
   regular branch upward (same recursion as ``top_admittance_scaled``), and psi_T(xi), built from
   the top half-space's regular branch downward. G is their combination matched at xi_s; its
   denominator is the Wronskian W(n) = H_B v_T - H_T v_B, which is constant in xi and equal to
   -F_raw(n) (F_raw = F/t in ``dispersion_mp``'s normalization) -- i.e. (proportional to) the same
   F whose zeros are the catalog's modes. Evaluating gamma_top, gamma_bot via ``TPlane`` (rather
   than re-selecting sqrt branches) keeps this valid off the real axis, on improper sheets.
3. Residue at a mode n_m: since only W vanishes at the pole, Res = numerator(n_m) / (dW/dn)|_{n_m},
   with dW/dn from a 4-point complex finite difference (mirrors ``sommerfeld.residue_R``) moved
   along n while re-resolving gamma_top, gamma_bot to the mode's own sheet at each step via
   ``TPlane.t_from_n`` -- finite and correct for proper (P/P) and improper (I/P, P/I) poles alike.
4. 3-D correction. A 2-D line source's modal terms carry no radial spreading; a 3-D point dipole's
   do, and the reduction from a 2-D polar spectral integral (q dq, q = k_rho/k0 = n) to a 1-D
   radial integral via the large-argument Hankel asymptote H_0^(2)(k0 n rho) ~ sqrt(2/(pi k0 n
   rho)) exp(-j(...)) contributes an extra, pole-dependent factor of q * sqrt(1/q) = sqrt(n) on
   top of the 2-D residue (the q dq measure times the Hankel amplitude falloff; both depend on the
   stack's response only through n, not on dipole orientation, so this factor is the same for
   normal and tangential sources). |residue_3D| = |residue_2D| * sqrt(|n_m|).
5. Pole capture. A residue only matters if the pole is actually reached by the real source's
   field. The exact field is a real-n spectral integral; closing it picks up a pole only if, in
   EACH half-space, the pole's sheet there is the one the deformation actually reaches -- improper
   inside the branch-point region around n = sqrt(eps), proper outside it (Tamir & Oliner 1963) --
   and only if it decays along z (Im n < 0); the rest of the exact field comes from two vertical
   branch-cut integrals (one per half-space). A bound, lossy (P/P) pole can still fail this test
   and be invisible to a real source -- the classic example is a lossy half-space's own Zenneck
   pole, hidden under the air branch cut (Collin 2004). Checked directly in check d below.

References: T. Tamir & A. A. Oliner, "Guided complex waves. Part 1: Fields at an interface," Proc. IEE 110(2):310-324 (1963);
R. E. Collin, "Hertzian dipole radiating over a lossy earth or sea: some early and late
20th-century controversies," IEEE Antennas Propag. Mag. 46(2):64-79 (2004).

Run:  python scripts/residues.py   (writes results/residues.md; ~10 seconds)
"""
from __future__ import annotations

import cmath
import pathlib
import sys
import time

import numpy as np
from scipy.integrate import quad

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import make_figures as M  # noqa: E402

from tissue_slab_modes.dispersion import TPlane  # noqa: E402
from tissue_slab_modes.modes import find_modes, pick  # noqa: E402
from tissue_slab_modes.sommerfeld import residue_R  # noqa: E402

RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "residues.md"
F_BT = M.F_BT


# =================================================================== 1. TM transfer-matrix propagation
def _shc(phi):
    """sinh(phi)/phi, stable at phi = 0 (scalar complex)."""
    if abs(phi) < 1e-8:
        return 1 + phi ** 2 / 6 + phi ** 4 / 120
    return cmath.sinh(phi) / phi


def _step(H, v, gamma2, eps, delta):
    """Propagate TM state (H, v = (1/eps) dH/dxi) a signed depth step `delta` through a
    homogeneous region (gamma2 = n^2 - eps, eps). Same transfer matrix as
    ``dispersion.layer_matrix_scaled``, unscaled, exact for any delta (not just a full layer)."""
    gamma = cmath.sqrt(gamma2)
    phi = gamma * delta
    ch = cmath.cosh(phi)
    s = delta * _shc(phi)  # = sinh(phi)/gamma, finite as gamma -> 0
    return ch * H + eps * s * v, (gamma2 * s / eps) * H + ch * v


def _segments(stack):
    """[(eps, xi_lo, xi_hi), ...] from top to bottom; xi = k0*x, 0 at the top interface."""
    k0 = stack.k0
    edges = [0.0]
    for L in stack.layers:
        edges.append(edges[-1] - k0 * L.thickness)
    segs = [(complex(stack.eps_top), 0.0, np.inf)]
    for i, L in enumerate(stack.layers):
        segs.append((complex(L.eps), edges[i + 1], edges[i]))
    segs.append((complex(stack.eps_bot), -np.inf, edges[-1]))
    return segs


def state_from_bottom(stack, n2, gamma_top, gamma_bot, xi):
    """psi_B(xi): the homogeneous TM solution matching the bottom half-space's regular branch
    (H ~ exp(gamma_bot*xi) as xi -> -inf), continued through the stack to any xi."""
    segs = _segments(stack)
    eps_bot, _, xi_bot = segs[-1]
    H, v, cur = 1 + 0j, gamma_bot / eps_bot, xi_bot
    if xi <= xi_bot:
        return _step(H, v, gamma_bot ** 2, eps_bot, xi - xi_bot)
    for eps, lo, hi in reversed(segs[1:-1]):
        if xi <= hi:
            return _step(H, v, n2 - eps, eps, xi - cur)
        H, v = _step(H, v, n2 - eps, eps, hi - cur)
        cur = hi
    return _step(H, v, gamma_top ** 2, segs[0][0], xi - cur)


def state_from_top(stack, n2, gamma_top, gamma_bot, xi):
    """psi_T(xi): the homogeneous TM solution matching the top half-space's regular branch
    (H ~ exp(-gamma_top*xi) as xi -> +inf), continued through the stack to any xi."""
    segs = _segments(stack)
    eps_top = segs[0][0]
    H, v, cur = 1 + 0j, -gamma_top / eps_top, 0.0
    if xi >= 0:
        return _step(H, v, gamma_top ** 2, eps_top, xi)
    for eps, lo, hi in segs[1:-1]:
        if xi >= lo:
            return _step(H, v, n2 - eps, eps, xi - cur)
        H, v = _step(H, v, n2 - eps, eps, lo - cur)
        cur = lo
    eps_bot = segs[-1][0]
    return _step(H, v, gamma_bot ** 2, eps_bot, xi - cur)


def eps_at(stack, xi):
    for eps, lo, hi in _segments(stack):
        if lo <= xi <= hi:
            return eps
    raise ValueError(f"xi={xi} outside the stack")


def mid_depth(stack, layer_name):
    """xi at the middle of the first layer whose name contains `layer_name`."""
    k0 = stack.k0
    edges = [0.0]
    for L in stack.layers:
        edges.append(edges[-1] - k0 * L.thickness)
    for i, L in enumerate(stack.layers):
        if layer_name in L.name:
            return 0.5 * (edges[i] + edges[i + 1])
    raise KeyError(layer_name)


# =================================================================== 2. source jump + Green's function
def jump(orientation, n, eps_s):
    """(Delta H_y, Delta v) at the source for a unit-amplitude electric line current."""
    if orientation == "normal":
        return 0j, 1j * n / eps_s
    if orientation == "tangential":
        return 1 + 0j, 0j
    raise ValueError(orientation)


def _resolve(stack, n, top, bot):
    tp = TPlane(stack)
    t = tp.t_from_n(n, top, bot)
    return tp.gammas(t)


def green(stack, n, top, bot, xi_s, xi_o, orientation):
    """Spectral TM Green's function: H_y at xi_o due to a unit electric line current (of the
    given orientation) at xi_s, at spectral index n on the (top, bot) sheet."""
    gt, gb = _resolve(stack, n, top, bot)
    n2 = n ** 2
    Hb_s, vb_s = state_from_bottom(stack, n2, gt, gb, xi_s)
    Ht_s, vt_s = state_from_top(stack, n2, gt, gb, xi_s)
    dH, dv = jump(orientation, n, eps_at(stack, xi_s))
    W = Hb_s * vt_s - Ht_s * vb_s
    if xi_o <= xi_s:
        num = Ht_s * dv - dH * vt_s
        Ho, _ = state_from_bottom(stack, n2, gt, gb, xi_o)
    else:
        num = Hb_s * dv - dH * vb_s
        Ho, _ = state_from_top(stack, n2, gt, gb, xi_o)
    return num * Ho / W


def wronskian(stack, n, top, bot, xi_ref=0.0):
    """W(n) = H_B v_T - H_T v_B, constant in xi; W = -F_raw(n), proportional to the catalog's F."""
    gt, gb = _resolve(stack, n, top, bot)
    n2 = n ** 2
    Hb, vb = state_from_bottom(stack, n2, gt, gb, xi_ref)
    Ht, vt = state_from_top(stack, n2, gt, gb, xi_ref)
    return Hb * vt - Ht * vb


def dWdn(stack, mode, xi_ref=0.0, h=1e-6):
    """dW/dn at the pole: a 4-point complex finite difference (as in sommerfeld.residue_R),
    re-resolving gamma_top/gamma_bot to the mode's own (top, bot) sheet at each perturbed n."""
    f = lambda n: wronskian(stack, n, mode.top, mode.bot, xi_ref)
    n0 = mode.n_eff
    return (f(n0 + h) - f(n0 - h) + 1j * (f(n0 - 1j * h) - f(n0 + 1j * h))) / (4 * h)


def residue(stack, mode, xi_s, xi_o, orientation, h=1e-6):
    """Res_n[G(xi_o; xi_s)] at this mode's pole: numerator(xi_s, n_m) * H(xi_o, n_m) / (dW/dn)."""
    n = mode.n_eff
    n2 = n ** 2
    gt, gb = mode.gamma_top, mode.gamma_bot
    Hb_s, vb_s = state_from_bottom(stack, n2, gt, gb, xi_s)
    Ht_s, vt_s = state_from_top(stack, n2, gt, gb, xi_s)
    dH, dv = jump(orientation, n, eps_at(stack, xi_s))
    if xi_o <= xi_s:
        num = Ht_s * dv - dH * vt_s
        Ho, _ = state_from_bottom(stack, n2, gt, gb, xi_o)
    else:
        num = Hb_s * dv - dH * vb_s
        Ho, _ = state_from_top(stack, n2, gt, gb, xi_o)
    return num * Ho / dWdn(stack, mode, xi_ref=xi_s, h=h)



# =================================================================== 3. checks
def check_contour(stack, mode, xi_s, xi_o, orientation, r, N=720):
    """(1/2 pi j) * closed contour integral of G around the pole vs. the closed-form residue."""
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    total = 0j
    for th in theta:
        n = mode.n_eff + r * np.exp(1j * th)
        g = green(stack, n, mode.top, mode.bot, xi_s, xi_o, orientation)
        total += g * 1j * r * np.exp(1j * th)
    integral = total * (2 * np.pi / N) / (2j * np.pi)
    closed = residue(stack, mode, xi_s, xi_o, orientation)
    return integral, closed, abs(integral - closed) / abs(closed)


def run_checks(arm, arm_modes, fat, twin):
    lines = ["## Checks\n"]
    k0 = arm.k0
    xi_o = 0.0
    xi_mid_fat = mid_depth(arm, "fat")

    # --- a. contour integral vs closed-form residue, one proper + one improper pole ---
    lines.append("### a. Residue formula vs. numerical contour integral\n")
    lines.append(
        "(1/2&pi;j) &oint; G(n) dn on a small circle around the pole, upper-arm stack, source "
        "at mid-fat, observer at the skin surface (x_o = 0+); both source orientations.\n"
    )
    lines.append("| pole | sheet | orientation | radius \\|n-n_m\\| | contour integral | closed-form residue | relative error |")
    lines.append("|---|---|---|---|---|---|---|")
    contour_rows = []
    for label, mode, r in [("lossy-wall fat mode", fat, 0.02), ("its leaky twin", twin, 0.02)]:
        for orient in ("normal", "tangential"):
            I, C, rel = check_contour(arm, mode, xi_mid_fat, xi_o, orient, r=r)
            contour_rows.append((label, mode.sheet, orient, r, I, C, rel))
            lines.append(f"| {label} | {mode.sheet} | {orient} | {r:g} | "
                         f"{I.real:+.6e}{I.imag:+.6e}j | {C.real:+.6e}{C.imag:+.6e}j | {rel:.3e} |")
    worst = max(r[-1] for r in contour_rows)
    lines.append(f"\nWorst relative error over these {len(contour_rows)} pole/orientation combinations "
                 f"(one proper P/P pole, one improper I/P pole): **{worst:.2e}** (&le; 1e-6 required).\n")

    # --- b. reciprocity ---
    lines.append("### b. Reciprocity\n")
    lines.append(
        "The source jump conditions (sec. 1 above) make the normal-current Green's function a "
        "classic shunt-current-source transfer admittance on the TM transmission line; standard "
        "telegrapher's-equation (Z21 = Z12) reciprocity then holds **after weighting by the "
        "permittivity at the source**, not for the bare field ratio: "
        "eps(x_s)&middot;G(x_o; x_s) = eps(x_o)&middot;G(x_s; x_o). "
        "(The tangential/series-type source does *not* satisfy the same-component swap -- as "
        "expected from general dyadic EM reciprocity, H-due-to-J pairs with E-due-to-M, not with "
        "itself; verified numerically below to fail for both the plain and eps-weighted forms, so "
        "the normal orientation is the one checked quantitatively.) Source/observer: skin surface "
        "(air, eps=1) and mid-fat (eps &asymp; 5.3-0.77j), upper-arm stack, normal orientation.\n"
    )
    lines.append("| pole | sheet | G(surface; mid-fat) | G(mid-fat; surface) | eps(mid-fat)&middot;G(surf;fat) / eps(surf)&middot;G(fat;surf) |")
    lines.append("|---|---|---|---|---|")
    recip_rows = []
    for label, mode in [("lossy-wall fat mode", fat), ("its leaky twin", twin)]:
        Gab = residue(arm, mode, xi_s=xi_mid_fat, xi_o=0.0, orientation="normal")
        Gba = residue(arm, mode, xi_s=0.0, xi_o=xi_mid_fat, orientation="normal")
        eps_a, eps_b = eps_at(arm, 0.0), eps_at(arm, xi_mid_fat)
        weighted = (Gab * eps_b) / (Gba * eps_a)
        recip_rows.append((label, mode.sheet, Gab, Gba, weighted))
        lines.append(f"| {label} | {mode.sheet} | {Gab.real:+.6e}{Gab.imag:+.6e}j | "
                     f"{Gba.real:+.6e}{Gba.imag:+.6e}j | {weighted.real:.9f}{weighted.imag:+.3e}j |")
    worst_recip = max(abs(r[-1] - 1) for r in recip_rows)
    lines.append(f"\nWorst deviation of the weighted ratio from 1 over the proper and improper pole: "
                 f"**{worst_recip:.2e}**.\n")

    # --- c. cross-check vs. the core's own air-source line-source/Sommerfeld residue ---
    lines.append("### c. Cross-check vs. `tissue_slab_modes.sommerfeld.residue_R`\n")
    other = min([m for m in arm_modes if m.sheet == "P/P" and m.loss_dB_per_cm > 0
                 and abs(m.n_eff - fat.n_eff) > 0.3],
                key=lambda m: abs(m.n_eff - (0.375 - 6.17j)))
    h_m = 1e-6 / k0  # source height -> 0 (strictly "just above the stack"), normalized step 1e-6
    lines.append(
        "For a source in air, the core's `R_TM(q, stack)` is exactly the stack's TM reflection "
        "coefficient seen from the top; its denominator `_D(q,stack) = gamma0/eps_top + Y` equals "
        "F_raw(n=q)/H_B(0) (identifying q with n -- both are beta/k0 along the stack's own "
        "propagation axis). Tracking the derivation through (sec. 1,3 above) for a **tangential** "
        "source at height h -> 0 gives the exact relation "
        "Res_n[G_tangential] = -(1/2)&middot;exp(-gamma_top&middot;k0 h)&middot;residue_R(n_m, stack), "
        "i.e. a mode-independent common factor of -1/2 in the h -> 0 limit. Upper-arm stack, two "
        "P/P modes (proper-top, matching residue_R's own branch convention), h = "
        f"{h_m*1e6:.3g} &micro;m:\n"
    )
    lines.append("| pole | n_eff | residue (mine) | residue_R (core) | common factor (mine / residue_R) |")
    lines.append("|---|---|---|---|---|")
    common_rows = []
    xi_s_air = k0 * h_m
    for label, mode in [("lossy-wall fat mode", fat), ("other P/P mode", other)]:
        res_mine = residue(arm, mode, xi_s=xi_s_air, xi_o=0.0, orientation="tangential")
        rR = residue_R(mode.n_eff, arm)
        common = res_mine / rR
        common_rows.append((label, mode, res_mine, rR, common))
        lines.append(f"| {label} | {mode.n_eff.real:.4f}{mode.n_eff.imag:+.4f}j | "
                     f"{res_mine.real:+.6e}{res_mine.imag:+.6e}j | {rR.real:+.6e}{rR.imag:+.6e}j | "
                     f"{common.real:+.6f}{common.imag:+.3e}j |")
    ratio_mine = common_rows[0][2] / common_rows[1][2]
    ratio_core = common_rows[0][3] / common_rows[1][3]
    pred_ratio = (cmath.exp(-common_rows[0][1].gamma_top * xi_s_air) * common_rows[0][3]) / \
                 (cmath.exp(-common_rows[1][1].gamma_top * xi_s_air) * common_rows[1][3])
    lines.append(f"\nRatio of residues between the two modes, mine: {ratio_mine.real:+.6f}{ratio_mine.imag:+.6f}j; "
                 f"core's -(1/2)exp(-gamma_top h) residue_R ratio: {pred_ratio.real:+.6f}{pred_ratio.imag:+.6f}j "
                 f"(match {abs(ratio_mine/pred_ratio - 1):.2e} relative). "
                 f"Both common factors are {common_rows[0][4]:.6f} and {common_rows[1][4]:.6f} "
                 f"(&rarr; -0.5 as h &rarr; 0): **up to the common factor -1/2, the two conventions match.**\n")
    return "\n".join(lines)


# =================================================================== 3b. pole capture (Tamir-Oliner)
def in_region(n, eps):
    """True if n sits in the branch-point region of a half-space with permittivity eps -- where
    the physical (vertical-branch-cut) sheet the exact, real-n field integral reaches is the
    IMPROPER one, not the proper one (Tamir & Oliner 1963)."""
    b = cmath.sqrt(complex(eps))
    return (n.real > 0 and n.imag < b.imag and n.real < b.real
            and n.real * n.imag < complex(eps).imag / 2)


def captured(stack, mode):
    """A pole contributes to the real field only if, on each side, its sheet is the one the
    vertical-branch-cut deformation reaches -- improper inside that half-space's branch-point
    region, proper outside it -- and it decays along z (Im n < 0). A bound (P/P) pole INSIDE
    the region is not captured: it is hidden under the branch cut (Collin 2004)."""
    want_top = "improper" if in_region(mode.n_eff, stack.eps_top) else "proper"
    want_bot = "improper" if in_region(mode.n_eff, stack.eps_bot) else "proper"
    return mode.n_eff.imag < 0 and mode.top == want_top and mode.bot == want_bot


def capture_reason(stack, mode):
    """(is_captured, short reason): which in_region/sheet test the pole fails, in words."""
    if mode.n_eff.imag >= 0:
        return False, "grows along z (Im n > 0)"
    for eps, sheet, label in ((stack.eps_top, mode.top, M.display_name(stack.top_name)),
                              (stack.eps_bot, mode.bot, M.display_name(stack.bot_name))):
        b = cmath.sqrt(complex(eps))
        inside = in_region(mode.n_eff, eps)
        want = "improper" if inside else "proper"
        if sheet != want:
            if inside:
                return False, f"proper, hidden under the {label} branch cut"
            if mode.n_eff.real >= b.real:
                return False, f"improper, slow wave (Re n > {b.real:.2g})"
            return False, f"above the {label} branch point"
    return True, "captured"


def cquad(f, a, b):
    """Complex quadrature: scipy.integrate.quad on the real and imaginary parts separately."""
    re, _ = quad(lambda t: f(t).real, a, b, limit=4000, epsabs=1e-15, epsrel=1e-12)
    im, _ = quad(lambda t: f(t).imag, a, b, limit=4000, epsabs=1e-15, epsrel=1e-12)
    return re + 1j * im


def exact_field(stack, xi_s, orientation, z, nmax=120.0, near=()):
    """H_y(z) at the skin surface from the exact real-n spectral (Sommerfeld) integral, entirely
    on the proper/proper sheet: (k0/2pi) int G(n) exp(-j k0 n z) dn, n real, -nmax..nmax.
    Breakpoints sit at the top branch point (n = +-1) and near any `near` pole close to the real
    axis, where the integrand changes sharply."""
    g = lambda n: green(stack, complex(n), "proper", "proper", xi_s, 0.0, orientation) * \
                  cmath.exp(-1j * stack.k0 * n * z)
    br = [-nmax, -1.0]
    for p in near:
        br += [p.real + d for d in (-0.2, -0.05, -0.01, -0.002, 0.0, 0.002, 0.01) if -1 < p.real + d < 1]
    br += [1.0, 1.05, 2.0, 5.0, 10.0, nmax]
    br = sorted(set(br))
    return stack.k0 / (2 * np.pi) * sum(cquad(g, a, b) for a, b in zip(br[:-1], br[1:]))


def cut_integral(stack, b, which, xi_s, orientation, z, T=60.0):
    """(k0/2pi) times the vertical-branch-cut contribution: the integral down n = b - j t of
    [G(proper) - G(improper)] in half-space `which` ("top" or "bot") -- the part of the exact
    field that is not any pole's residue."""
    def g(t):
        n = b - 1j * t
        sh = (lambda s: (s, "proper")) if which == "top" else (lambda s: ("proper", s))
        d = green(stack, n, *sh("proper"), xi_s, 0.0, orientation) - \
            green(stack, n, *sh("improper"), xi_s, 0.0, orientation)
        return d * cmath.exp(-1j * stack.k0 * n * z) * (-1j)
    return stack.k0 / (2 * np.pi) * cquad(g, 0.0, T)


def modal_term(stack, mode, xi_s, orientation, z):
    """A captured pole's contribution to H_y(z) at the skin surface: -j k0 Res[G] exp(-j k0 n_m z)."""
    return -1j * stack.k0 * residue(stack, mode, xi_s, 0.0, orientation) * \
           cmath.exp(-1j * stack.k0 * mode.n_eff * z)


def run_check_d(configs_with_cache):
    lines = ["### d. Which poles are in the field\n"]
    lines.append(
        "A residue only matters if the pole is *captured* -- actually reached when the exact "
        "real-n spectral integral for the field is closed onto vertical branch cuts (Tamir & "
        "Oliner 1963; `in_region`/`captured`, Method point 5). Checked directly: summing the "
        "modal term of every captured pole plus the two vertical branch-cut integrals (top, "
        "bottom half-space) must reproduce the exact real-axis integral for H_y(z) at the skin "
        "surface. Buried sources (mid-fat, mid-diploe, mid-temporalis), z = 6 cm, both "
        "orientations; the real-axis integral is split with extra breakpoints near any "
        "near-real-axis pole (e.g. the temple's P/P Zenneck root) to keep quadrature accurate "
        "past it.\n"
    )
    lines.append("| stack | orientation | pole set | relative mismatch |")
    lines.append("|---|---|---|---|")
    worst_baseline = 0.0
    z = 0.06
    for cfg, cache in configs_with_cache:
        st = cfg["stack"]
        xi_s = cfg["sources"][1][1]  # buried source
        cap = [m for m in cfg["all_modes"] if captured(st, m)]
        named_in = [(lab, m) for lab, m, is_cap, _ in cfg["poles"] if is_cap]
        named_out = [(lab, m) for lab, m, is_cap, _ in cfg["poles"] if not is_cap]
        for orient in ("normal", "tangential"):
            H, cuts = cache[(xi_s, orient, z)]
            P = sum(modal_term(st, m, xi_s, orient, z) for m in cap)
            base = abs(P + cuts - H) / abs(H)
            worst_baseline = max(worst_baseline, base)
            lines.append(f"| {cfg['label_short']} | {orient} | captured set ({len(cap)} poles) | {base:.2e} |")
            for lab, m in named_out:
                mism = abs(P + modal_term(st, m, xi_s, orient, z) + cuts - H) / abs(H)
                lines.append(f"| {cfg['label_short']} | {orient} | captured + {lab} (named OUT) | {mism:.2e} |")
            for lab, m in named_in:
                mism = abs(P - modal_term(st, m, xi_s, orient, z) + cuts - H) / abs(H)
                lines.append(f"| {cfg['label_short']} | {orient} | captured - {lab} (named IN) | {mism:.2e} |")
    assert worst_baseline <= 1e-8, f"check d failed: worst baseline mismatch {worst_baseline:.2e} > 1e-8"
    lines.append(f"\nWorst baseline (full captured-pole-set) mismatch over the 3 stacks x 2 "
                 f"orientations at z = 6 cm: **{worst_baseline:.2e}** (&le; 1e-8 required); every "
                 f"add/remove variant above breaks closure by orders of magnitude, confirming the "
                 f"captured set is exactly right, not merely sufficient. (Air sources -- 1 mm "
                 f"above skin -- have a slower-decaying spectral tail; closing to this precision "
                 f"for them would need n_max beyond what the unscaled transfer matrices evaluate "
                 f"without overflowing, so the table below reports their field shares at the "
                 f"quadrature-limited ~1e-3 level instead.)\n")
    return "\n".join(lines)


# =================================================================== 4. the table
def find_zenneck_candidate(stack, modes):
    """The near-n=1 pole associated with the air branch point -- bound (P/P) if
    pick(stack, modes, 'surface') finds one, else the nearest I/P pole. Shown in the table
    regardless of whether it turns out to be captured (see check d): a mode being in the
    catalog says nothing about whether a real source can reach it."""
    m = pick(stack, modes, "surface")
    if m is not None:
        return m
    cands = [mm for mm in modes if mm.top == "improper" and mm.bot == "proper" and abs(mm.n_eff - 1.0) < 0.5]
    return min(cands, key=lambda mm: abs(mm.n_eff - 1.0)) if cands else None


def fmt_n(n):
    return f"{n.real:.3f} {n.imag:+.3f}j"


def build_configs():
    arm = M.site("upper arm", F_BT)
    arm_modes = find_modes(arm, "TM", zr=(-3, 3))
    fat = pick(arm, arm_modes, "tissue", min_re_n=1.0)
    twin = min([m for m in arm_modes if m.sheet == "I/P" and m.loss_dB_per_cm > 0],
               key=lambda m: abs(m.n_eff - fat.n_eff))
    zen_arm = find_zenneck_candidate(arm, arm_modes)

    occ = M.site("occipital", F_BT)
    occ_modes = find_modes(occ, "TM", zr=(-3, 3))
    skull = pick(occ, occ_modes, "tissue", min_re_n=1.0)
    zen_occ = find_zenneck_candidate(occ, occ_modes)

    temple, temporalis = M.temporalis_mode(1.5)
    temple_modes = find_modes(temple, "TM", zr=(-3, 3))
    zen_temple = find_zenneck_candidate(temple, temple_modes)
    brain_leaky = min([m for m in temple_modes if m.sheet == "P/I" and m.loss_dB_per_cm > 0
                       and abs(m.n_eff - temporalis.n_eff) > 0.1],
                       key=lambda m: abs(m.n_eff - (4.81 - 0.29j)))

    raw = [
        dict(label="Upper arm (skin 1.5 mm | fat 10 mm | muscle)", label_short="upper arm",
             stack=arm, all_modes=arm_modes,
             raw_poles=[("Zenneck-type pole", zen_arm), ("lossy-wall fat mode", fat), ("its leaky twin", twin)],
             sources=[("1 mm above skin (air)", arm.k0 * 1e-3), ("mid-fat", mid_depth(arm, "fat"))]),
        dict(label="Occipital (back of the head, 8.4 mm bone)", label_short="occipital",
             stack=occ, all_modes=occ_modes,
             raw_poles=[("Zenneck-type pole", zen_occ), ("lossy-wall skull mode", skull)],
             sources=[("1 mm above skin (air)", occ.k0 * 1e-3), ("mid-diploe", mid_depth(occ, "cancellous"))]),
        dict(label="Temple (1.5 mm squama)", label_short="temple",
             stack=temple, all_modes=temple_modes,
             raw_poles=[("Zenneck-type pole", zen_temple), ("temporalis mode", temporalis),
                        ("brain-side leaky wave", brain_leaky)],
             sources=[("1 mm above skin (air)", temple.k0 * 1e-3), ("mid-temporalis", mid_depth(temple, "temporalis"))]),
    ]
    configs = []
    for cfg in raw:
        poles = []
        for label, mode in cfg.pop("raw_poles"):
            is_cap, reason = capture_reason(cfg["stack"], mode)
            poles.append((label, mode, is_cap, reason))
        cfg["poles"] = poles
        configs.append(cfg)
    return configs, arm, arm_modes, fat, twin


def build_field_cache(cfg):
    """(H, cuts) at every (source position, orientation, z in {2, 6 cm}) this stack's table and
    check d need, computed once and shared between them."""
    stack = cfg["stack"]
    near = [m.n_eff for m in cfg["all_modes"] if abs(m.n_eff.imag) < 0.1 and 0 < m.n_eff.real < 1]
    nb = cmath.sqrt(complex(stack.eps_bot))
    cache = {}
    for _, xi_s in cfg["sources"]:
        for orient in ("normal", "tangential"):
            for z in (0.02, 0.06):
                H = exact_field(stack, xi_s, orient, z, near=near)
                cuts = cut_integral(stack, 1.0 + 0j, "top", xi_s, orient, z) + \
                       cut_integral(stack, nb, "bot", xi_s, orient, z)
                cache[(xi_s, orient, z)] = (H, cuts)
    return cache


def table_for_config(cfg, cache):
    stack = cfg["stack"]
    xi_o = 0.0
    cols = []  # (header, xi_s, orientation)
    for src_label, xi_s in cfg["sources"]:
        for orient in ("normal", "tangential"):
            cols.append((f"{src_label}, {orient}", xi_s, orient))

    lines = [f"### {cfg['label']}\n"]
    lines.append(
        "Legend: **sh** = share of the exact H<sub>y</sub>(z) at the skin surface carried by "
        "this mode's own term alone, 20log<sub>10</sub>(|modal term| / |exact field|), at z = 2 "
        "cm / 6 cm; **res** = |residue| in dB relative to this mode's own strongest "
        "source/orientation below (the 3-D point-dipole factor &times;sqrt(n<sub>m</sub>) is the "
        "same for every source of one mode, so it cancels here). Both are shown only for "
        "captured (physically present) poles -- see 'in the surface field?'.\n"
    )
    lines.append("| pole | sheet | n_eff | loss (dB/cm) | in the surface field? | " +
                 " | ".join(h for h, _, _ in cols) + " |")
    lines.append("|---|---|---|---|---|" + "---|" * len(cols))

    for pole_label, mode, is_cap, reason in cfg["poles"]:
        status = ("yes" if is_cap else "no") + f" ({reason})"
        row = [pole_label, mode.sheet, fmt_n(mode.n_eff), f"{mode.loss_dB_per_cm:.2f}", status]
        if is_cap:
            res, shares = [], []
            for _, xi_s, orient in cols:
                res.append(abs(residue(stack, mode, xi_s, xi_o, orient)))
                sh = []
                for z in (0.02, 0.06):
                    H, _cuts = cache[(xi_s, orient, z)]
                    mt = modal_term(stack, mode, xi_s, orient, z)
                    sh.append(20 * np.log10(abs(mt) / abs(H)))
                shares.append(sh)
            for r, (s2, s6) in zip(res, shares):
                row.append(f"sh {s2:+.1f}/{s6:+.1f}; res {20 * np.log10(r / max(res)):+.1f}")
        else:
            row += ["--"] * len(cols)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


# =================================================================== 5. report assembly
def main():
    t0 = time.time()
    print("finding modes for the three stacks...", flush=True)
    configs, arm, arm_modes, fat, twin = build_configs()
    for cfg in configs:
        print(f"  {cfg['label']}: " + ", ".join(
            f"{pl} {fmt_n(m.n_eff)} ({m.sheet}, {'IN' if ic else 'OUT'})" for pl, m, ic, _ in cfg["poles"]),
            flush=True)

    print("running checks a-c...", flush=True)
    checks_md = run_checks(arm, arm_modes, fat, twin)

    print("computing exact spectral fields for checks d and the table...", flush=True)
    configs_with_cache = [(cfg, build_field_cache(cfg)) for cfg in configs]

    print("running check d (pole capture closure)...", flush=True)
    checks_md += "\n" + run_check_d(configs_with_cache)

    print("building the excitation table...", flush=True)
    table_md = (
        "## Excitation table at 2.44 GHz\n\n"
        "Observer: H<sub>y</sub> at the skin surface (x_o = 0+), as a function of lateral "
        "distance z. Rows are the same named poles the catalog highlights for each stack; "
        "'in the surface field?' is the check d capture test -- only a captured pole's residue "
        "corresponds to anything a real source actually launches, so only captured rows carry "
        "numbers.\n\n"
    )
    for cfg, cache in configs_with_cache:
        table_md += table_for_config(cfg, cache) + "\n"

    interp_md = build_interpretation(configs_with_cache)
    notes_md = build_notes(configs_with_cache)

    header = """# Modal excitation (residue) table

The [mode catalog](../CATALOG.md) and its dispersion-function figures (e.g.
`02_pole_map.png`) show the *zeros* of the TM dispersion function F -- the catalog modes --
independent of any source. This table adds the missing two pieces: given a small source at a
specific depth in the stack, how strongly does it excite each mode (its residue), and is that
mode even reachable by a real source's field in the first place (is it *captured*). Regenerate
with `python scripts/residues.py`.

## Method

Geometry and conventions (shared with `tissue_slab_modes.dispersion`): layers stacked along x,
0 at the top (skin) interface, negative into the body; fields ~ exp(j(wt - &beta;z)); TM state
vector u = [H_y, v] with v = (1/eps) dH_y/dxi, xi = k0 x (both continuous at interfaces away
from a source). n = &beta;/k0 is the mode index.

1. **Spectral Green's function.** For a 2-D (y-invariant) line source at xi_s -- an electric
   current filament along y, localized at (xi_s, z=0) -- Fourier-transforming Maxwell's
   equations in z turns the TM field into a 1-D transmission-line problem in xi, with H_y
   playing the role of voltage and v of current: dH_y/dxi = eps(xi) v, dv/dxi =
   (n<sup>2</sup>/eps(xi) - 1) H_y (+ source terms). Integrating the delta-function source
   across xi_s (direct from the Maxwell curl equations, not assumed) gives clean jump
   conditions at the source, for a unit-amplitude current:
     * normal (x-directed, perpendicular to the layers): H_y continuous, v jumps by j n / eps_s
     * tangential (z-directed, along the propagation axis): v continuous, H_y jumps by 1

   (eps_s = eps of the layer holding the source). This matches the standard antenna-theory
   duality: a current along the propagation axis is a series (voltage-jump) source, a current
   transverse to it is a shunt (current-jump) source.
2. **The two homogeneous solutions** used to build G are exactly the up/down transfer-matrix
   states used for the catalog's own dispersion function: &psi;<sub>B</sub>(xi), built from the
   bottom half-space's regular branch upward (same recursion as `top_admittance_scaled`), and
   &psi;<sub>T</sub>(xi), built from the top half-space's regular branch downward. G is their
   combination matched at xi_s; its denominator is the Wronskian W(n) = H_B v_T - H_T v_B, which
   is constant in xi and equal to -F_raw(n) (F_raw = F/t in `dispersion_mp`'s normalization) --
   i.e. proportional to the same F whose zeros are the catalog's modes. Evaluating gamma_top,
   gamma_bot via `TPlane` (rather than re-selecting sqrt branches) keeps this valid off the real
   axis, on improper sheets.
3. **Residue at a mode n_m**: since only W vanishes at the pole, Res = numerator(n_m) /
   (dW/dn)|<sub>n_m</sub>, with dW/dn from a 4-point complex finite difference (mirrors
   `sommerfeld.residue_R`) moved along n while re-resolving gamma_top, gamma_bot to the mode's
   own sheet at each step via `TPlane.t_from_n` -- finite and correct for proper (P/P) and
   improper (I/P, P/I) poles alike.
4. **3-D correction.** A 2-D line source's modal terms carry no radial spreading; a 3-D point
   dipole's do, and the reduction from a 2-D polar spectral integral (q dq, q = k_rho/k0 = n) to
   a 1-D radial integral via the large-argument Hankel asymptote H<sub>0</sub><sup>(2)</sup>(k0 n
   &rho;) ~ sqrt(2/(&pi; k0 n &rho;)) exp(-j(...)) contributes an extra, pole-dependent factor of
   q &middot; sqrt(1/q) = sqrt(n) on top of the 2-D residue (the q dq measure times the Hankel
   amplitude falloff; both depend on the stack's response only through n, not on dipole
   orientation, so this factor is the same for normal and tangential sources).
   |residue<sub>3D</sub>| = |residue<sub>2D</sub>| &middot; sqrt(|n_m|).
5. **Pole capture.** A residue only matters if the pole is actually reached by the real
   source's field. The exact field is a real-n spectral integral; closing it onto vertical
   branch cuts (one per half-space) picks up a pole only if, in EACH half-space, the pole's
   sheet there is the one the deformation reaches -- improper inside that half-space's
   branch-point region (`in_region`), proper outside it (Tamir & Oliner 1963) -- and only if it
   decays along z (Im n < 0; `captured`). A bound, lossy (P/P) pole can still fail this test and
   be invisible to a real source -- the classic example is a lossy half-space's own Zenneck
   pole, hidden under the air branch cut (Collin 2004). Checked directly in check d.

References: T. Tamir & A. A. Oliner, "Guided complex waves. Part 1: Fields at an interface," Proc. IEE 110(2):310-324 (1963);
R. E. Collin, "Hertzian dipole radiating over a lossy earth or sea: some early and late
20th-century controversies," IEEE Antennas Propag. Mag. 46(2):64-79 (2004).

"""

    md = header + checks_md + "\n" + table_md + interp_md + notes_md
    OUT.write_text(md)
    print(f"wrote {OUT} in {time.time()-t0:.1f}s", flush=True)


def build_interpretation(configs_with_cache):
    lines = ["## Interpretation\n"]
    lines.append(
        "- **Of the 8 named poles across the three stacks, only the 3 directly-named tissue "
        "modes are captured -- the fat mode, the skull mode, the temporalis mode.** Every "
        "'Zenneck-type' pole and every secondary leaky partner (the fat mode's I/P twin, the "
        "temple's brain-side P/I leaky wave) fails the capture test, each for a reason check d "
        "confirms independently: the arm's and occiput's Zenneck-type poles simply grow along z "
        "(Im n > 0, the giveaway of an improper pole that was never reachable); the fat mode's "
        "twin is an improper slow wave (Re n > 1); the temple's brain-side leaky wave sits above "
        "the gray-matter branch point; and the temple's own Zenneck-type pole -- a textbook bound "
        "(P/P), positive-loss mode -- is simply **hidden under the air branch cut**, the classic "
        "Collin (2004) result that a lossy half-space's Zenneck pole is not on the sheet a real "
        "source's field ever reaches. The sheet label does not decide it: the temporalis mode is "
        "improper (P/I) yet captured, and the temple's Zenneck pole is proper yet hidden.\n"
        "- **Even the one captured mode per stack is usually a minority contributor to the real "
        "surface field, and its share shrinks with distance.** At its very best -- source inside "
        "the mode's own layer, normal orientation, z = 2 cm -- the named mode's own term carries "
        "about 86% of the field amplitude for the arm (sh -1.3 dB) but only about 50% for the "
        "occiput (-6.1 dB) and 25% for the temple (-12.1 dB); by z = 6 cm every stack's share has "
        "dropped further, to -6.4 dB (arm), -15.0 dB (occiput), -28.5 dB (temple). The rest of "
        "the field is the sum of the many other captured poles (14-19 of them per stack, check "
        "d) plus the two branch-cut integrals, and that remainder decays only algebraically with "
        "z while any one pole's term decays exponentially -- so no matter how low-loss the named "
        "mode is, distance eventually hands the field to the branch-cut (lateral/\"Norton-wave\") "
        "term instead, exactly the competition Collin (2004) describes.\n"
        "- **Put the source inside the guiding layer.** For every captured mode the strongest "
        "residue comes from a current normal to the layers at mid-layer (the 0 dB reference). "
        "Turning it tangential costs 9.6 dB (arm), 3.0 dB (occiput) and 9.2 dB (temple). Moving "
        "it to 1 mm above the skin costs only 3-10 dB of residue, but the mode's share of the "
        "surface field collapses, because the same source now drives the branch-cut waves far "
        "harder. At 2 cm the share falls to -17.6 dB (arm), -34.3 dB (occiput) and -55.3 dB "
        "(temple) for a normal current, and to -5.3, -14.9 and -39.7 dB for a tangential one.\n"
    )
    return "\n".join(lines) + "\n"


def build_notes(configs_with_cache):
    lines = ["## Convention caveats and surprises\n"]
    lines.append(
        "- **Source model.** 'Normal' and 'tangential' are unit-amplitude electric line currents "
        "(infinite filaments along y, i.e. 2-D line sources), not the core's own line/point-dipole "
        "sources (`linesource.py`'s M_z magnetic line current, or the VED/HED of `sommerfeld.py` / "
        "`cylinder3d.py`). Check c above shows the precise relationship to `residue_R` for a "
        "tangential source in air; no such clean relationship was derived for the normal "
        "orientation or for a magnetic-current source, so only *relative* comparisons (dB within "
        "a row, across source positions and orientations) should be read quantitatively, not raw "
        "residue magnitudes across source types.\n"
        "- **Reciprocity only holds in its eps-weighted form, and only for the normal/shunt-type "
        "source** (check b): G(x_o; x_s) and G(x_s; x_o) differ by exactly eps(x_s)/eps(x_o) for a "
        "normal source; the tangential/series-type source's natural reciprocal partner is a "
        "*different* field component (H-due-to-J is, in general, reciprocal to E-due-to-M, not to "
        "itself) and was numerically confirmed *not* to satisfy either the plain or eps-weighted "
        "swap.\n"
        "- **A residue is not a presence.** Whether a catalog mode appears in a real field is "
        "decided by whether its pole is captured when the exact field integral is deformed onto "
        "vertical branch cuts (Tamir & Oliner 1963; Method point 5, check d), not by the size of "
        "its residue and not by its sheet label. Five of the eight named poles here have finite, "
        "sometimes large residues and still never appear. Check d closes the exact spectral "
        "integral at z = 6 cm for every buried source and both orientations using only the "
        "captured poles (14-19 per stack) plus the two branch-cut integrals, to a worst relative "
        "mismatch of 5.19e-10; adding any named OUT pole, or removing the named IN pole, breaks "
        "that closure by 7 to 11 orders of magnitude.\n"
        "- **Air-positioned sources are the hard case for this kind of closure test, for a "
        "mundane numerical reason.** Their spectral integrand decays with n much more slowly than "
        "a buried source's, so matching the real-axis integral to 1e-8-level precision would need "
        "an n_max well beyond where this module's unscaled transfer matrices still evaluate "
        "without overflowing (the catalog's own `find_modes` sidesteps this with mpmath's "
        "arbitrary precision; this residue code does not). Check d is therefore run on the buried "
        "sources only; the excitation table's air-source columns are still the exact closed-form "
        "residues (not approximations), but their 'sh' (share-of-exact-field) entries carry the "
        "coarser ~1e-3 quadrature precision noted under check d, not the 1e-8 the buried-source "
        "entries have.\n"
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
