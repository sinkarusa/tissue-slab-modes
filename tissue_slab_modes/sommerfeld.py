"""Hertzian dipoles above a layered half-space: Sommerfeld integrals and pole residues.

A vertical (normal-to-skin) electric dipole (VED) of moment I*l at height h above the
top interface (air side). With q = k_rho / k0, q_z = sqrt(1 - q^2) (Im q_z <= 0) and the
TM (H-field / A_z) reflection coefficient of the stack R(q), the reflected normal field
at (rho, height h) is [Sommerfeld; Banos 1966; Michalski & Mosig, IEEE TAP 45, 508 (1997);
Chew, Waves and Fields in Inhomogeneous Media, ch. 2]

    E_z^r(rho) = -(I l k0^3 / (4 pi w eps0)) \\int_0^inf R(q) q^3/q_z J0(k0 q rho) e^{-2 j k0 h q_z} dq.

Input resistance (self term, rho -> 0), normalised to the free-space value
R_fs = eta0 (k0 l)^2 / (6 pi):

    R11 / R_fs = 1 + (3/2) Re \\int_0^inf R(q) q^3/q_z e^{-2 j k0 h q_z} dq .

Check: PEC (R = 1), h -> 0 gives R11 = 2 R_fs, and for finite h the classic
R/R_fs = 1 - 3[cos(2kh)/(2kh)^2 - sin(2kh)/(2kh)^3]  [Balanis, Antenna Theory, sec. 4.7].

Guided-mode (pole) contribution. Writing J0 = (H0^(2)(.) + H0^(1)(.))/2 and closing the
q-contour in the lower half plane, every proper pole q_p of R (Im q_p < 0) contributes

    E_z^p(rho) = (j I l k0^3 / (4 w eps0)) Res_q[R](q_p) q_p^3/q_zp e^{-2 j k0 h q_zp} H0^(2)(k0 q_p rho)

(the remaining branch-cut integral around q = 1 is the space wave + Norton surface wave).
Mutual impedance Z21 = -E_z l / I; the transducer gain between two identical lossless,
conjugate-matched Hertzian dipoles is  G = |Z21|^2 / (4 R11 R22)  [e.g. Balanis ch. 8].
Because R11 includes power absorbed by the tissue, G automatically includes the
near-body antenna efficiency.

The reflection coefficient follows from the same transfer matrix as the dispersion
function: with Y = u1/u0 at the top interface (TM, u = [H_y, (1/eps) dH_y/dxi]),
    R = (gamma0/eps_top - Y) / (gamma0/eps_top + Y),   gamma0 = j q_z,
so that the poles of R are exactly the proper-sheet zeros of the dispersion function.
"""
from __future__ import annotations

import numpy as np
from scipy.special import hankel2, j0

from .dispersion import top_admittance_scaled
from .stack import Stack


def qz_of(q):
    q = np.asarray(q, dtype=complex)
    qz = np.sqrt(1 - q ** 2 + 0j)
    return np.where(qz.imag > 0, -qz, qz)


def gamma_bot_proper(q, stack):
    g = np.sqrt(np.asarray(q, dtype=complex) ** 2 - stack.eps_bot)
    return np.where(g.real < 0, -g, g)


def R_TM(q, stack: Stack, qz=None):
    """TM reflection coefficient (H_y / A_z) seen from the top half-space (air)."""
    q = np.asarray(q, dtype=complex)
    qz = qz_of(q) if qz is None else qz
    g0 = 1j * qz
    gb = gamma_bot_proper(q, stack)
    u0, u1, _ = top_admittance_scaled(q ** 2, stack, gb, "TM")
    Y = u1 / u0
    a = g0 / stack.eps_top
    return (a - Y) / (a + Y)


def _D(q, stack):
    """gamma0/eps_top + Y : vanishes at the proper poles."""
    q = np.asarray(q, dtype=complex)
    g0 = 1j * qz_of(q)
    gb = gamma_bot_proper(q, stack)
    u0, u1, _ = top_admittance_scaled(q ** 2, stack, gb, "TM")
    return g0 / stack.eps_top + u1 / u0


def residue_R(qp, stack: Stack, h=1e-7):
    """Res_q R at a proper pole qp = n_eff (numerical derivative of the denominator)."""
    qp = complex(qp)
    dD = (_D(qp + h, stack) - _D(qp - h, stack) + 1j * (_D(qp - 1j * h, stack) - _D(qp + 1j * h, stack))) / (4 * h)
    num = 2j * qz_of(qp) / stack.eps_top  # (a - Y) at the pole equals 2a
    return complex(num / dD)


_GL16 = np.polynomial.legendre.leggauss(16)


def _gl(a, b, n):
    """Composite 16-point Gauss-Legendre with ~n nodes on [a, b]."""
    panels = max(1, int(np.ceil(n / 16)))
    edges = np.linspace(a, b, panels + 1)
    x, w = _GL16
    mid, half = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
    return (mid[:, None] + half[:, None] * x).ravel(), (half[:, None] * w).ravel()


def _spectral_nodes(stack, h, rho, n_prop=2000, n_ev=40000):
    """Quadrature nodes on the real q axis: q = sin(theta) on [0,1], q = cosh(u) beyond."""
    k0 = stack.k0
    th, wth = _gl(0, np.pi / 2, max(n_prop, 32 * k0 * rho))
    q1 = np.sin(th) + 0j
    w1 = wth * np.sin(th) ** 3 + 0j                        # q^3/qz dq = sin^3 dtheta
    e1 = np.exp(-2j * k0 * h * np.cos(th))
    umax = np.arcsinh(60 / (2 * k0 * h))                   # e^{-2 k0 h sinh u} < e^{-60}
    # enough nodes for the J0(k0 q rho) oscillation (~16 nodes per period)
    n_osc = 16 * k0 * rho * np.cosh(umax) / np.pi
    u, wu = _gl(0, umax, max(n_ev, n_osc))
    q2 = np.cosh(u) + 0j
    w2 = wu * 1j * np.cosh(u) ** 3                         # q^3/qz dq = j cosh^3 du
    e2 = np.exp(-2 * k0 * h * np.sinh(u))
    return (q1, w1 * e1), (q2, w2 * e2)


def R_TE(q, stack: Stack, qz=None):
    """TE (E_y) reflection coefficient seen from the top half-space (air)."""
    q = np.asarray(q, dtype=complex)
    qz = qz_of(q) if qz is None else qz
    g0 = 1j * qz
    gb = gamma_bot_proper(q, stack)
    u0, u1, _ = top_admittance_scaled(q ** 2, stack, gb, "TE")
    Y = u1 / u0
    return (g0 - Y) / (g0 + Y)


def hed_input_resistance(stack: Stack, h: float, n_prop=2000, n_ev=40000) -> float:
    """R11 / R_fs for a horizontal (tangential) electric dipole at height h:
        R11/R_fs = 1 + (3/4) Re \\int_0^inf [R_TE/q_z - R_TM q_z] q e^{-2 j k0 h q_z} dq
    (PEC check: 1 - (3/2)[sin x/x + cos x/x^2 - sin x/x^3], x = 2 k0 h  [Balanis sec. 4.7.2])."""
    k0 = stack.k0
    th, wth = _gl(0, np.pi / 2, n_prop)
    q1 = np.sin(th) + 0j
    qz1 = np.cos(th) + 0j
    e1 = np.exp(-2j * k0 * h * np.cos(th))
    # [R_TE/qz - R_TM qz] q dq with q = sin, dq = cos dtheta
    I1 = np.sum((R_TE(q1, stack, qz1) - R_TM(q1, stack, qz1) * qz1 ** 2) * np.sin(th) * e1 * wth)
    umax = np.arcsinh(60 / (2 * k0 * h))
    u, wu = _gl(0, umax, n_ev)
    q2 = np.cosh(u) + 0j
    qz2 = -1j * np.sinh(u)
    e2 = np.exp(-2 * k0 * h * np.sinh(u))
    # q = cosh u, dq = sinh u du, qz = -j sinh u:  [R_TE/qz - R_TM qz] q dq
    I2 = np.sum((R_TE(q2, stack, qz2) * 1j - R_TM(q2, stack, qz2) * (-1j) * np.sinh(u) ** 2) * np.cosh(u) * e2 * wu)
    return float(1 + 0.75 * (I1 + I2).real)


def air_radiation_fraction(stack: Stack, h: float, orientation="normal", n=4000) -> float:
    """Power radiated into the upper (air) half-space / free-space power of the same dipole.
    VED: (3/4) \\int_0^{pi/2} sin^3 |1 + R_TM e^{-2jk0h cos}|^2 d theta
    HED: (3/8) \\int_0^{pi/2} [cos^2 |1 - R_TM e|^2 + |1 + R_TE e|^2] sin d theta
    (array factor of the dipole and its image; R evaluated at q = sin theta)."""
    th, w = _gl(0, np.pi / 2, n)
    q = np.sin(th) + 0j
    qz = np.cos(th) + 0j
    e = np.exp(-2j * stack.k0 * h * np.cos(th))
    if orientation == "normal":
        return float(0.75 * np.sum(np.sin(th) ** 3 * np.abs(1 + R_TM(q, stack, qz) * e) ** 2 * w))
    return float(0.375 * np.sum((np.cos(th) ** 2 * np.abs(1 - R_TM(q, stack, qz) * e) ** 2 +
                                 np.abs(1 + R_TE(q, stack, qz) * e) ** 2) * np.sin(th) * w))


def antenna_air_efficiency(stack: Stack, h: float, orientation="normal") -> float:
    """Fraction of the input power of a lossless dipole at height h that is radiated into
    air (the rest is absorbed by the tissue or carried by surface/guided waves)."""
    r = ved_input_resistance(stack, h) if orientation == "normal" else hed_input_resistance(stack, h)
    return air_radiation_fraction(stack, h, orientation) / r


def ved_input_resistance(stack: Stack, h: float, n_prop=2000, n_ev=40000) -> float:
    """R11 / R_fs for a VED at height h (m) above the stack."""
    (q1, w1), (q2, w2) = _spectral_nodes(stack, h, 0, n_prop, n_ev)
    I = np.sum(R_TM(q1, stack) * w1) + np.sum(R_TM(q2, stack) * w2)
    return float(1 + 1.5 * I.real)


def ved_reflected_field(stack: Stack, h: float, rho: float, n_prop=4000, n_ev=200000):
    """Normalised reflected E_z at distance rho:  K_r = \\int R q^3/qz J0 e^{..} dq."""
    k0 = stack.k0
    (q1, w1), (q2, w2) = _spectral_nodes(stack, h, rho, n_prop, n_ev)
    return complex(np.sum(R_TM(q1, stack) * w1 * j0(k0 * q1.real * rho)) +
                   np.sum(R_TM(q2, stack) * w2 * j0(k0 * q2.real * rho)))


def ved_direct_field(stack: Stack, rho: float) -> complex:
    """Free-space E_z between two side-by-side z-dipoles, in the same normalisation as
    the spectral integrals (i.e. E_z / (-(I l k0^3)/(4 pi w eps0)))."""
    k0r = stack.k0 * rho
    # E_z = -j eta0 k0 I l/(4 pi rho) [1 + 1/(j k0 rho) - 1/(k0 rho)^2] e^{-j k0 rho}
    # and eta0 k0 = k0^2/(w eps0)  =>  divide by -(k0^3/(4 pi w eps0)):
    return complex(1j / k0r * (1 + 1 / (1j * k0r) - 1 / k0r ** 2) * np.exp(-1j * k0r))


def gain_from_K(K: complex, r11: float) -> float:
    """Link transducer gain from a normalised field K (E_z = -(I l k0^3/(4 pi w eps0)) K).

    Z21 = -E_z l / I = (l^2 k0^3 / (4 pi w eps0)) K = 1.5 R_fs K   =>   G = |1.5 K|^2 / (4 r11^2).
    """
    return float(abs(1.5 * K) ** 2 / (4 * r11 ** 2))


def ved_pole_field(stack: Stack, h: float, rho: float, qp: complex, path=None) -> complex:
    """Normalised pole contribution K_p (same normalisation as ved_reflected_field).

    E_z^p = (j k0^3 I l /(4 w eps0)) Res q^3/qz e H0^(2)  =  -(I l k0^3/(4 pi w eps0)) K_p
      =>  K_p = -j pi Res q^3/qz e^{-2 j k0 h qz} H0^(2)(k0 qp rho).
    If `path` is given as a list of (qp_i, length_i) segments, the Hankel function is
    replaced by its large-argument form with the accumulated complex phase
    sum_i k0 qp_i L_i (adiabatic / WKB transport along a non-uniform head).
    """
    k0 = stack.k0
    qp = complex(qp)
    qz = complex(qz_of(qp))
    res = residue_R(qp, stack)
    if path is None:
        H = hankel2(0, k0 * qp * rho)
    else:
        phase = sum(k0 * complex(q) * L for q, L in path)
        H = np.sqrt(2 / (np.pi * k0 * qp * rho)) * np.exp(-1j * (phase - np.pi / 4))
    return complex(-1j * np.pi * res * qp ** 3 / qz * np.exp(-2j * k0 * h * qz) * H)
