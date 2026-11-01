"""Regression and analytic-limit tests.  Run:  python -m pytest -q"""
import numpy as np
import mpmath as mp
import pytest

from tissue_slab_modes import Layer, Stack, EPS0, find_modes, dispersion_mp, Rect, find_roots
from tissue_slab_modes.dispersion import dispersion_scaled, TPlane, F_symmetric_slab_tm
from tissue_slab_modes.apm import make_polisher
from tissue_slab_modes.sommerfeld import (ved_input_resistance, ved_reflected_field,
                                          ved_direct_field, gain_from_K)

F0 = 2.45e9
W = 2 * np.pi * F0
E = lambda er, s: er - 1j * s / (W * EPS0)


def fat_channel(dfat=25e-3):
    return Stack(F0, "air", 1.0, [Layer("skin", E(38.0, 1.46), 2e-3), Layer("fat", E(5.28, 0.105), dfat)],
                 "muscle", E(52.7, 1.74), "fat channel")


def test_symmetric_slab_matches_notebook_equation():
    """N-layer function vs the notebook's fRaw1 (eps_i = 2.25 slab, d = 0.8 lambda, in air)."""
    lam = 299792458 / F0
    slab = Stack(F0, "air", 1.0, [Layer("core", 2.25, 0.8 * lam)], "air", 1.0)
    f_nb = lambda b: F_symmetric_slab_tm(b, 2.25, 1.0, 0.8)

    from tissue_slab_modes.dispersion import top_admittance_scaled
    def f_tm(b):
        g = complex(np.sqrt(b ** 2 - 1 + 0j))
        u0, u1, _ = top_admittance_scaled(np.array([b ** 2]), slab, g, "TM")
        return complex(g * u0[0] + u1[0])

    from scipy.optimize import brentq
    for guess in (1.40, 1.13):
        b_nb = float(mp.re(mp.findroot(f_nb, guess)))
        b_new = brentq(lambda b: f_tm(b).real, guess - 0.05, guess + 0.05, xtol=1e-14)
        assert b_new == pytest.approx(b_nb, abs=1e-10)
    assert b_nb == pytest.approx(1.1358, abs=1e-3)


def test_scaled_function_has_same_phase_and_log():
    st = fat_channel()
    z = np.array([0.3 + 0.2j, -0.7 + 2.0j, 1.1 - 1.0j])
    F, S = dispersion_scaled(z, st)
    for zi, Fi, Si in zip(z, F, S):
        Fm = complex(dispersion_mp(zi, st))
        assert np.angle(Fi / Fm) == pytest.approx(0, abs=1e-9)
        assert np.log(abs(Fi)) + Si == pytest.approx(np.log(abs(Fm)), rel=1e-9)


def test_apm_finds_fat_ibc_mode():
    st = fat_channel()
    bound = [m for m in find_modes(st) if m.is_bound and m.loss_dB_per_cm < 3]
    assert len(bound) == 1
    m = bound[0]
    assert m.n_eff == pytest.approx(2.1645 - 0.3428j, abs=2e-4)
    assert m.loss_dB_per_cm == pytest.approx(1.529, abs=0.01)   # Asan et al. expect ~1.5 dB/cm in humans


def test_all_roots_are_roots():
    st = fat_channel(10e-3)
    for m in find_modes(st):
        assert abs(complex(dispersion_mp(m.z, st))) < 1e-8 * (1 + abs(m.t) ** 2)


def test_apm_count_is_not_aliased():
    """Abdomen (skin 2 | fat 20 | muscle) at 6.584 GHz: a coarse contour once aliased a full 2 pi
    phase turn between two samples, so the count reported phantom zeros. The leaves then divided
    by m0 ~ 0 and sent the polisher to z ~ inf, which stalled. Every root must be a root, the total
    must equal the winding number of the box (274, stable from 9k to 143k contour points), and the
    fat mode must match an independent local Newton solve."""
    import signal
    from tissue_slab_modes.materials import eps

    f = 6.583940151762977e9
    st = Stack(f, "air", 1.0, [Layer("skin", eps("skin", f), 2e-3), Layer("fat", eps("fat", f), 20e-3)],
               "muscle", eps("muscle", f))
    tp = TPlane(st)
    a = np.pi / 1000
    rect = Rect(-3 - 0.0031, 3 + 0.0047, -np.pi - 2 * a, np.pi + a)

    def stalled(*_):
        raise TimeoutError("APM stalled")
    old = signal.signal(signal.SIGALRM, stalled)
    signal.alarm(60)
    try:
        roots = find_roots(lambda z: dispersion_scaled(z, st, "TM", tp), rect,
                           polish=make_polisher(lambda z: dispersion_mp(z, st)))
        modes = find_modes(st, zr=(-3, 3))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert len(roots) == 274
    for z, _, _ in roots:
        assert abs(complex(dispersion_mp(z, st))) < 1e-8 * (1 + abs(np.exp(z)) ** 2)
    fat = min(modes, key=lambda m: abs(m.n_eff - (2.2058 - 0.2717j)))
    assert fat.n_eff == pytest.approx(2.2057863881 - 0.2716604718j, abs=1e-8)


def test_ved_over_pec_matches_balanis():
    pec = Stack(F0, "air", 1.0, [], "PEC", 1 - 1e12j)
    for h in (5e-3, 20e-3, 60e-3):
        x = 2 * pec.k0 * h
        exact = 1 - 3 * (np.cos(x) / x ** 2 - np.sin(x) / x ** 3)
        assert ved_input_resistance(pec, h) == pytest.approx(exact, rel=2e-4)


def test_free_space_link_is_friis():
    fs = Stack(F0, "air", 1.0, [], "air", 1.0 + 1e-12j)
    G = gain_from_K(ved_direct_field(fs, 0.5), 1.0)
    friis = 1.5 ** 2 * (fs.lam0 / (4 * np.pi * 0.5)) ** 2
    assert 10 * np.log10(G) == pytest.approx(10 * np.log10(friis), abs=0.02)


def test_te_symmetric_slab_matches_closed_form():
    """TE0 of a symmetric n=1.5 slab (d = 0.8 lambda): tan(kappa k0 d/2) = gamma/kappa [Chew ch. 2]."""
    from scipy.optimize import brentq
    from tissue_slab_modes.dispersion import top_admittance_scaled
    lam = 299792458 / F0
    k0d = 2 * np.pi * 0.8
    slab = Stack(F0, "air", 1.0, [Layer("core", 2.25, 0.8 * lam)], "air", 1.0)

    def f_te(b):
        g = np.sqrt(b ** 2 - 1)
        u0, u1, _ = top_admittance_scaled(np.array([b ** 2 + 0j]), slab, g, "TE")
        return complex(g * u0[0] + u1[0]).real

    def closed(b):
        kap, g = np.sqrt(2.25 - b ** 2), np.sqrt(b ** 2 - 1)
        return np.tan(kap * k0d / 2) - g / kap

    b_ref = brentq(closed, 1.40, 1.46)
    b_new = brentq(f_te, 1.40, 1.46)
    assert b_new == pytest.approx(b_ref, abs=1e-10)


def test_pole_residue_matches_full_sommerfeld_far_away():
    """Grounded dielectric slab: far from the source the TM0 surface-wave pole term must
    reproduce the exact Sommerfeld integral (space wave decays faster) [MM97, Chew ch. 2]."""
    from tissue_slab_modes.sommerfeld import _D, ved_pole_field
    st = Stack(F0, "air", 1.0, [Layer("slab", 4 - 0.004j, 6e-3)], "PEC", 1 - 1e12j)
    qp = complex(mp.findroot(lambda q: complex(_D(complex(q), st)), mp.mpc(1.05, -0.0005), solver="muller"))
    h, rho = 2e-3, 2.0
    Kf = ved_direct_field(st, rho) + ved_reflected_field(st, h, rho)
    Kp = ved_pole_field(st, h, rho, qp)
    assert abs(Kp / Kf) == pytest.approx(1.0, abs=0.03)
    assert abs(np.angle(Kp / Kf)) < 0.05


def test_hed_over_pec_and_energy_conservation():
    """HED over PEC vs Balanis; for a lossless ground all input power is radiated into air."""
    from tissue_slab_modes.sommerfeld import hed_input_resistance, air_radiation_fraction
    pec = Stack(F0, "air", 1.0, [], "PEC", 1 - 1e12j)
    for h in (5e-3, 20e-3, 60e-3):
        x = 2 * pec.k0 * h
        exact = 1 - 1.5 * (np.sin(x) / x + np.cos(x) / x ** 2 - np.sin(x) / x ** 3)
        assert hed_input_resistance(pec, h) == pytest.approx(exact, rel=2e-4)
        assert air_radiation_fraction(pec, h, "tangential") == pytest.approx(exact, rel=2e-4)
        assert air_radiation_fraction(pec, h, "normal") == pytest.approx(ved_input_resistance(pec, h), rel=2e-4)


def test_cole_cole_reproduces_itis_values():
    """Gabriel 4-pole Cole-Cole [G96] at 2.45 GHz vs the IT'IS / Gabriel table."""
    from tissue_slab_modes.materials import REFERENCE_2G45, eps, sigma
    for t, (er, sg) in REFERENCE_2G45.items():
        assert eps(t, 2.45e9).real == pytest.approx(er, abs=0.01)
        assert sigma(t, 2.45e9) == pytest.approx(sg, abs=0.001)


# ----------------------------------------------------------------------------- loss guiding
def test_te_loss_guiding_equals_gain_guiding_exactly():
    """TE: shifting every permittivity by the same j c (loss in cladding -> gain in core)
    shifts n^2 by j c and nothing else [Siegman JOSA A 20, 1617 (2003): only relative gain matters]."""
    from tissue_slab_modes.loss_guiding import sym_modes
    from tissue_slab_modes.materials import eps
    f = 2.44e9
    ef, em = eps("fat", f), eps("muscle", f)
    c = -em.imag
    a = sym_modes(ef, em, 25e-3, f, "TE")
    b = sym_modes(ef + 1j * c, em + 1j * c, 25e-3, f, "TE")
    nb = [m.n_eff ** 2 - 1j * c for m in b]
    assert len(a) == len(b)
    assert max(min(abs(m.n_eff ** 2 - y) for y in nb) for m in a) < 1e-9
    assert sorted(m.bound for m in a) == sorted(m.bound for m in b)


def test_te_threshold_matches_analytic_curve():
    """Numerical TE0 threshold (continuation in cladding loss) lies on the analytic
    kappa tan kappa = j eta curve."""
    from tissue_slab_modes.loss_guiding import antiguide_thresholds, te_threshold_curve
    from tissue_slab_modes.materials import eps
    f = 2.44e9
    ef, em = eps("fat", f), eps("muscle", f)
    r = antiguide_thresholds(ef.real, em.real, -em.imag, 8e-3, f, "TE")
    dn, g, _ = te_threshold_curve(eta_max=10, n=600)
    g_analytic = np.interp(r["DeltaN"], dn[::-1], g[::-1])
    assert r["G_loss"] == pytest.approx(g_analytic, rel=2e-3)
    assert r["G_gain"] == pytest.approx(r["G_loss"], rel=1e-6)


def test_multilayer_symmetric_core_reduces_to_single_layer():
    from tissue_slab_modes.loss_guiding import sym_modes, symml_modes
    from tissue_slab_modes.materials import eps
    f = 2.44e9
    ef, em = eps("fat", f), eps("muscle", f)
    a = [m.n_eff for m in sym_modes(ef, em, 8e-3, f)]
    b = [m.n_eff for m in symml_modes([(ef, 3e-3), (ef, 5e-3)], em, f)]
    assert len(a) == len(b)
    assert max(min(abs(x - y) for y in b) for x in a) < 1e-9
