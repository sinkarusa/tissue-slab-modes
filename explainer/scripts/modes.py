"""Every mode the README discusses: its numbers, its profile and a launch map.

    python scripts/modes.py                 # everything
    python scripts/modes.py --no-scan       # skip the truly-bound frequency scan (the slow part)
    python scripts/modes.py --no-figures    # numbers only

Writes
    results/modes.md, results/modes.json    every value, next to the value the README or CATALOG quotes
    figures/modes/<id>.png                  transverse profile of each mode
    figures/modes/<id>_launch.png           the mode alone, launched by its own profile
    figures/modes/launch_maps.png, .gif     the five launch maps together (snapshot; one period)

Stacks: make_figures.site() and temple_with_bone(), plus four built here: the upper arm with 8 mm
of fat, the assumed limb (air | skin 1.5 | fat 10 | muscle 25 | cortical bone 6 mm | yellow marrow),
8 mm of fat between muscle half-spaces, and 25 mm of muscle between lossless fat (the lossy-core
thought experiment). Every mode comes from the argument-principle solver in tissue-slab-modes
(https://github.com/sinkarusa/tissue-slab-modes); tissue permittivities are the Gabriel 4-pole
Cole-Cole model [G96]; gold is the Drude fit used for figure 05. Fields ~ exp(j(wt - beta z)); x is
normal to the layers (0 at the top interface, negative into the body); n = beta/k0 is complex and
Im eps < 0 is loss.

A launch map is idealized on purpose: it shows one mode by itself, as if a perfect, mode-matched
launcher had excited nothing else. A real antenna, a dipole say, also excites other modes and
branch-cut waves.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import (C0, F_BT, _follow, display_name, eps_gold, single_interface, site,  # noqa: E402
                          temple_with_bone, temporalis_mode, tm0_sym)
from tissue_slab_modes import Layer, Stack, find_modes  # noqa: E402
from tissue_slab_modes import plotting as P  # noqa: E402
from tissue_slab_modes.fields import mode_profile, power_fractions  # noqa: E402
from tissue_slab_modes.loss_guiding import antiguide_thresholds  # noqa: E402
from tissue_slab_modes.loss_guiding import sym_modes  # noqa: E402
from tissue_slab_modes.materials import BODY_SITES, eps  # noqa: E402
from tissue_slab_modes.modes import pick  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGM = ROOT / "figures" / "modes"
F_UWB = 7.99e9
DB = 20 / np.log(10)                     # nepers -> dB
DPI = 170
ZR_WIDE = (-3.5, 3.5)                    # APM window (z = ln t) wide enough for n up to ~8
SCAN_GHZ = (0.4, 0.9, 1.5, 2.44, 3.5, 4.5, 5.5, 6.0, 6.5)
ONSET_GHZ = (6.75, 7.0)                  # head stacks only: where the CSF mode appears

NAMES = {"bone cortical": "cortical bone", "bone cancellous": "cancellous bone", "bone marrow yellow": "yellow marrow",
         "grey matter": "gray matter", "csf": "CSF", "temporalis muscle": "temporalis"}


def nice(name):
    return NAMES.get(name, display_name(name))


def group(name):
    """Region name for power shares: cortical and cancellous bone count as one 'bone'."""
    key = name.lower()
    if "marrow" in key:
        return "yellow marrow"
    return "bone" if "bone" in key else nice(name)


def fill(name):
    key = name.lower()
    if "marrow" in key:
        return "#f1dfb9"
    if "gold" in key:
        return "#e9cf74"
    return P.fill_for(name)


# =========================================================================== modes
@dataclass
class Found:
    """One root of a dispersion function, from find_modes (Mode) or sym_modes (SymRoot)."""
    pol: str
    n_eff: complex
    gamma_top: complex
    gamma_bot: complex
    loss: float                                    # dB/cm

    @classmethod
    def of(cls, m):
        return cls(m.pol, complex(m.n_eff), complex(m.gamma_top), complex(m.gamma_bot), float(m.loss_dB_per_cm))

    @classmethod
    def sym(cls, r, pol="TM"):
        g = complex(r.gamma_clad)                  # symmetric slab: the same cladding on both sides
        return cls(pol, complex(r.n_eff), g, g, float(r.loss_dB_per_cm))

    @property
    def sheet(self):
        return f"{'P' if self.gamma_top.real > 0 else 'I'}/{'P' if self.gamma_bot.real > 0 else 'I'}"

    @property
    def b2a(self):
        """beta / 2 alpha: above 1 the mode oscillates faster than it decays (wave-like)."""
        return self.n_eff.real / (2 * abs(self.n_eff.imag))


@dataclass
class Spec:
    id: str
    title: str                                     # as printed on the figure (quoting rule)
    where: str
    find: Callable                                 # () -> (Stack, Found, notes)
    above: float                                   # plotted height above the top interface (unit)
    below: float                                   # plotted depth below the bottom interface (unit)
    expected: list = field(default_factory=list)   # [(quantity, value as quoted, source), ...]
    logy: bool = False
    unit: str = "mm"
    launch: bool = False


_ROOTS = {}


def roots(key, st, pol, zr):
    if key not in _ROOTS:
        _ROOTS[key] = find_modes(st, pol, zr=zr)
    return _ROOTS[key]


def stack(f, layers, bot, label):
    """air | layers [(tissue, mm), ...] from the top down | bot (tissue half-space)."""
    return Stack(f, "air", 1.0, [Layer(t, eps(t, f), d * 1e-3) for t, d in layers], bot, eps(bot, f), label=label)


def limb(f):
    """ASSUMED geometry: 25 mm of muscle over planar cortical bone and yellow marrow."""
    return stack(f, [("skin", 1.5), ("fat", 10.0), ("muscle", 25.0), ("bone cortical", 6.0)], "bone marrow yellow",
                 "limb (assumed)")


def sides(st, fm):
    """Per half-space: proper or not, and the evanescent-or-wave test Re(n^2 - eps) > 0. For a wave,
    the transverse index sqrt(eps - n^2) = -j gamma, its angle from the normal, and whether it travels
    away from the guide. Decay is along the depth, in dB/cm (negative: it grows); depth_lam0 is the 1/e
    depth of the field in free-space wavelengths (as in figure 05), None where the field grows."""
    out = {}
    for which, name, e, g in (("top", st.top_name, st.eps_top, fm.gamma_top),
                              ("bottom", st.bot_name, st.eps_bot, fm.gamma_bot)):
        e, g = complex(e), complex(g)
        q = complex(fm.n_eff ** 2 - e)
        d = dict(medium=nice(name), eps=e, gamma=g, proper=bool(g.real > 0), re_n2_minus_eps=q.real,
                 evanescent=bool(q.real > 0), decay_dB_per_cm=float(DB * g.real * st.k0 / 100),
                 depth_lam0=float(1 / (2 * np.pi * g.real)) if g.real > 0 else None)
        if not d["evanescent"]:
            d.update(kappa=-1j * g, angle_deg=float(np.degrees(np.arctan2(fm.n_eff.real, abs(g.imag)))),
                     direction="away from the guide" if g.imag > 0 else "toward the guide")
        out[which] = d
    return out


def kind(sd, b2a):
    if not (sd["top"]["proper"] and sd["bottom"]["proper"]):
        grows = [s["medium"] for s in sd.values() if not s["proper"]]
        return f"leaky: grows into the {' and '.join(grows)}"
    if sd["top"]["evanescent"] and sd["bottom"]["evanescent"]:
        return "truly bound" if b2a > 1 else "truly bound, but overdamped"
    return '"bound": proper only because of loss' + (", overdamped (β/2α < 1)" if b2a < 1 else "")


def profile(st, fm, above, below, npts=4000):
    """mode_profile, plus the single-interface case (no layers), which mode_profile cannot normalize."""
    if st.layers:
        return mode_profile(st, fm.n_eff, fm.gamma_top, fm.gamma_bot, fm.pol, above=above, below=below, npts=npts)
    k0 = st.k0
    x = np.linspace(-below, above, npts)
    H = np.empty_like(x, dtype=complex)
    up = x > 0
    H[up] = np.exp(-fm.gamma_top * k0 * x[up])
    H[~up] = np.exp(fm.gamma_bot * k0 * x[~up])
    return dict(x=x, edges=np.array([0.0]), eps=np.where(x > 0, st.eps_top, st.eps_bot), pol=fm.pol, n_eff=fm.n_eff,
                gamma_top=fm.gamma_top, gamma_bot=fm.gamma_bot, H_top=1.0, H_bot=1.0, k0=k0, Hy=H)


def shares(st, fm):
    """Fraction of the guided power in each region (None when a side is improper: the tail is infinite)."""
    if fm.sheet != "P/P":
        return None
    D = st.total_thickness
    pr = profile(st, fm, max(0.5 * D, 0.02 * st.lam0), max(0.5 * D, 0.02 * st.lam0), npts=1200)
    out = {}
    for k, v in power_fractions(st, pr).items():
        out[group(k)] = out.get(group(k), 0.0) + float(v)
    return out


def truly_bound(st, fm):
    return fm.loss > 0 and kind(sides(st, fm), fm.b2a) == "truly bound"


def best_share(st, ms, region):
    """The truly bound root with the largest power share in `region`, and how many truly bound roots exist."""
    cands = [Found.of(m) for m in ms]
    cands = [c for c in cands if truly_bound(st, c)]
    return max(cands, key=lambda c: shares(st, c).get(region, 0.0)), len(cands)


# --------------------------------------------------------------------------- finders: () -> (Stack, Found, notes)
def f_zenneck():
    st = site("temple", F_BT)
    return st, Found.of(pick(st, roots("temple", st, "TM", (-2.2, 2.2)), "surface")), {
        "selection": "lowest-loss P/P root with more than half its power in air (pick 'surface'), as for figure 01"}


def f_fat(name, key):
    def find():
        st = site(name, F_BT)
        return st, Found.of(pick(st, roots(key, st, "TM", (-3, 3)), "tissue", min_re_n=1.0)), {
            "selection": "lowest-loss P/P root with Re n >= 1 and under half its power in air (pick 'tissue')"}
    return find


def f_twin():
    st = site("upper arm", F_BT)
    ms = roots("arm", st, "TM", (-3, 3))
    fat = pick(st, ms, "tissue", min_re_n=1.0)
    twin = min([m for m in ms if m.sheet == "I/P" and m.loss_dB_per_cm > 0], key=lambda m: abs(m.n_eff - fat.n_eff))
    return st, Found.of(twin), {"selection": "the I/P root nearest the fat mode, as in figure 02"}


def f_fat_8mm():
    st = stack(F_BT, [("skin", 1.5), ("fat", 8.0)], "muscle", "upper arm, 8 mm fat")
    return st, Found.of(pick(st, find_modes(st, "TM", zr=(-3, 3)), "tissue", min_re_n=1.0)), {
        "selection": "as for the 10 mm upper arm"}


def f_fat_sym():
    ef, em = eps("fat", F_BT), eps("muscle", F_BT)
    st = Stack(F_BT, "muscle", em, [Layer("fat", ef, 8e-3)], "muscle", em, label="muscle | fat 8 mm | muscle")
    return st, Found.sym(tm0_sym(8e-3, em, ef)), {
        "selection": "TM0 of the symmetric slab: lowest-loss bound root with Re n > 1 (make_figures.tm0_sym), "
                     "the model of figure 04"}


def f_skull():
    st = site("occipital", F_BT)
    return st, Found.of(pick(st, roots("occ", st, "TM", (-2.2, 2.2)), "tissue", min_re_n=1.0)), {
        "selection": "pick 'tissue', as for figure 01"}


def f_temporalis(d_mm):
    def find():
        st, m = temporalis_mode(d_mm)
        return st, Found.of(m), {"selection": f"lowest-loss P/P root at 6 mm of bone with 2 < Re n < 7, continued "
                                              f"to {d_mm} mm (make_figures.temporalis_mode), as for figure 07"}
    return find


def f_limb(f):
    def find():
        st = limb(f)
        fm, n = best_share(st, roots(f"limb{f}", st, "TE", ZR_WIDE), "muscle")
        return st, fm, {"selection": f"truly bound TE root with the largest power share in muscle "
                                     f"({n} truly bound TE roots at this frequency)"}
    return find


def f_csf():
    st = site("occipital", F_UWB)
    fm, n = best_share(st, roots("occ_uwb_TE", st, "TE", ZR_WIDE), "CSF")
    return st, fm, {"selection": f"truly bound TE root with the largest power share in CSF ({n} truly bound TE roots)"}


def f_lossy_core():
    ef, em = eps("fat", F_UWB).real, eps("muscle", F_UWB)
    st = Stack(F_UWB, "fat (lossless)", ef, [Layer("muscle", em, 25e-3)], "fat (lossless)", ef,
               label="lossless fat | muscle 25 mm | lossless fat")
    rs = sorted([r for r in sym_modes(em, ef, 25e-3, F_UWB, "TM") if r.bound and 0 < r.loss_dB_per_cm < 3
                 and 1.5 < r.n_eff.real < 3], key=lambda r: r.loss_dB_per_cm)
    single = complex(np.sqrt(ef * em / (ef + em)))
    single = single if single.imag <= 0 else -single
    return st, Found.sym(rs[0]), {
        "selection": "lowest-loss bound TM root with 1.5 < Re n < 3: one of the two coupled Zenneck-type waves on the "
                     "fat | muscle faces",
        "partner": [complex(r.n_eff) for r in rs[1:]],
        "single interface (lossless fat | muscle)": single}


def lossy_core_fat_loss():
    """The lossy-core pair (25 mm muscle between fat, 7.99 GHz, TM) followed from lossless fat to real fat:
    it is proper only while the fat is (nearly) lossless."""
    e_f, em = complex(eps("fat", F_UWB)), complex(eps("muscle", F_UWB))
    k0 = 2 * np.pi * F_UWB / C0
    rows, prev = [], None
    for s in (0.0, 0.25, 0.5, 1.0):
        ef = complex(e_f.real, s * e_f.imag)
        rs = [r for r in sym_modes(em, ef, 25e-3, F_UWB, "TM") if 1.5 < r.n_eff.real < 3 and r.loss_dB_per_cm > 0]
        if prev is None:
            r = min((r for r in rs if r.bound and r.loss_dB_per_cm < 3), key=lambda r: r.loss_dB_per_cm)
        else:
            r = min(rs, key=lambda r: abs(complex(r.n_eff) - prev))
        prev, g = complex(r.n_eff), complex(r.gamma_clad)
        zen = complex(np.sqrt(ef * em / (ef + em)))
        zen = zen if zen.imag <= 0 else -zen
        rows.append(dict(scale=s, n=prev, loss=float(r.loss_dB_per_cm), proper=bool(g.real > 0), re_gamma_fat=g.real,
                         zenneck_loss=float(-DB * zen.imag * k0 / 100)))
    return dict(rows=rows, fat_eps=e_f, fat_bulk_loss=float(-DB * np.sqrt(e_f).imag * k0 / 100))


def _surface_wave(f, name, e, label, note):
    """TM surface wave on an air | e interface (single_interface: n = sqrt(e/(1+e)), proper air side)."""
    n, ga = (complex(v[0]) for v in single_interface(np.array([e])))
    st = Stack(f, "air", 1.0, [], name, e, label=label)
    # TM continuity at the interface: gamma_air / 1 + gamma_e / e = 0
    return st, Found("TM", n, ga, -e * ga, float(-DB * n.imag * st.k0 / 100)), {"selection": note}


def f_zenneck_skin():
    return _surface_wave(F_BT, "skin", complex(eps("skin", F_BT)), "air | skin, 2.44 GHz",
                         "n = sqrt(eps/(1+eps)) on a skin half-space, as in figure 05")


def f_gold():
    f = C0 / 10e-6
    return _surface_wave(f, "gold", complex(eps_gold(f)), "air | gold, 10 µm",
                         "n = sqrt(eps/(1+eps)) at a free-space wavelength of 10 µm, Drude gold as in figure 05")


README = "README"
CAT = "CATALOG"
SPECS = [
    Spec("zenneck_temple", "Zenneck surface wave on skin", "temple, 2.44 GHz", f_zenneck, 420, 25,
         [("n", "0.973 − 0.023j", README), ("loss", "0.10", README), ("sheet", "P/P", README),
          ("share:air", "99", README), ("depth:top", "1.8", "decision Q40")], logy=True, launch=True),
    Spec("zenneck_skin", "Zenneck surface wave on skin", "skin half-space (no layers), 2.44 GHz", f_zenneck_skin,
         1500, 5, [("depth:top", "7.6", "decision Q40")], logy=True),
    Spec("fat_arm", "Fat mode", "upper arm, 10 mm fat, 2.44 GHz", f_fat("upper arm", "arm"), 6, 14,
         [("n", "1.832 − 0.667j", README), ("loss", "2.96", README), ("sheet", "P/P", README),
          ("share:fat", "82", README)], launch=True),
    Spec("fat_arm_twin", "The fat mode's leaky twin", "upper arm, 10 mm fat, 2.44 GHz", f_twin, 6, 14,
         [("n", "1.797 − 0.866j", README), ("loss", "3.85", README), ("sheet", "I/P", README)]),
    Spec("fat_arm_8mm", "Fat mode, 8 mm of fat", "upper arm, 8 mm fat, 2.44 GHz", f_fat_8mm, 6, 14,
         [("n", "1.70 − 0.82j", README), ("kappa:bottom", "7.14 − 0.70j", README), ("angle:bottom", "13", README),
          ("decay:bottom", "3.1", README)]),
    Spec("fat_abdomen", "Fat mode, abdomen", "abdomen, 20 mm fat, 2.44 GHz", f_fat("abdomen", "abd"), 6, 14),
    Spec("fat_wrist", "Fat mode, volar wrist", "volar wrist, 3 mm fat, 2.44 GHz", f_fat("wrist", "wrist"), 6, 14),
    Spec("fat_sym_8mm", "Fat mode, symmetric model", "8 mm fat between muscle half-spaces, 2.44 GHz", f_fat_sym, 12,
         12, [("loss", "3.77", README)]),
    Spec("skull_occipital", "Skull mode", "back of the head, 8.4 mm bone, 2.44 GHz", f_skull, 6, 10,
         [("n", "2.949 − 1.246j", README), ("loss", "5.53", README), ("sheet", "P/P", README),
          ("share:bone", "46", README), ("b2a", "1.2", README), ("kappa:bottom", "6.48 − 0.46j", README),
          ("angle:bottom", "24", README), ("decay:bottom", "2.0", README)], launch=True),
    Spec("temporalis_6mm", "Temporalis mode, 6 mm of bone", "temple, 6 mm squama, 2.44 GHz", f_temporalis(6.0), 6, 14,
         [("n", "4.49 − 0.97j", README), ("loss", "4.3", README), ("sheet", "P/P", README)]),
    Spec("temporalis_1p5mm", "Temporalis mode, 1.5 mm of bone", "temple, 1.5 mm squama, 2.44 GHz",
         f_temporalis(1.5), 6, 14,
         [("n", "5.151 − 1.331j", README), ("loss", "5.91", README), ("sheet", "P/I", README)], launch=True),
    Spec("limb_muscle_2p44", "Limb-muscle mode", "limb (assumed), 2.44 GHz", f_limb(2.44e9), 5, 5,
         [("n", "7.04 − 0.89j", README), ("loss", "4.0", README), ("share:muscle", "98", README)], launch=True),
    Spec("limb_muscle_0p9", "Limb-muscle mode", "limb (assumed), 0.9 GHz", f_limb(0.9e9), 5, 10,
         [("n", "6.35 − 1.29j", README), ("loss", "2.1", README), ("share:muscle", "86", README)]),
    Spec("limb_muscle_0p4", "Limb-muscle mode", "limb (assumed), 0.4 GHz", f_limb(0.4e9), 5, 20,
         [("n", "5.02 − 2.10j", README), ("loss", "1.5", README), ("share:muscle", "56", README),
          ("b2a", "1.2", README)]),
    Spec("csf_uwb", "CSF mode", "back of the head, 2 mm CSF, 7.99 GHz", f_csf, 3, 5,
         [("n", "6.66 − 1.58j", README), ("loss", "22.9", README), ("share:CSF", "47", README)]),
    Spec("lossy_core_uwb", "Lossy-core mode", "25 mm muscle between lossless fat, 7.99 GHz", f_lossy_core, 12, 12,
         [("n", "2.089 − 0.032j", README), ("loss", "0.47", README)]),
    Spec("gold_spp", "Surface plasmon on gold", "air | Drude gold, 10 µm free-space wavelength", f_gold, 400, 0.1,
         logy=True, unit="µm"),
]


# =========================================================================== quoted vs computed
def _num(s):
    return s.replace("−", "-").replace(" ", "")


def _dec(s):
    return max((len(p.split(".")[1]) if "." in p else 0) for p in re.findall(r"\d+\.?\d*", s))


def fmt_c(z, d=3):
    return f"{z.real:.{d}f} {'−' if z.imag < 0 else '+'} {abs(z.imag):.{d}f}j"


def fmt_n(z):
    """Effective index: three decimals, or six when it sits within 0.001 of 1 (the gold plasmon at 10 µm)."""
    return fmt_c(z, 6 if abs(z.real - 1) < 1e-3 else 3)


def fmt_note(v):
    if isinstance(v, complex):
        return fmt_c(v)
    if isinstance(v, list):
        return ", ".join(fmt_c(c) for c in v) or "none"
    return str(v)


def compare(rec, qty, quoted):
    """Round the computed value the way the quote is written; return (computed as text, matches?)."""
    sd = rec["sides"]
    if qty == "sheet":
        return rec["sheet"], rec["sheet"] == quoted
    d = _dec(quoted)
    if qty in ("n", "kappa:bottom"):
        v = rec["n_eff"] if qty == "n" else sd["bottom"].get("kappa", complex("nan"))
        q = complex(_num(quoted))
        ok = math.isclose(round(v.real, d), q.real, abs_tol=1e-9) and math.isclose(round(v.imag, d), q.imag, abs_tol=1e-9)
        return fmt_c(v, d), ok
    if qty.startswith("share:"):
        v = 100 * (rec["shares"] or {}).get(qty.split(":")[1], float("nan"))
    else:
        v = {"loss": rec["loss_dB_per_cm"], "b2a": rec["beta_over_2alpha"],
             "angle:bottom": sd["bottom"].get("angle_deg", float("nan")),
             "decay:bottom": sd["bottom"]["decay_dB_per_cm"],
             "depth:top": sd["top"]["depth_lam0"] or float("nan")}[qty]
    return f"{v:.{d}f}", math.isclose(round(v, d), float(_num(quoted)), abs_tol=1e-9)


# =========================================================================== figures
def _scale(spec):
    return 1e-6 if spec.unit == "µm" else 1e-3


def symlog(ax, lo, hi, linthresh):
    """Symmetric-log height axis (linear through the layers, logarithmic far out in the air), plain tick labels."""
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.4)
    cands = (0, 1, 10, 100, 400) if linthresh == 1 else (-20, -10, 0, 10, 100, 400, 1000)
    ticks = [t for t in cands if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])


def plot_profile(spec, st, fm, rec):
    u = _scale(spec)
    pr = profile(st, fm, spec.above * u, spec.below * u)
    comp = "Hy" if fm.pol == "TM" else "Ey"
    x, edges = pr["x"] / u, pr["edges"] / u
    bot = edges[-1] - spec.below
    fig, ax = plt.subplots(figsize=(5.2, 6.6))
    spans = [(0, spec.above, st.top_name)] + [(edges[i + 1], edges[i], L.name) for i, L in enumerate(st.layers)]
    spans += [(bot, edges[-1], st.bot_name)]
    for lo, hi, name in spans:
        ax.axhspan(lo, hi, color=fill(name), lw=0)
        if (hi - lo > 1.2 or spec.logy) and name != "air":
            ax.text(0.98, (lo + hi) / 2, nice(name), ha="right", va="center", fontsize=8, color=P.INK2,
                    transform=ax.get_yaxis_transform())
    for e in edges:
        ax.axhline(e, color=P.MUTED, lw=0.5)
    h = np.abs(pr[comp])
    sel = (x >= bot) & (x <= spec.above)
    ax.plot(h[sel] / h[sel].max(), x[sel], color=P.SERIES[0], lw=2.2)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(bot, spec.above)
    if spec.logy:
        symlog(ax, bot, spec.above, 15 if spec.unit == "mm" else 1)
        ax.text(0.5, spec.above / 2, "air", ha="center", fontsize=8, color=P.INK2)
    comp_tex = r"|H$_y$|" if fm.pol == "TM" else r"|E$_y$|"
    ax.set_xlabel(f"{comp_tex} (normalized)")
    ylab = "height above the skin" if st.top_name == "air" else "position across the stack"
    ax.set_ylabel(f"{ylab} ({spec.unit})")
    ax.set_title(f"{spec.title}\n{spec.where}, {fm.pol}", loc="left")
    sh = rec["shares"]
    where = ("; ".join(f"{100 * v:.0f} % in {k}" for k, v in sorted(sh.items(), key=lambda kv: -kv[1])[:2])
             if sh else "power share undefined: the field grows away from the guide")
    sides_txt = "; ".join(f"{w}: {'evanescent' if s['evanescent'] else 'a wave'} in the {s['medium']}"
                          for w, s in rec["sides"].items())
    ax.text(0.0, -0.11, f"n = {fmt_n(fm.n_eff)}   {fm.loss:.2f} dB/cm   sheet {fm.sheet}   β/2α = {fm.b2a:.2f}\n"
            f"{rec['kind']}\n{where}\n{sides_txt}", transform=ax.transAxes, fontsize=7.8, color=P.INK, va="top")
    fig.savefig(FIGM / f"{spec.id}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


LAUNCH_FLOOR_DB = 40.0             # launch maps: color floor, dB below the field's peak at the launch plane
LAUNCH_TICKS = (1.0, 0.5, 0.0, -0.5, -1.0)
LAUNCH_TICK_LABELS = ("0 dB", "−20 dB", "below −40 dB", "−20 dB", "0 dB")


def signed_log(v):
    """Signed log scale for the launch maps: sign(v) (1 + dB/LAUNCH_FLOOR_DB), clipped at the floor, so the
    peak at the launch plane maps to +-1 and anything 40 dB below it to 0 (white)."""
    db = 20 * np.log10(np.maximum(np.abs(v), 1e-30))
    return np.sign(v) * np.clip(1 + db / LAUNCH_FLOOR_DB, 0.0, 1.0)


def launch_run(spec, st, fm):
    """Distance shown along the guide (m): 10 cm for the tissue modes; four guide wavelengths for the air-dominated
    surface waves (spec.logy), whose guide wavelength is ~12 cm (Zenneck) or 10 um (gold), so a fixed 10 cm would
    show less than one period of the one and alias the other."""
    return 4 * 2 * np.pi / (st.k0 * fm.n_eff.real) if spec.logy else 0.10


def dist_unit(spec):
    return ("µm", 1e6) if spec.unit == "µm" else ("cm", 100)


def launch_field(spec, st, fm, zmax=0.10, nx=420, nz=600):
    """The mode by itself, psi(x) exp(-j k0 n z): what a perfect, mode-matched launcher at z = 0 would excite,
    normalized to its peak at z = 0. Also returns the permittivity on the same height grid; z is in dist_unit."""
    u = _scale(spec)
    pr = profile(st, fm, spec.above * u, spec.below * u, npts=8000 if spec.logy else nx)
    x, psi, eps = pr["x"] / u, pr["Hy"] if fm.pol == "TM" else pr["Ey"], pr["eps"]
    if spec.logy:                     # resample: dense in the layers, sparse far out in the air
        lo = pr["edges"][-1] / u - spec.below
        xs = np.concatenate([np.linspace(lo, 15, 300), np.geomspace(15.5, spec.above, 120)])
        cint = lambda f: np.interp(xs, x, f.real) + 1j * np.interp(xs, x, f.imag)
        psi, eps, x = cint(psi), cint(eps), xs
    psi = psi / np.abs(psi).max()
    z = np.linspace(0, zmax, nz)
    return x, z * dist_unit(spec)[1], psi[:, None] * np.exp(-1j * st.k0 * fm.n_eff * z)[None, :], pr["edges"] / u, eps


def launch_title(spec, st, fm):
    return (f"{spec.title}: {spec.where}, {fm.pol}   (n = {fmt_c(fm.n_eff, 2)}, {fm.loss:.1f} dB/cm, "
            f"{rec_kind_short(st, fm)})" + ("; note the four-wavelength run" if spec.logy else ""))


def launch_xlabel(spec):
    return f"distance from the launch plane ({dist_unit(spec)[0]})"


def _layer_lines(ax, spec, edges):
    for e in ([0.0] if spec.logy else edges):
        ax.axhline(e, color=P.INK2, lw=0.6)


def draw_launch(ax, spec, st, fm, phase=0.0, title=True):
    """Air-dominated surface waves (spec.logy) get a linear height axis up to spec.above and a four-wavelength run
    (launch_run): they live in the air, so a millimeter-scale panel would blow the thin tissue layers up to fill the
    frame."""
    x, z, F, edges, _ = launch_field(spec, st, fm, zmax=launch_run(spec, st, fm))
    mesh = ax.pcolormesh(z, x, signed_log(np.real(F * np.exp(1j * phase))), cmap="RdBu_r", vmin=-1, vmax=1,
                         shading="auto", rasterized=True)
    bottom = edges[-1] - spec.below
    _layer_lines(ax, spec, edges)
    if spec.logy:
        labels = [(st.top_name, spec.above / 2), ("tissue" if st.layers else st.bot_name, bottom / 2)]
    else:
        names = [st.top_name] + [L.name for L in st.layers] + [st.bot_name]
        tops = [spec.above] + list(edges)
        bots = list(edges) + [bottom]
        labels = [(name, (lo + hi) / 2) for name, hi, lo in zip(names, tops, bots) if hi - lo > 1.5 or name == "air"]
    for name, y in labels:
        ax.text(1.005, y, nice(name), transform=ax.get_yaxis_transform(), fontsize=7.5, va="center", color=P.INK2)
    ax.set_ylim(bottom, spec.above)
    ax.set_ylabel(f"height ({spec.unit})")
    if title:
        ax.set_title(launch_title(spec, st, fm), loc="left", fontsize=9)
    return mesh, F


def draw_eps_profile(ae, ap, spec, st, fm, legend=False):
    """The two columns beside a launch map: the permittivity (eps' and eps'' = -Im eps, for exp(+jwt)) and the
    mode's transverse profile |psi| (H_y for TM, E_y for TE), both against the map's height axis."""
    x, _, F, edges, eps = launch_field(spec, st, fm, zmax=0.0, nz=1)
    psi = F[:, 0]
    ae.plot(eps.real, x, color=P.SERIES[0], lw=1.2, label="ε′")
    ae.plot(-eps.imag, x, color=P.SERIES[3], lw=1.2, ls="--", label="ε″")
    ae.set_xlim(0, 75)
    ap.plot(np.abs(psi), x, color=P.INK, lw=1.2)
    ap.set_xlim(0, 1.08)
    ap.text(0.96, 0.95, r"|H$_y$|" if fm.pol == "TM" else r"|E$_y$|", transform=ap.transAxes, ha="right", va="top",
            fontsize=8)
    for ax in (ae, ap):
        _layer_lines(ax, spec, edges)
        ax.tick_params(labelsize=7)
    plt.setp(ap.get_yticklabels(), visible=False)
    if legend:
        ae.legend(fontsize=7, loc="upper right", frameon=False)


def launch_colorbar(fig, mesh, axes=None, cax=None, **kw):
    """Horizontal bar under the panels, so it never collides with the layer names at the right edge. The color
    depth is the level in dB; the hue is the sign (the two half-cycles), so the dB ticks repeat on both sides."""
    where = dict(cax=cax) if cax is not None else dict(ax=axes, location="bottom")
    cb = fig.colorbar(mesh, ticks=LAUNCH_TICKS, orientation="horizontal", **where, **kw)
    cb.ax.set_xticklabels(LAUNCH_TICK_LABELS, fontsize=7.5)
    cb.ax.text(0.0, 1.25, "← blue: field pointing one way", transform=cb.ax.transAxes, ha="left", fontsize=7.5)
    cb.ax.text(1.0, 1.25, "red: pointing the other way →", transform=cb.ax.transAxes, ha="right", fontsize=7.5)
    cb.set_label("field level relative to its peak at the launch plane (log scale)", fontsize=8)
    return cb


def rec_kind_short(st, fm):
    k = kind(sides(st, fm), fm.b2a)
    return k.split(":")[0]


def launch_single(spec, st, fm):
    fig, ax = plt.subplots(figsize=(10.0, 3.0))
    mesh, _ = draw_launch(ax, spec, st, fm)
    launch_colorbar(fig, mesh, ax, pad=0.3, shrink=0.6, aspect=40)
    ax.set_xlabel(launch_xlabel(spec))
    fig.text(0.01, -0.06, "The mode alone, as if launched by a perfect, mode-matched source at z = 0; a real antenna "
             "also excites other waves. Color: the main field component, signed, on a log scale down to 40 dB below "
             "its peak at z = 0.", fontsize=7.5, color=P.INK2)
    fig.savefig(FIGM / f"{spec.id}_launch.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def launch_gif(spec, st, fm, frames=16):
    """One GIF per mode: the same idealized launch as the stills, animated over one period."""
    fig, ax = plt.subplots(figsize=(9.0, 2.8))
    mesh, F = draw_launch(ax, spec, st, fm)
    launch_colorbar(fig, mesh, ax, pad=0.3, shrink=0.6, aspect=40)
    ax.set_xlabel(launch_xlabel(spec))

    def update(k):
        mesh.set_array(signed_log(np.real(F * np.exp(2j * np.pi * k / frames))).ravel())
        return [mesh]

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(FIGM / f"{spec.id}_launch.gif", writer=animation.PillowWriter(fps=10), dpi=80)
    plt.close(fig)


def launch_combined(found, frames=40):
    """README Figure 3: the five modes, one row each: permittivity, mode profile, and the field along the guide; as a still
    and as a GIF (one period in frames / 10 s)."""
    sel = [(s, st, fm) for s, st, fm in found if s.launch]
    order = ["limb_muscle_2p44", "fat_arm", "skull_occipital", "temporalis_1p5mm", "zenneck_temple"]
    sel.sort(key=lambda t: order.index(t[0].id))

    def build(figsize):
        n = len(sel)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n + 1, 3, width_ratios=[1.0, 1.0, 5.0], height_ratios=[1.0] * n + [0.07],
                              hspace=0.85, wspace=0.07, top=0.94, bottom=0.05)
        out = []
        for i, (s, st, fm) in enumerate(sel):
            ae = fig.add_subplot(gs[i, 0])
            ap, am = fig.add_subplot(gs[i, 1], sharey=ae), fig.add_subplot(gs[i, 2], sharey=ae)
            out.append(draw_launch(am, s, st, fm, title=False))
            draw_eps_profile(ae, ap, s, st, fm, legend=i == 0)
            ae.set_ylabel(am.get_ylabel())
            am.set_ylabel("")
            plt.setp(am.get_yticklabels(), visible=False)
            ae.set_title(launch_title(s, st, fm), loc="left", fontsize=9)
            if i == n - 1:
                ae.set_xlabel("permittivity", fontsize=8)
                ap.set_xlabel("mode profile", fontsize=8)
                am.set_xlabel(launch_xlabel(s))
        cbar_gs = gs[n, 2].subgridspec(1, 3, width_ratios=[0.15, 0.7, 0.15])
        launch_colorbar(fig, out[0][0], cax=fig.add_subplot(cbar_gs[0, 1]))
        fig.suptitle("Five modes, each launched by its own profile (idealized: a real antenna excites other waves too)",
                     x=0.02, ha="left", fontweight="bold", fontsize=10)
        return fig, out

    fig, _ = build((12.0, 13.0))
    fig.savefig(FIGM / "launch_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, out = build((11.0, 12.0))

    def update(k):
        ph = 2 * np.pi * k / frames
        for mesh, F in out:
            mesh.set_array(signed_log(np.real(F * np.exp(1j * ph))).ravel())
        return [m for m, _ in out]

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(FIGM / "launch_maps.gif", writer=animation.PillowWriter(fps=10), dpi=72)
    plt.close(fig)


# =========================================================================== checks
def scan_truly_bound():
    """Truly bound roots (P/P, evanescent on both sides, beta/2alpha > 1), TM and TE, in the five stacks."""
    rows = []
    for name, label in (("abdomen", "abdomen"), ("upper arm", "upper arm"), ("wrist", "volar wrist"),
                        ("occipital", "back of the head"), ("temple", "temple")):
        for pol in ("TM", "TE"):
            freqs = SCAN_GHZ + (ONSET_GHZ if name in ("occipital", "temple") else ())
            cells = {}
            for fg in freqs:
                st = site(name, fg * 1e9)
                tb = [Found.of(m) for m in find_modes(st, pol, zr=ZR_WIDE)]
                tb = [fm for fm in tb if truly_bound(st, fm)]
                cells[fg] = [fm.n_eff for fm in tb]
            rows.append(dict(stack=label, pol=pol, cells=cells))
            print(f"  scan {label:15s} {pol}: truly bound at "
                  f"{[fg for fg, v in cells.items() if v] or 'none'}", flush=True)
    return rows


def wavelengths():
    out = []
    for t in ("air", "fat", "bone cortical", "bone cancellous", "skin", "dura", "muscle", "grey matter", "csf"):
        e = 1.0 + 0j if t == "air" else complex(eps(t, F_BT))
        lam = C0 / F_BT / np.sqrt(e).real * 100
        out.append(dict(tissue=nice(t), eps=e, index=float(np.sqrt(e).real), wavelength_cm=float(lam),
                        plane_wave_loss_dB_per_cm=float(-DB * np.sqrt(e).imag * 2 * np.pi * F_BT / C0 / 100)))
    return out


def tube_cutoffs():
    """TE11 cutoff of a circular guide with perfectly conducting walls, filled with the core's thickness-weighted
    mean eps' (2.44 GHz values): f_c = 1.841 c / (2 pi a sqrt(eps')), a = half the core width. High-eps lossy
    walls are only 'nearly conducting', so this is a first-order screen, not a solved mode."""
    out = []
    for name, s in BODY_SITES.items():
        if s.get("geometry") != "tube":
            continue
        row = dict(site=name, source=s["source"], cutoff_GHz={})
        for var, core in s["variants"].items():
            D = sum(d for _, d in core) * 1e-3
            e = sum(eps(t, F_BT).real * d if t != "air" else d for t, d in core) / sum(d for _, d in core)
            row["cutoff_GHz"][var] = float(1.841 * C0 / (2 * np.pi * (D / 2) * np.sqrt(e)) / 1e9)
            row.setdefault("diameter_mm", {})[var] = D * 1e3
        out.append(row)
    return out


def lossy_core_threshold():
    """Smallest core eps'' at which 'muscle (lossless) | fat | muscle (lossless)' carries a proper root.
    antiguide_thresholds walks core GAIN (Siegman); conjugating every eps maps a gain core to a lossy core with
    the same |eps''| and keeps each root's sheet, so the gain threshold is also the lossy-core threshold. On the
    lossy-core side the root that turns proper is the conjugate of the leaky fundamental (a new Nagel-type
    loss-guided root); the leaky fundamental itself stays improper."""
    ef, em = eps("fat", F_BT), eps("muscle", F_BT)
    L = -em.imag
    out = []
    for d_mm in (5.0, 8.0, 10.0):
        for pol in ("TM", "TE"):
            res = antiguide_thresholds(ef.real, em.real, L, d_mm * 1e-3, F_BT, pol=pol, s_max=12.0)
            sg = res["s_gain"]
            out.append(dict(fat_mm=d_mm, pol=pol, eps2_threshold=None if sg is None else float(sg * L)))
    return dict(rows=out, fat_eps2=float(-ef.imag))


def temporalis_threshold(d_hi=2.5, step=0.01):
    """Bone thickness where the temporalis mode turns from leaky (P/I) to P/P: start from the leaky 1.5 mm mode
    (make_figures.temporalis_mode) and follow it upward in 0.01 mm steps. Following it down from 6 mm instead jumps
    to another root near 4-5 mm. Interpolates where Re gamma on the brain side crosses zero."""
    st, m = temporalis_mode(1.5)
    prev = (1.5, m.gamma_bot.real)
    for d in np.arange(1.5 + step, d_hi + 1e-9, step):
        m = _follow(temple_with_bone(F_BT, d), m.top, m.bot, m.n_eff)
        if m is None:
            raise RuntimeError(f"temporalis mode lost at {d:.2f} mm bone")
        if (m.gamma_bot.real > 0) != (prev[1] > 0):
            return float(prev[0] + (d - prev[0]) * prev[1] / (prev[1] - m.gamma_bot.real))
        prev = (d, m.gamma_bot.real)
    return None


def symmetric_checks():
    ef, em = eps("fat", F_BT), eps("muscle", F_BT)
    lossless = tm0_sym(8e-3, em, ef.real)
    gap = {d: tm0_sym(d * 1e-3, em, ef) for d in (25.0, 1.0)}
    return {"lossless fat, real muscle, 8 mm (figure 03, s = 1)": dict(n=lossless.n_eff, loss=lossless.loss_dB_per_cm,
                                                                         bound=bool(lossless.bound)),
            "Re n at 25 mm of fat (figure 04)": gap[25.0].n_eff.real,
            "Re n at 1 mm of fat (figure 04)": gap[1.0].n_eff.real}


# =========================================================================== output
def cjson(o):
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def write_md(recs, checks, runtime):
    L = ["# Mode values behind the README", "",
         "Every mode the README discusses, recomputed by [`scripts/modes.py`](../scripts/modes.py) with the "
         "argument-principle solver in [tissue-slab-modes](https://github.com/sinkarusa/tissue-slab-modes) and the "
         "Gabriel 4-pole Cole-Cole permittivities [G96]. Regenerate with `python scripts/modes.py` "
         f"(this run: {runtime / 60:.1f} min; `--no-scan` skips the slowest part). Fields ~ exp(j(ωt − βz)); "
         "n = β/k₀ is complex; loss is −Im n · k₀ in dB/cm.", "",
         "Kinds: **truly bound** = proper on both sides, evanescent on both sides (Re(n² − ε) > 0) and wave-like "
         "(β/2α > 1). **\"bound\"** = proper on both sides only because of loss: on at least one side the field is a "
         "damped wave, not an evanescent tail. **leaky** = improper on at least one side: the field grows away from "
         "the guide.", "",
         "## Modes", "",
         "| id | mode | where | pol | n | loss (dB/cm) | sheet | kind | β/2α | largest power shares |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        sh = (", ".join(f"{k} {100 * v:.0f} %" for k, v in sorted(r["shares"].items(), key=lambda kv: -kv[1])[:3])
              if r["shares"] else "undefined (improper side)")
        L.append(f"| `{r['id']}` | {r['title']} | {r['where']} | {r['pol']} | {fmt_n(r['n_eff'])} | "
                 f"{r['loss_dB_per_cm']:.2f} | {r['sheet']} | {r['kind']} | {r['beta_over_2alpha']:.2f} | {sh} |")
    L += ["", "## Each half-space: evanescent tail or a wave?", "",
          "For a wave, √(ε − n²) is its transverse index, the angle is measured from the normal to the layers, and "
          "the decay is along the depth (negative: the field grows). The 1/e depth is in free-space wavelengths at "
          f"the mode's own frequency (λ₀ = {C0 / F_BT * 100:.1f} cm at 2.44 GHz, 10 µm for gold).", "",
          "| id | side | medium | Re(n² − ε) | field there | √(ε − n²) | angle | direction | decay (dB/cm) | "
          "1/e depth (λ₀) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        for w, s in r["sides"].items():
            wave = not s["evanescent"]
            kap = fmt_c(s["kappa"], 2) if wave else ""
            ang = f"{s['angle_deg']:.1f}°" if wave else ""
            dep = "" if s["depth_lam0"] is None else f"{s['depth_lam0']:.3g}"
            L.append(f"| `{r['id']}` | {w} | {s['medium']} | {s['re_n2_minus_eps']:.2f} | "
                     f"{'wave' if wave else 'evanescent'} | {kap} | {ang} | {s.get('direction', '')} | "
                     f"{s['decay_dB_per_cm']:.2f} | {dep} |")
    L += ["", "## Against the numbers the README and CATALOG quote", "",
          "The computed value is rounded the way the quote is written.", "",
          "| id | quantity | quoted | source | computed | |", "|---|---|---|---|---|---|"]
    mism = []
    for r in recs:
        for qty, quoted, src, comp, ok in r["checks"]:
            L.append(f"| `{r['id']}` | {qty} | {quoted} | {src} | {comp} | {'✓' if ok else '**differs**'} |")
            if not ok:
                mism.append(f"`{r['id']}` {qty}: quoted {quoted} ({src}), computed {comp}")
    L += ["", "## Selection rules and notes", ""]
    for r in recs:
        notes = "; ".join(f"{k}: {fmt_note(v)}" for k, v in r["notes"].items())
        L.append(f"- `{r['id']}` ({r['stack'] or r['where']}): {notes}")

    L += ["", "## Checks", ""]
    if checks.get("scan"):
        fr = sorted({f for row in checks["scan"] for f in row["cells"]})
        L += ["### Truly bound modes in the five standard stacks", "",
              f"Every root in the APM window Re z ∈ {ZR_WIDE} tested; a cell lists the truly bound roots "
              "(– = none).", "",
              "| stack | pol | " + " | ".join(f"{f:g} GHz" for f in fr) + " |",
              "|---|---|" + "---|" * len(fr)]
        for row in checks["scan"]:
            cells = [(", ".join(fmt_c(n, 2) for n in row["cells"][f]) or "–") if f in row["cells"] else ""
                     for f in fr]
            L.append(f"| {row['stack']} | {row['pol']} | " + " | ".join(cells) + " |")
        first = [(f, row["stack"], row["pol"]) for row in checks["scan"] for f in sorted(row["cells"])
                 if row["cells"][f]]
        L += ["", (f"First truly bound root: {min(first)[0]:g} GHz ({min(first)[1]}, {min(first)[2]}); none at "
                   f"{', '.join(f'{f:g}' for f in fr if f < min(first)[0])} GHz." if first else
                   "No truly bound root at any scanned frequency."), ""]
    else:
        L += ["### Truly bound modes in the five standard stacks", "", "Skipped (`--no-scan`).", ""]
    L += ["### Index, wavelength and plane-wave loss in each tissue, 2.44 GHz", "",
          "| tissue | ε | index Re √ε | wavelength (cm) | plane-wave loss (dB/cm) |", "|---|---|---|---|---|"]
    for w in checks["wavelengths"]:
        L.append(f"| {w['tissue']} | {fmt_c(w['eps'], 1)} | {w['index']:.2f} | {w['wavelength_cm']:.1f} | "
                 f"{w['plane_wave_loss_dB_per_cm']:.2f} |")
    L += ["", "### Tube cutoffs (lowest mode, TE11, of a tube with conducting walls)", "",
          "A tube carries nothing below this frequency. The walls are treated as perfect conductors and the core as "
          "its thickness-weighted mean ε′, so this is a first-order screen, not a solved mode. Geometries: "
          "`BODY_SITES` in tissue-slab-modes `materials.py`.", "",
          "| site | core width thin / typical / thick (mm) | cutoff thin / typical / thick (GHz) | geometry source |",
          "|---|---|---|---|"]
    for t in checks["tubes"]:
        dm = " / ".join(f"{v:.0f}" for v in t["diameter_mm"].values())
        fc = " / ".join(f"{v:.1f}" for v in t["cutoff_GHz"].values())
        L.append(f"| {t['site']} | {dm} | {fc} | {t['source']} |")
    lc = checks["lossy_core"]
    L += ["", "### Lossy-core threshold (Nagel, Blair & Scarpulla's geometry)", "",
          f"Smallest fat ε″ at which lossless muscle | fat | lossless muscle carries a proper mode at 2.44 GHz "
          f"(real fat has ε″ = {lc['fat_eps2']:.2f}). This is a new loss-guided root, not the leaky fundamental "
          "turning proper.", "",
          "| fat (mm) | pol | ε″ threshold |", "|---|---|---|"]
    for row in lc["rows"]:
        th = row["eps2_threshold"]
        L.append(f"| {row['fat_mm']:g} | {row['pol']} | {'none below s_max' if th is None else format(th, '.2f')} |")
    lf = checks["lossy_core_fat_loss"]
    L += ["", "### Lossy-core pair with the fat's own loss restored (7.99 GHz, TM, 25 mm muscle)", "",
          f"Followed from lossless fat (the `lossy_core_uwb` row) to real fat (ε = {fmt_c(lf['fat_eps'], 2)}), whose "
          f"plane-wave loss is {lf['fat_bulk_loss']:.2f} dB/cm. γ_fat is normalized to k0.", "",
          "| fat loss | n | loss (dB/cm) | fat side | Re γ_fat | Zenneck formula, fat \\| muscle (dB/cm) |",
          "|---|---|---|---|---|---|"]
    for row in lf["rows"]:
        L.append(f"| ×{row['scale']:g} | {fmt_n(row['n'])} | {row['loss']:.2f} | "
                 f"{'proper' if row['proper'] else 'improper'} | {row['re_gamma_fat']:+.3f} | {row['zenneck_loss']:.2f} |")
    sym = checks["symmetric"]
    ll = sym["lossless fat, real muscle, 8 mm (figure 03, s = 1)"]
    thr = checks["temporalis_threshold_mm"]
    r25, r1 = sym["Re n at 25 mm of fat (figure 04)"], sym["Re n at 1 mm of fat (figure 04)"]
    tm = [(row["fat_mm"], row["eps2_threshold"]) for row in lc["rows"] if row["pol"] == "TM" and row["eps2_threshold"]]
    L += ["", "### Other quoted values", "",
          f"- Temporalis mode turns leaky below {thr:.2f} mm of bone (decision Q37: \"bound\" down to 1.64 mm).",
          f"- 8 mm of lossless fat in real muscle (figure 03, s = 1): n = {fmt_c(ll['n'])}, {ll['loss']:.2f} dB/cm, "
          f"{'proper' if ll['bound'] else 'improper'} (CATALOG: \"bound\" at 3.0 dB/cm).",
          f"- Figure 04: Re n = {r25:.2f} at 25 mm of fat and {r1:.2f} at 1 mm (README: 2.29 and 4.42).",
          "- Lossy core: CATALOG quotes ε″ ≈ 3.5 for 8 mm of fat and ε″·d ≈ 28 mm; computed (TM) "
          + ", ".join(f"{t:.2f} at {d:g} mm (ε″·d = {t * d:.1f} mm)" for d, t in tm) + ".", ""]
    other = [(f"{thr:.2f}" == "1.64",
              f"temporalis threshold: quoted \"bound\" down to 1.64 mm (decision Q37), computed P/P only above "
              f"{thr:.2f} mm"),
             (any(d == 8 and f"{t:.1f}" == "3.5" for d, t in tm) and all(26 <= t * d <= 30 for d, t in tm),
              "lossy-core threshold: quoted ε″ ≈ 3.5 for 8 mm and ε″·d ≈ 28 mm (CATALOG), computed "
              + ", ".join(f"{t:.2f} at {d:g} mm" for d, t in tm) + " (TM)"),
             (bool(ll["bound"]) and f"{ll['loss']:.1f}" == "3.0",
              f"figure 03 lossless fat: quoted \"bound\" at 3.0 dB/cm (CATALOG), computed {ll['loss']:.2f} dB/cm, "
              f"{'proper' if ll['bound'] else 'improper'}"),
             ((f"{r25:.2f}", f"{r1:.2f}") == ("2.29", "4.42"),
              f"figure 04 Re n: quoted 2.29 and 4.42 (README), computed {r25:.2f} and {r1:.2f}")]
    mism += [msg for ok, msg in other if not ok]
    L += ["## Mismatches", ""] + ([f"- {m}" for m in mism] or ["None."]) + [""]
    L += ["## Figures", "",
          "- `figures/modes/<id>.png`: the transverse profile of each mode above.",
          "- `figures/modes/<id>_launch.png`, `figures/modes/<id>_launch.gif` and `figures/modes/launch_maps.png` / "
          "`.gif`: the mode alone, launched by its own profile at z = 0. This is idealized on purpose: a real antenna, "
          "a dipole say, also excites other modes and branch-cut waves.", ""]
    (RES / "modes.md").write_text("\n".join(L))
    return mism


def main(scan=True, figures=True):
    t0 = time.time()
    FIGM.mkdir(parents=True, exist_ok=True)
    recs, found = [], []
    for spec in SPECS:
        st, fm, notes = spec.find()
        sd = sides(st, fm)
        rec = dict(id=spec.id, title=spec.title, where=spec.where, stack=st.label, f_GHz=st.freq / 1e9, pol=fm.pol,
                   layers=[(nice(L.name), L.thickness * 1e3, complex(L.eps)) for L in st.layers],
                   top=(nice(st.top_name), complex(st.eps_top)), bottom=(nice(st.bot_name), complex(st.eps_bot)),
                   n_eff=fm.n_eff, loss_dB_per_cm=fm.loss, sheet=fm.sheet, beta_over_2alpha=fm.b2a,
                   kind=kind(sd, fm.b2a), sides=sd, shares=shares(st, fm), notes=notes)
        rec["checks"] = [(q, v, src, *compare(rec, q, v)) for q, v, src in spec.expected]
        recs.append(rec)
        found.append((spec, st, fm))
        bad = [c for c in rec["checks"] if not c[4]]
        print(f"{spec.id:18s} n = {fmt_n(fm.n_eff)}  {fm.loss:6.2f} dB/cm  {fm.sheet}  {rec['kind']}"
              + (f"   DIFFERS: {[(c[0], c[1], c[3]) for c in bad]}" if bad else ""), flush=True)
        if figures:
            plot_profile(spec, st, fm, rec)
            if spec.launch:
                launch_single(spec, st, fm)
            launch_gif(spec, st, fm)
    if figures:
        launch_combined(found)
    checks = dict(wavelengths=wavelengths(), tubes=tube_cutoffs(), lossy_core=lossy_core_threshold(),
                  lossy_core_fat_loss=lossy_core_fat_loss(), temporalis_threshold_mm=temporalis_threshold(),
                  symmetric=symmetric_checks(), scan=scan_truly_bound() if scan else None)
    runtime = time.time() - t0
    mism = write_md(recs, checks, runtime)
    (RES / "modes.json").write_text(json.dumps(dict(modes=recs, checks=checks), default=cjson, indent=1))
    print(f"wrote {RES / 'modes.md'} and modes.json; {len(mism)} mismatches; {runtime / 60:.1f} min")


if __name__ == "__main__":
    main(scan="--no-scan" not in sys.argv[1:], figures="--no-figures" not in sys.argv[1:])
