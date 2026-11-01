"""Figures for the tissue-modes-explainer post.

    python scripts/make_figures.py                  # all figures -> figures/*.png
    python scripts/make_figures.py family skull     # selected figures (names in FIGS)
    python scripts/make_figures.py --eps family     # also print eps' - j eps'' on every cross-section layer

Every mode comes from the argument-principle (APM) solver in tissue-slab-modes
(https://github.com/sinkarusa/tissue-slab-modes); tissue permittivities are the Gabriel
4-pole Cole-Cole model [G96]. Gold is a Drude fit (Ordal et al., Appl. Opt. 24, 4493 (1985):
omega_p = 72800 cm^-1, omega_tau = 215 cm^-1). Fields ~ exp(j(wt - beta z)), Im eps < 0 = loss.
"""
from pathlib import Path

import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D

from tissue_slab_modes import APMLog, Layer, Stack, find_modes
from tissue_slab_modes import plotting as P
from tissue_slab_modes.dispersion import TPlane, classify, dispersion_mp, dispersion_scaled
from tissue_slab_modes.fields import mode_profile, power_fractions
from tissue_slab_modes.loss_guiding import F_sym_np, antiguide_thresholds, sym_modes, tm_threshold_scale
from tissue_slab_modes.materials import eps, head_stacks
from tissue_slab_modes.modes import Mode, pick

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
P.style()
F_BT = 2.44e9
C0 = 299792458.0
DPI = 170
SHOW_EPS = False          # --eps: annotate cross-section layers with eps = eps' - j eps'' at the plotted frequency


def display_name(name):
    """Tissue name as printed on figures (the core's tissue keys use British 'grey')."""
    return name.replace("grey", "gray")


def layer_label(name, e=None):
    name = display_name(name).replace(" muscle", "")
    if SHOW_EPS and e is not None:
        e = complex(e)
        return f"{name}  ε = {e.real:.1f} − {abs(e.imag):.1f}j"
    return name


def save(fig, name):
    fig.savefig(FIG / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def site(name, f):
    """Planar stacks used in the post (thicknesses in mm; see tissue-slab-modes materials.py)."""
    L = lambda t, d: Layer(t, eps(t, f), d * 1e-3)
    if name == "abdomen":
        return Stack(f, "air", 1.0, [L("skin", 2.0), L("fat", 20.0)], "muscle", eps("muscle", f), "abdomen")
    if name == "upper arm":
        return Stack(f, "air", 1.0, [L("skin", 1.5), L("fat", 10.0)], "muscle", eps("muscle", f), "upper arm")
    if name == "wrist":
        return Stack(f, "air", 1.0, [L("skin", 1.3), L("fat", 3.0), L("tendon", 3.0)], "muscle",
                     eps("muscle", f), "volar wrist")
    if name == "occipital":
        return head_stacks(f, "typical")["occipital"]
    if name == "temple":
        return head_stacks(f, "typical")["temporal"]
    raise KeyError(name)


def domain_rgb(F, S, sat=0.42):
    """Domain coloring: hue = arg F, banded lightness = log10|F| (half-decade bands).
    The overflow scale exp(S) is real and positive, so arg F is exact."""
    hue = (np.angle(F) / (2 * np.pi)) % 1.0
    lg = (np.log(np.abs(F) + 1e-300) + S) / np.log(10)
    band = 2 * lg - np.floor(2 * lg)
    val = 0.80 + 0.20 * band
    return hsv_to_rgb(np.dstack([hue, np.full_like(hue, sat), val]))


def pi_ticks(ax, lo=-1.0, hi=1.0):
    t = np.arange(lo, hi + 1e-9, 0.5)
    lab = {-1: r"$-\pi$", -0.5: r"$-\pi/2$", 0: "0", 0.5: r"$\pi/2$", 1: r"$\pi$"}
    ax.set_yticks(np.pi * t)
    ax.set_yticklabels([lab[round(x, 1)] for x in t])


# =========================================================================== 0. what a slab guide is
def even_resid(r, ec, ew, d):
    """Residual of the even TM slab equation (kappa/ec) tan(kappa k0 d/2) = gamma_w/ew at root r."""
    k0 = 2 * np.pi * F_BT / C0
    kap, gw = np.sqrt(ec - r.n_eff ** 2), complex(r.gamma_clad)
    return abs(kap / ec * np.tan(kap * k0 * d / 2) - gw / ew) / abs(gw / ew)


def slab_profile(r, ec, ew, d, x):
    """Even TM H_y(x) of a symmetric slab root r (core ec, thickness d, walls ew), H_y(0) = 1:
    cos(kappa k0 x) inside, cos(kappa k0 d/2) exp(-gamma_w k0 (|x| - d/2)) outside, gamma_w on r's sheet."""
    k0 = 2 * np.pi * F_BT / C0
    kap, gw = np.sqrt(ec - r.n_eff ** 2), complex(r.gamma_clad)
    assert even_resid(r, ec, ew, d) < 1e-8                         # the root satisfies the equation in the README
    inside = np.cos(kap * k0 * x)
    outside = np.cos(kap * k0 * d / 2) * np.exp(-gw * k0 * (abs(x) - d / 2))
    return np.where(abs(x) <= d / 2, inside, outside), kap, gw


def fig_slab_intro(d=8e-3, x_max=40e-3, z_max=80e-3):
    """Three symmetric slabs at 2.44 GHz drawn the way a textbook draws a slab guide: the stack with its
    ray picture (left of each pair) and the even TM0 profile H_y(x) on the same x axis (right)."""
    ef, em = eps("fat", F_BT), eps("muscle", F_BT)
    cases = [("(a) Slower core, as in a fiber", "muscle core between fat walls", em, ef, "muscle", "fat",
              "total internal reflection: the walls\nonly hold an evanescent tail"),
             ("(b) Faster core between lossy, high-ε walls", "fat core between muscle walls", ef, em, "fat", "muscle",
              "partial reflection at each grazing bounce;\nwhat refracts into the wall is absorbed"),
             ("(c) The same, with the wall loss switched off", "fat core between lossless muscle walls", ef,
              complex(em.real), "fat", "muscle", "what refracts into the wall is never\nabsorbed: it leaves for good")]
    x = np.linspace(-x_max, x_max, 1601)
    fig = plt.figure(figsize=(17, 7.0))
    subs = fig.subfigures(1, 3, wspace=0.04)
    out = {}
    for sf, (title, sub, ec, ew, cname, wname, how) in zip(subs, cases):
        rs = [r for r in sym_modes(ec, ew, d, F_BT)
              if r.loss_dB_per_cm > 0 and r.n_eff.real > 1 and even_resid(r, ec, ew, d) < 1e-8]
        r = min(rs, key=lambda r: r.loss_dB_per_cm)                 # the even TM0 mode
        H, kap, gw = slab_profile(r, ec, ew, d, x)
        out[sub] = dict(n=r.n_eff, loss=r.loss_dB_per_cm, proper=r.bound, gamma_w=gw)
        a, b = sf.subplots(1, 2, sharey=True, gridspec_kw=dict(width_ratios=[2.0, 1.15], wspace=0.05))
        sf.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.17)
        for ax in (a, b):
            for lo, hi, name in ((-x_max, -d / 2, wname), (-d / 2, d / 2, cname), (d / 2, x_max, wname)):
                ax.axhspan(lo * 1e3, hi * 1e3, color=P.fill_for(name), lw=0, zorder=0)
            for e in (-d / 2, d / 2):
                ax.axhline(e * 1e3, color=P.INK2, lw=0.8, zorder=1)
        # ray picture: grazing angle from the core's transverse wavenumber, refraction from the wall's
        th = np.arctan2(kap.real, r.n_eff.real)
        phi = np.arctan2(abs(gw.imag), r.n_eff.real)
        step = d / np.tan(th)
        zb = np.arange(0.0, z_max + step, step)
        ys = np.where(np.arange(len(zb)) % 2 == 0, -d / 2, d / 2)
        a.plot(zb * 1e3, ys * 1e3, color=P.INK, lw=1.6, zorder=3)
        for k in range(0, len(zb) - 1, 2):
            zm, ym = (zb[k] + zb[k + 1]) / 2, (ys[k] + ys[k + 1]) / 2
            a.annotate("", xy=(zm * 1e3 + 1.2 * np.cos(th), ym * 1e3 + 1.2 * np.sin(th) * np.sign(ys[k + 1] - ys[k])),
                       xytext=(zm * 1e3, ym * 1e3), arrowprops=dict(arrowstyle="-|>", color=P.INK, lw=1.2), zorder=4)
        if cname == "fat":                                          # refracted rays, faded by the field envelope
            for zk, yk in zip(zb[1:], ys[1:]):
                if zk > z_max:
                    break
                h = np.linspace(0, x_max - d / 2, 60)
                amp = np.exp(-2 * gw.real * 2 * np.pi * F_BT / C0 * h)     # power along the refracted ray
                amp = np.clip(amp / max(amp.max(), 1.0), 0, 1)
                zz, yy = zk + h / np.tan(phi), np.sign(yk) * (d / 2 + h)
                for i in range(len(h) - 1):
                    a.plot(zz[i:i + 2] * 1e3, yy[i:i + 2] * 1e3, color=P.SERIES[1], lw=1.4,
                           alpha=float(0.15 + 0.85 * amp[i]), zorder=2, solid_capstyle="butt")
        Z = z_max * 1e3
        a.set_xlim(0, Z)
        a.set_ylim(-x_max * 1e3, x_max * 1e3)
        a.set_xlabel("z, along the guide (mm)")
        a.set_ylabel("x, across the layers (mm)")
        a.text(2, (d / 2 + 1.5e-3) * 1e3, rf"wall $\varepsilon_w$ ({wname})", fontsize=10, color=P.INK, va="bottom")
        a.annotate("", xy=(Z - 3, d / 2 * 1e3), xytext=(Z - 3, -d / 2 * 1e3),
                   arrowprops=dict(arrowstyle="<->", color=P.INK, lw=1.0, shrinkA=0, shrinkB=0), zorder=5)
        a.text(Z - 2, -(d / 2 + 1.5e-3) * 1e3, rf"core $\varepsilon_c$ ({cname}), $d$ = {d * 1e3:g} mm",
               fontsize=10, color=P.INK, ha="right", va="top")
        a.text(2, x_max * 1e3 - 2, how, fontsize=9.5, color=P.INK, va="top")
        if cname == "muscle":
            a.text(2, -x_max * 1e3 + 2, r"the mode: $H_y(x)\,e^{j(\omega t-\beta z)}$, $\beta = k_0 n$  $\rightarrow$",
                   fontsize=10, color=P.INK, va="bottom")
        # the mode profile on the same x axis
        b.plot(H.real, x * 1e3, color=P.SERIES[0], lw=1.7, zorder=3, label=r"Re $H_y$, one instant")
        b.plot(abs(H), x * 1e3, color=P.INK2, lw=1.0, ls="--", zorder=3, label=r"$\pm|H_y|$, envelope")
        b.plot(-abs(H), x * 1e3, color=P.INK2, lw=1.0, ls="--", zorder=3)
        b.axvline(0, color=P.MUTED, lw=0.6, zorder=1)
        lim = max(1.15, 1.08 * abs(H).max())
        b.set_xlim(-lim, lim)
        b.set_xlabel(r"$H_y$, with $H_y(0) = 1$")
        b.tick_params(labelleft=False)
        grow = "grows with |x|" if gw.real < 0 else "dies with |x|"
        b.text(0.5, 0.985, rf"$e^{{-\gamma_w k_0 (|x|-d/2)}}$" + f"\n{grow}", transform=b.transAxes,
               fontsize=9.5, ha="center", va="top", color=P.INK, bbox=dict(fc=P.fill_for(wname), ec="none", pad=1))
        b.text(0.5, 0.5, r"$\cos(\kappa k_0 x)$", transform=b.transAxes, fontsize=9.5, ha="center", va="center",
               bbox=dict(fc=P.fill_for(cname), ec="none", pad=0.5), zorder=4)
        kind = ("truly bound" if (r.n_eff ** 2 - ew).real > 0 else "\"bound\" through wall loss") if r.bound else "leaky"
        sf.suptitle(f"{title}\n{sub}", fontsize=11.5, x=0.04, ha="left", fontweight="bold")
        sf.text(0.04, 0.005,
                f"n = {r.n_eff.real:.2f} − {-r.n_eff.imag:.2f}j,  {r.loss_dB_per_cm:.2f} dB/cm\n"
                rf"$\gamma_w$ = {gw.real:.2f} {'+' if gw.imag >= 0 else '−'} {abs(gw.imag):.2f}j:  ".replace("-", "−")
                + rf"Re $\gamma_w$ {'>' if gw.real > 0 else '<'} 0, {'proper' if r.bound else 'improper'} → {kind}",
                fontsize=10.5, va="bottom", color=P.INK)
    subs[0].axes[1].legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 0.06), framealpha=0.95)
    save(fig, "00_slab_guide.png")
    return out


# =========================================================================== 1. the family at 2.44 GHz
def fig_mode_family():
    temple, occ, arm = site("temple", F_BT), site("occipital", F_BT), site("upper arm", F_BT)
    mt = find_modes(temple, "TM", zr=(-2.2, 2.2))
    mo = find_modes(occ, "TM", zr=(-2.2, 2.2))
    ma = find_modes(arm, "TM", zr=(-3, 3))
    zen = pick(temple, mt, "surface")
    fat = pick(arm, ma, "tissue", min_re_n=1.0)
    skull = pick(occ, mo, "tissue", min_re_n=1.0)
    _, leaky = temporalis_mode(1.5)
    panels = [
        (temple, zen, "Zenneck-type surface wave", "temple", 420, 25, True),
        (arm, fat, "Fat mode, \"bound\"", "upper arm, 10 mm fat", 6, 14, False),
        (occ, skull, "Skull mode, \"bound\"", "occiput, 8.4 mm bone", 6, 10, False),
        (temple, leaky, "Temporalis mode, leaky", "temple, 1.5 mm squama", 6, 14, False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 6.4), gridspec_kw=dict(wspace=0.55))
    for ax, (st, m, title, where, above, below, logy) in zip(axes, panels):
        pr = mode_profile(st, m.n_eff, m.gamma_top, m.gamma_bot, "TM", above=above * 1e-3,
                          below=below * 1e-3, npts=4000)
        frac = power_fractions(st, pr)
        x = pr["x"] * 1e3
        edges = pr["edges"] * 1e3
        bot = edges[-1] - below
        spans = [(0, above, st.top_name, st.eps_top)]
        spans += [(edges[i + 1], edges[i], L.name, L.eps) for i, L in enumerate(st.layers)]
        spans += [(bot, edges[-1], st.bot_name, st.eps_bot)]
        for lo, hi, name, e in spans:
            ax.axhspan(lo, hi, color=P.fill_for(name), lw=0)
            if hi - lo > (1.2 if not logy else 0.0) and name != "air":
                ax.text(0.98, (lo + hi) / 2, layer_label(name, e), ha="right", va="center",
                        fontsize=7.0 if SHOW_EPS else 7.5, color=P.INK2, transform=ax.get_yaxis_transform())
        for e in edges:
            ax.axhline(e, color=P.MUTED, lw=0.5)
        h = np.abs(pr["Hy"])
        sel = (x >= bot) & (x <= above)
        ax.plot(h[sel] / h[sel].max(), x[sel], color=P.SERIES[0], lw=2.2)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(bot, above)
        if logy:
            ax.set_yscale("symlog", linthresh=15, linscale=1.4)
            ax.set_yticks([-20, -10, 0, 10, 100, 400])
            ax.set_yticklabels(["-20", "-10", "0", "10", "100", "400"])
            ax.text(0.5, 200, "air", ha="center", fontsize=8, color=P.INK2)
        ax.set_xlabel(r"|H$_y$| (normalized)")
        ax.set_ylabel("height above skin (mm)" if ax is axes[0] else "")
        groups = {}
        for k, v in frac.items():
            g = "bone" if "bone" in k else k
            groups[g] = groups.get(g, 0.0) + v
        core = max(groups.items(), key=lambda kv: kv[1])
        ax.set_title(f"{title}\n{where}", loc="left")
        where_it_is = (f"{100 * core[1]:.0f} % of power in {display_name(core[0])}" if m.is_bound
                       else f"leaky: the field grows into the {display_name(st.bot_name)}")
        ax.text(0.02, -0.17, f"n = {m.n_eff.real:.2f} {m.n_eff.imag:+.2f}j   {m.loss_dB_per_cm:.1f} dB/cm\n"
                f"sheet {m.sheet};  {where_it_is}", transform=ax.transAxes, fontsize=8, color=P.INK, va="top")
    fig.suptitle("Four TM modes of layered tissue at 2.44 GHz (Bluetooth)", x=0.06, ha="left",
                 fontweight="bold", y=1.0)
    save(fig, "01_mode_family.png")
    return dict(zenneck=zen, fat=fat, skull=skull, leaky=leaky)


# =========================================================================== 2. pole map (APM)
def fig_pole_map():
    st = site("upper arm", F_BT)
    zr = (-3.0, 3.0)
    log = APMLog()
    modes = find_modes(st, "TM", zr=zr, log=log)
    n = 900
    X, Y = np.meshgrid(np.linspace(*zr, n), np.linspace(-np.pi, np.pi, n))
    Fz, S = dispersion_scaled(X + 1j * Y, st, "TM")
    _, _, code = P.sheet_map(st, zr, (-np.pi, np.pi), n)

    fig = plt.figure(figsize=(15, 6.6))
    a = fig.add_axes([0.05, 0.1, 0.5, 0.78])
    b = fig.add_axes([0.63, 0.1, 0.35, 0.78])
    a.imshow(domain_rgb(Fz, S), extent=(*zr, -np.pi, np.pi), origin="lower", aspect="auto",
             interpolation="bilinear")
    a.contour(X, Y, code, levels=[0.5, 1.5, 2.5], colors=P.INK, linewidths=0.9)
    P.plot_apm_boxes(a, log)
    for p in a.patches:
        p.set_alpha(0.35)
    for m in modes:
        a.plot(m.z.real, m.z.imag, P.SHEET_MARK[m.sheet], ms=7.5, mfc=P.SHEET_COLOR[m.sheet], mec="white", mew=1.4)
    a.set_xlim(*zr)
    a.set_ylim(-np.pi, np.pi)
    pi_ticks(a)
    a.set_xlabel(r"Re $z$ = ln|t|")
    a.set_ylabel(r"Im $z$ = arg t")
    a.set_title("Upper arm (skin 1.5 | fat 10 mm | muscle), TM, 2.44 GHz: the dispersion function on "
                "the uniformized plane\nhue = arg F, bands = |F| (half-decades); every color wheel is a zero; "
                "black lines separate the four Riemann sheets", loc="left", fontsize=9.5)

    P.plot_nplane(b, modes, nmax=4.0, imax=7.0)
    sheet_label = {"P/P": "proper / proper", "I/P": "improper air / proper body (leaky to air)",
                   "P/I": "proper air / improper body (leaky into body)", "I/I": "improper / improper"}
    b.legend(handles=[Line2D([], [], ls="", marker=P.SHEET_MARK[s], ms=7, mfc=P.SHEET_COLOR[s], mec=P.SURF,
                             label=f"{s}: {sheet_label[s]}") for s in P.SHEET_ORDER], loc="upper left", fontsize=8.5)
    fat = pick(st, modes, "tissue", min_re_n=1.0)
    b.set_xlim(0, 3.0)
    b.set_ylim(-1.5, 1.5)
    lab_fat = f"fat mode, \"bound\" through wall loss\n{fat.loss_dB_per_cm:.1f} dB/cm"
    b.annotate(lab_fat, (fat.n_eff.real, fat.n_eff.imag), xytext=(-150, -40), textcoords="offset points",
               fontsize=8.5, arrowprops=dict(arrowstyle="-", color=P.MUTED, lw=0.8))
    twin = min([m for m in modes if m.sheet == "I/P" and m.loss_dB_per_cm > 0], key=lambda m: abs(m.n_eff - fat.n_eff))
    b.annotate(f"leaky twin (grows into air)\n{twin.loss_dB_per_cm:.2f} dB/cm", (twin.n_eff.real, twin.n_eff.imag),
               xytext=(20, -35), textcoords="offset points", fontsize=8.5,
               arrowprops=dict(arrowstyle="-", color=P.MUTED, lw=0.8))
    b.set_title("The same zeros as complex effective index n = (β − jα)/k₀", loc="left")
    save(fig, "02_pole_map.png")
    return modes, len(log.boxes)


# =========================================================================== 3. loss guiding vs gain guiding
def fig_threshold():
    ef, em = eps("fat", F_BT), eps("muscle", F_BT)
    d = 8e-3
    kd = 2 * np.pi * F_BT / C0 * d
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(wspace=0.18))
    out = {}
    for ax, pol in zip(axes, ("TM", "TE")):
        res = antiguide_thresholds(ef.real, em.real, -em.imag, d, F_BT, pol=pol, s_max=12.0)
        zs = {k: np.array([r.z for _, r in p]) for k, p in res["paths"].items()}
        allz = np.concatenate(list(zs.values()))
        zr = (allz.real.min() - 0.35, allz.real.max() + 0.35)
        zi = (min(allz.imag.min(), np.pi / 2) - 0.35, max(allz.imag.max(), np.pi / 2) + 0.35)
        n = 700
        X, Y = np.meshgrid(np.linspace(*zr, n), np.linspace(*zi, n))
        Fz, S = F_sym_np(X + 1j * Y, ef, em, kd, pol)
        ax.imshow(domain_rgb(Fz, S), extent=(*zr, *zi), origin="lower", aspect="auto", interpolation="bilinear")
        ax.axhline(np.pi / 2, color=P.INK, lw=1.2)
        ax.text(zr[0] + 0.03, np.pi / 2 + 0.03, "leaky (improper): field grows into the cladding",
                fontsize=8.5, va="bottom", color=P.INK)
        ax.text(zr[0] + 0.03, np.pi / 2 - 0.03, "proper: field decays into the cladding",
                fontsize=8.5, va="top", color=P.INK)
        styles = {"loss": dict(color=P.SERIES[1], lw=3.2, label="add LOSS to the walls (tissue)"),
                  "gain": dict(color=P.SERIES[0], lw=1.6, ls=(0, (4, 2)), label="add GAIN to the core (Siegman)")}
        for k in ("loss", "gain"):
            z = zs[k]
            ax.plot(z.real, z.imag, **styles[k])
            s_th = res[f"s_{k}"]
            if s_th is not None:
                sv = np.array([s for s, _ in res["paths"][k]])
                zt = np.interp(s_th, sv, z.real) + 1j * np.interp(s_th, sv, z.imag)
                ax.plot(zt.real, zt.imag, "o", ms=7, mfc="white", mec=styles[k]["color"], mew=2)
        z0 = zs["loss"][0]
        ax.plot(z0.real, z0.imag, "s", ms=8, mfc=P.INK, mec="white")
        ax.annotate("lossless antiguide\n(fat ε'=5.3 inside muscle ε'=52.7)", (z0.real, z0.imag),
                    xytext=(10, 12), textcoords="offset points", fontsize=8.5)
        # real tissue: s = 1 on the loss path
        s1 = min(res["paths"]["loss"], key=lambda sr: abs(sr[0] - 1.0))[1]
        ax.plot(s1.z.real, s1.z.imag, "*", ms=15, mfc=P.SERIES[3], mec=P.INK, mew=0.8)
        state = "\"bound\"" if s1.bound else "still leaky"
        ax.annotate(f"real muscle loss (lossless fat): {s1.loss_dB_per_cm:.1f} dB/cm, {state}", (s1.z.real, s1.z.imag),
                    xytext=(10, -14), textcoords="offset points", fontsize=8.5)
        ax.set_xlim(*zr)
        ax.set_ylim(*zi)
        ax.set_xlabel(r"Re $z$ = ln|γ$_{clad}$|")
        ax.set_ylabel(r"Im $z$ = arg γ$_{clad}$")
        thr = (f"thresholds: loss s = {res['s_loss']:.2f}, gain s = {res['s_gain']:.2f}"
               if res["s_loss"] and res["s_gain"] else "")
        ax.set_title(f"{pol}0 of an 8 mm fat slab in muscle, 2.44 GHz\n"
                     + ("paths coincide exactly: only V² = ΔN + jG matters" if pol == "TE"
                        else "paths differ: ε also enters the TM boundary condition") + f"\n{thr}",
                     loc="left", fontsize=9.5)
        ax.legend(loc="lower right", fontsize=8.5)
        out[pol] = res
    fig.suptitle("Turning up the imaginary contrast moves the pole across the branch cut "
                 "(o = threshold, s = imaginary contrast / real muscle loss)", x=0.06, ha="left",
                 fontweight="bold", y=1.0)
    save(fig, "03_loss_vs_gain_guiding.png")
    return out


# =========================================================================== 4. lossy walls (MIM / coal seam)
def tm0_sym(d, eclad, ecore):
    rs = [r for r in sym_modes(ecore, eclad, d, F_BT) if r.loss_dB_per_cm > 0 and r.n_eff.real > 1]
    b = [r for r in rs if r.bound]
    return min(b or rs, key=lambda r: r.loss_dB_per_cm)


def fig_lossy_walls():
    ef, em = eps("fat", F_BT), eps("muscle", F_BT)
    k0 = 2 * np.pi * F_BT / C0
    pec = -20 / np.log(10) * np.sqrt(ef).imag * k0 / 100          # TEM between perfect plates
    ds = np.geomspace(1e-3, 40e-3, 34)
    rd = [tm0_sym(d, em, ef) for d in ds]
    _, path = tm_threshold_scale(ef, em.real, em.imag, 8e-3, F_BT, s_max=40.0)
    ss = np.array([s for s, _ in path])
    ls = np.array([r.loss_dB_per_cm for _, r in path])
    bs = np.array([r.bound for _, r in path])

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(wspace=0.3))
    nre = np.array([r.n_eff.real for r in rd])
    bound = np.array([r.bound for r in rd])
    a.plot(ds * 1e3, nre, color=P.SERIES[0])
    a.plot(ds[~bound] * 1e3, nre[~bound], "o", ms=4, mfc="white", mec=P.SERIES[0])
    a.axhline(np.sqrt(ef).real, color=P.MUTED, lw=1, ls="--")
    a.text(40, np.sqrt(ef).real + 0.04, "bulk fat √ε'", ha="right", fontsize=8.5, color=P.INK2)
    a.set_xscale("log")
    a.set_xlabel("fat thickness d (mm)")
    a.set_ylabel(r"Re $n_{eff}$")
    a.set_title("Index rises as the gap closes;\nresistive walls slow and damp it together", loc="left")

    lo = np.array([r.loss_dB_per_cm for r in rd])
    b.plot(ds * 1e3, lo, color=P.SERIES[1], label="TM0, fat between real muscle")
    b.plot(ds[~bound] * 1e3, lo[~bound], "o", ms=4, mfc="white", mec=P.SERIES[1], label="… leaky (below threshold)")
    b.axhline(pec, color=P.MUTED, lw=1, ls="--")
    b.text(1.05, pec * 1.12, f"perfect-metal walls: fat loss only, {pec:.2f} dB/cm", fontsize=8.5, color=P.INK2)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("fat thickness d (mm)")
    b.set_ylabel("modal loss (dB/cm)")
    b.legend(fontsize=8.5, loc="upper right")
    b.set_title("Thicker core, lower loss: wall loss\nis diluted over the gap", loc="left")

    c.plot(ss[bs], ls[bs], color=P.SERIES[2], label="\"bound\"")
    c.plot(ss[~bs], ls[~bs], color=P.SERIES[2], ls=":", lw=2.4, label="leaky")
    c.axhline(pec, color=P.MUTED, lw=1, ls="--")
    c.axvline(1.0, color=P.SERIES[3], lw=1.2)
    c.text(1.08, ls.max() * 0.97, "real muscle", fontsize=8.5, color=P.INK2, va="top")
    c.set_xscale("symlog", linthresh=0.3, linscale=0.6)
    c.set_xlim(0, ss.max())
    c.set_xticks([0, 0.3, 1, 3, 10, 30])
    c.set_xticklabels(["0", "0.3", "1", "3", "10", "30"])
    c.set_xlabel("muscle loss scale s (σ → s·σ)")
    c.set_ylabel("modal loss (dB/cm)")
    c.legend(fontsize=8.5)
    c.set_title("Coal-seam effect: lossier walls\nguide better (8 mm fat)", loc="left")
    save(fig, "04_lossy_walls.png")
    return dict(pec=pec, path=[(s, r.loss_dB_per_cm) for s, r in path if s in (0.0, 1.0)])


# =========================================================================== 5. Zenneck vs SPP
def eps_gold(f):
    """Drude gold, Ordal et al. 1985 (exp(+jwt): Im < 0 is loss)."""
    w = 2 * np.pi * f
    wp, g = 2 * np.pi * C0 * 72800e2, 2 * np.pi * C0 * 215e2
    return 1 - wp ** 2 / (w * (w - 1j * g))


def single_interface(e):
    """TM surface wave on an air | e interface: n = sqrt(e/(1+e)), proper air-side decay."""
    n = np.sqrt(e / (1 + e))
    n = np.where(n.imag > 0, -n, n)
    ga = np.sqrt(n ** 2 - 1)
    ga = np.where(ga.real < 0, -ga, ga)
    return n, ga


def fig_zenneck_spp():
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.0), gridspec_kw=dict(wspace=0.25))
    ft = np.geomspace(1e7, 1e11, 300)
    fg = np.geomspace(1e7, 4e14, 500)
    curves = [("skin (Gabriel)", ft, np.array([eps("skin", f) for f in ft]), P.SERIES[1]),
              ("muscle (Gabriel)", ft, np.array([eps("muscle", f) for f in ft]), P.SERIES[2]),
              ("gold (Drude)", fg, eps_gold(fg), P.SERIES[3])]
    for lab, f, e, col in curves:
        n, ga = single_interface(e)
        depth = 1 / ga.real / (2 * np.pi)                     # 1/e depth in air, in vacuum wavelengths
        a.plot(f, depth, color=col, label=lab)
        b.plot(f, n.real - 1, color=col, label=lab)
    for fx, lab in ((2.44e9, "Bluetooth"), (3e13, "10 µm (gold)")):
        for ax in (a, b):
            ax.axvline(fx, color=P.MUTED, lw=0.8, ls=":")
        a.text(fx * 1.15, 3e3, lab, fontsize=8.5, color=P.INK2, rotation=90, va="top")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("frequency (Hz)")
    a.set_ylabel(r"air-side 1/e field depth  /  λ$_0$")
    a.set_title("Confinement: both reaches grow with |ε|, but a Zenneck\n"
                "wave's grows far faster as the surface's loss angle shrinks", loc="left")
    a.legend()
    b.set_xscale("log")
    b.set_yscale("symlog", linthresh=1e-5)
    b.axhline(0, color=P.INK, lw=0.8)
    b.text(2e7, 3e-4, "Re n > 1: slow, truly bound (Re ε < −1)  → surface plasmon", fontsize=8.5)
    b.text(2e7, -3e-4, "Re n < 1: fast, tilted wavefront (Re ε > 0)  → Zenneck wave", fontsize=8.5, va="top")
    b.set_xlabel("frequency (Hz)")
    b.set_ylabel(r"Re $n$ − 1")
    b.set_title("Same formula n = √(ε/(1+ε));\nthe sign of Re ε decides slow vs fast", loc="left")
    save(fig, "05_zenneck_vs_plasmon.png")


# =========================================================================== 6. frequency
BANDS = [(402e6, "MICS\n402 MHz", 0), (915e6, "ISM\n915 MHz", 0), (2.44e9, "BLE\n2.44 GHz", 0),
         (5.8e9, "Wi-Fi 5.8", 1), (6.49e9, "UWB 6.5", 0), (7.99e9, "UWB 8.0", 1)]


def _follow(st, top, bot, n_prev, dz=None):
    """Polish the root of `st` that continues a previous root (index n_prev on sheet top/bot):
    map n_prev onto this stack's t-plane and Newton/secant in z = ln t. Local, so it follows one
    branch, where a global APM search returns every root and leaves the matching to the caller.
    dz, if given, sets the secant's second starting point to z0 + dz (mpmath's default, z0 + 1/4,
    can overshoot to a far root)."""
    tp = TPlane(st)
    z0 = np.log(tp.t_from_n(n_prev, top, bot))
    start = mp.mpc(z0) if dz is None else (mp.mpc(z0), mp.mpc(z0 + dz))
    try:
        z = complex(mp.findroot(lambda z: dispersion_mp(z, st, "TM"), start))
    except (ValueError, ZeroDivisionError):
        return None
    c = classify(np.exp(z), tp)
    n = c["n_eff"]
    if abs(n - n_prev) >= 0.3:
        return None
    return Mode("TM", z, c["t"], n, c["gamma_top"], c["gamma_bot"], c["top"], c["bot"],
                loss_dB_per_cm=-20 / np.log(10) * n.imag * st.k0 / 100, residual=0.0)


def track(name, f0=F_BT, f_lo=0.3e9, f_hi=10e9, n=60):
    """Identify the TM mode made "bound" by wall loss at f0 with the APM, then follow it up and down
    in frequency by local continuation (it may cross from "bound" to a leaky sheet) until lost."""
    st = site(name, f0)
    m0 = pick(st, find_modes(st, "TM", zr=(-3, 3)), "tissue", min_re_n=1.0)
    rows = [(f0, m0)]
    for fs in (np.geomspace(f0, f_hi, n)[1:], np.geomspace(f0, f_lo, n)[1:]):
        m = m0
        for f in fs:
            m = _follow(site(name, f), m.top, m.bot, m.n_eff)
            if m is None:
                break
            rows.append((f, m))
    rows.sort(key=lambda r: r[0])
    return [(f, m.loss_dB_per_cm, m.n_eff, m.is_bound) for f, m in rows]


def _grow(mask):
    """Widen a boolean mask by one sample on each side so adjacent plotted segments meet."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    return out


def _plot_branch(a, b, rows, col, lab, ls="-"):
    """Loss (a) and beta/2alpha (b) of one tracked branch: `ls` where bound, dotted where leaky,
    faded where overdamped (beta/2alpha < 1)."""
    f = np.array([r[0] for r in rows]) / 1e9
    lo = np.array([r[1] for r in rows])
    n = np.array([r[2] for r in rows])
    bd = np.array([r[3] for r in rows])
    q = n.real / (-2 * n.imag)                                     # beta / (2 alpha)
    for ax, y in ((a, lo), (b, q)):
        for wave, alpha in ((q >= 1, 1.0), (q < 1, 0.35)):
            for sheet, style in ((bd, ls), (~bd, ":")):
                m = _grow(wave) & _grow(sheet)
                ax.plot(f, np.where(m, y, np.nan), color=col, ls=style, alpha=alpha,
                        lw=2.4 if style == ":" else None,
                        label=lab if (alpha == 1.0 and style == ls and ax is a) else None)


def fig_frequency():
    sites = [("abdomen", "abdomen, 20 mm fat"), ("upper arm", "upper arm, 10 mm fat"),
             ("wrist", "volar wrist, 3 mm fat"), ("occipital", "skull, occiput")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.0), gridspec_kw=dict(wspace=0.22))
    out = {}
    for (key, lab), col in zip(sites, P.SERIES):
        out[key] = track(key)
        _plot_branch(a, b, out[key], col, lab)
    # a second bound skull mode, identified at 5.8 GHz, loses less than the 2.44 GHz branch at UWB
    out["occipital, 2nd mode"] = track("occipital", f0=5.8e9)
    _plot_branch(a, b, out["occipital, 2nd mode"], P.SERIES[3], "skull, 2nd branch (mostly fat)", ls="--")
    for ax in (a, b):
        ax.set_xscale("log")
        for fx, lab, row in BANDS:
            ax.axvline(fx / 1e9, color=P.MUTED, lw=0.7, ls=":")
            ax.text(fx / 1e9, 1.0 + 0.07 * row, lab, transform=ax.get_xaxis_transform(), fontsize=7.5,
                    ha="center", va="bottom", color=P.INK2)
        ax.set_xlabel("frequency (GHz)")
        ax.set_xlim(0.3, 10)
        ax.set_xticks([0.3, 0.5, 1, 2, 5, 10])
        ax.set_xticklabels(["0.3", "0.5", "1", "2", "5", "10"])
        ax.minorticks_off()
    a.set_yscale("log")
    a.set_yticks([1, 2, 3, 5, 10, 20])
    a.set_yticklabels(["1", "2", "3", "5", "10", "20"])
    a.set_ylabel("modal loss (dB/cm)")
    a.legend(loc="lower right", fontsize=8.5, title_fontsize=8.5,
             title="solid: \"bound\"   dotted: leaky\nfaded: overdamped (β/2α < 1)")
    b.set_yscale("log")
    b.set_yticks([0.1, 0.3, 1, 3])
    b.set_yticklabels(["0.1", "0.3", "1", "3"])
    b.axhline(1, color=P.MUTED, lw=0.8)
    b.axhspan(0.01, 1, color=P.MUTED, alpha=0.08, lw=0)
    b.set_ylim(0.06, 5)
    b.text(0.32, 0.8, "overdamped: decays faster than it oscillates", fontsize=8, color=P.INK2, va="top")
    b.set_ylabel(r"β / 2α   (rad of phase per e-fold, 4.34 dB, of power decay)")
    a.set_title("Modal loss vs frequency\n\n", loc="left")
    b.set_title("How wave-like it is: β/2α > 1 means more phase than decay\n\n", loc="left")
    save(fig, "06_frequency.png")
    return out

# =========================================================================== 7. skull: bound -> leaky
def temple_with_bone(f, d_mm):
    """Temple stack (tissue-slab-modes 'typical' anatomy) with the squama thickness set to d_mm."""
    st = site("temple", f)
    layers = [Layer(L.name, L.eps, d_mm * 1e-3) if "bone" in L.name else L for L in st.layers]
    return Stack(f, st.top_name, st.eps_top, layers, st.bot_name, st.eps_bot, f"temple, bone {d_mm} mm")


def temporalis_mode(d_mm, f=F_BT):
    """The temporalis-guided TM mode at squama thickness d_mm: the lowest-loss bound mode at 6 mm
    bone, continued exactly to d_mm (it turns leaky into the brain below ~1.64 mm). Another P/I
    root near n = 4.8 has lower dB/cm but its field grows ~2x within 14 mm of brain: a brain-side
    leaky wave, not this mode."""
    st = temple_with_bone(f, 6.0)
    m = min([m for m in find_modes(st, "TM", zr=(-2.2, 2.2)) if m.is_bound and m.loss_dB_per_cm > 0
             and 2.0 < m.n_eff.real < 7.0], key=lambda m: m.loss_dB_per_cm)
    for d in np.linspace(6.0, d_mm, 60)[1:]:
        st = temple_with_bone(f, d)
        nxt = _follow(st, m.top, m.bot, m.n_eff)
        if nxt is None:                                # e.g. d_mm = 2.0 or 2.5: the default secant overshoots
            nxt = _follow(st, m.top, m.bot, m.n_eff, dz=1e-3)
        if nxt is None:
            raise RuntimeError(f"temporalis mode lost at {d:.2f} mm bone")
        m = nxt
    return st, m


def fig_skull_thickness():
    """Follow the bone-guided TM mode from thick (bound) to thin (leaky into the brain) temporal bone."""
    ds = np.linspace(6.0, 0.6, 46)
    rows, prev = [], None
    for d in ds:
        st = temple_with_bone(F_BT, d)
        ms = [m for m in find_modes(st, "TM", zr=(-2.2, 2.2)) if m.loss_dB_per_cm > 0 and m.top == "proper"
              and 2.0 < m.n_eff.real < 7.0]
        if prev is None:
            m = min([m for m in ms if m.is_bound], key=lambda m: m.loss_dB_per_cm)
        else:
            m = min(ms, key=lambda m: abs(m.n_eff - prev), default=None)
            if m is None or abs(m.n_eff - prev) > 0.6:
                break
        prev = m.n_eff
        rows.append((d, m))
    d = np.array([r[0] for r in rows])
    lo = np.array([r[1].loss_dB_per_cm for r in rows])
    nr = np.array([r[1].n_eff.real for r in rows])
    bd = np.array([r[1].is_bound for r in rows])
    occ = pick(site("occipital", F_BT), find_modes(site("occipital", F_BT), "TM", zr=(-2.2, 2.2)), "tissue",
               min_re_n=1.0)

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(wspace=0.3, width_ratios=[1.2, 1, 1]))
    for ax, y, lab in ((a, lo, "modal loss (dB/cm)"), (b, nr, r"Re $n_{eff}$")):
        ax.plot(d[bd], y[bd], "o-", ms=3.5, color=P.SERIES[0], label="\"bound\" (decays into the brain)")
        ax.plot(d[~bd], y[~bd], "^:", ms=4, color=P.SERIES[2], label="leaky (grows into the brain)")
        ax.axvspan(1.0, 2.0, color=P.SERIES[3], alpha=0.12, lw=0)
        ax.set_xlabel("temporal bone thickness (mm)")
        ax.set_ylabel(lab)
    a.annotate("measured squama\n1.5 ± 0.5 mm", xy=(1.5, 1.0), xycoords=("data", "axes fraction"),
               xytext=(3.4, 0.97), textcoords=("data", "axes fraction"), ha="left", va="top",
               fontsize=8.5, color=P.INK2, arrowprops=dict(arrowstyle="-", color=P.MUTED, lw=0.7))
    a.legend(fontsize=8.5, loc="lower left")
    a.set_title("Temple: mode vs temporal bone thickness, 2.44 GHz\n"
                "(scalp | fat | temporalis | bone | dura | CSF | brain)", loc="left")
    b.set_title("Index: well below muscle (7.3) and brain (7.1)", loc="left")
    # profiles: thick (bound) vs measured (leaky)
    c.axhspan(0, 8.1, color=P.fill_for("muscle"), lw=0)
    c.text(0.98, 4, layer_label("temporalis", eps("muscle", F_BT)), ha="right", fontsize=8, color=P.INK2,
           transform=c.get_yaxis_transform())
    c.axhspan(-4.0, 0, color=P.fill_for("bone"), lw=0)
    c.axhline(-1.5, color=P.MUTED, lw=0.8, ls="--")
    c.text(0.98, -2.8, layer_label("bone (4 mm; dashed: 1.5 mm)", eps("bone cortical", F_BT)), ha="right",
           fontsize=8, color=P.INK2, transform=c.get_yaxis_transform())
    for (dd, col, ls) in ((4.0, P.SERIES[0], "-"), (1.5, P.SERIES[2], "--")):
        st, r = temporalis_mode(dd)
        pr = mode_profile(st, r.n_eff, r.gamma_top, r.gamma_bot, "TM", above=3e-3, below=12e-3, npts=3000)
        x = pr["x"] * 1e3
        top_of_bone = [pr["edges"][i] * 1e3 for i, L in enumerate(st.layers) if "bone" in L.name][0]
        c.plot(np.abs(pr["Hy"]), x - top_of_bone, color=col, ls=ls,
               label=f"{dd} mm bone: {r.loss_dB_per_cm:.1f} dB/cm, sheet {r.sheet}")
    c.set_ylabel("height above the outer bone surface (mm)")
    c.set_xlabel(r"|H$_y$| (normalized in the layers)")
    c.legend(fontsize=8, loc="upper right")
    c.set_title("The field rides in the temporalis; thin bone\nlets it tunnel into the brain", loc="left")
    save(fig, "07_skull_thickness.png")
    return rows, occ


FIGS = dict(slab=fig_slab_intro, family=fig_mode_family, poles=fig_pole_map, threshold=fig_threshold,
            walls=fig_lossy_walls, zenneck=fig_zenneck_spp, frequency=fig_frequency, skull=fig_skull_thickness)



if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    SHOW_EPS = "--eps" in sys.argv[1:]
    for name in args or FIGS:
        res = FIGS[name]()
        if name == "family":
            for k, m in res.items():
                print(f"{k:8s} {m}")
        elif name == "poles":
            print(f"pole map: {len(res[0])} zeros, {res[1]} contour integrals")
        elif name == "threshold":
            for pol, r in res.items():
                print(pol, "s_loss", r["s_loss"], "s_gain", r["s_gain"], "G_loss", r["G_loss"], "G_gain", r["G_gain"])
        elif name == "walls":
            print("lossy walls", res)
        elif name == "frequency":
            for k, rows in res.items():
                lk = [r[0] for r in rows if not r[3]]
                print(f"{k:10s} tracked {rows[0][0]/1e9:.2f}-{rows[-1][0]/1e9:.2f} GHz; leaky "
                      f"{'nowhere' if not lk else f'{min(lk)/1e9:.2f}-{max(lk)/1e9:.2f} GHz'}")
                for fx in (0.915e9, 5.8e9):
                    f, lo, n, b = min(rows, key=lambda r: abs(np.log(r[0] / fx)))
                    print(f"    {f/1e9:.3f} GHz  n = {n.real:.3f}{n.imag:+.3f}j  {lo:.2f} dB/cm  bound={b}")
        elif name == "slab":
            for k, v in res.items():
                print(f"{k:42s} {v}")
        elif name == "skull":
            rows, occ = res
            i = next((k for k, r in enumerate(rows) if not r[1].is_bound), None)   # rows run from thick to thin bone
            if i is None:
                print(f"skull: \"bound\" over the whole range; occipital {occ}")
            else:
                last = f"{rows[i - 1][0]:.2f} mm" if i > 0 else "nowhere"
                print(f"skull: \"bound\" down to {last} (last P/P grid point; leaky from {rows[i][0]:.2f} mm); "
                      f"occipital {occ}")
            for dd, m in rows[::5]:
                print(f"  {dd:4.1f} mm  {m}")
