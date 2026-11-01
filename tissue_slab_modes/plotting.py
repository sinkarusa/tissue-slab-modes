"""Explainer and result figures (matplotlib, static PNG for the README)."""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MRect, FancyArrowPatch

from .dispersion import TPlane, dispersion_scaled
from .stack import EPS0

# reference categorical palette (dataviz skill, light mode), fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SHEET_ORDER = ["P/P", "I/P", "P/I", "I/I"]
SHEET_COLOR = dict(zip(SHEET_ORDER, SERIES[:4]))
SHEET_MARK = {"P/P": "o", "I/P": "s", "P/I": "^", "I/I": "D"}
SHEET_LABEL = {"P/P": "proper / proper (bound)", "I/P": "improper air / proper body (leaky to air)",
               "P/I": "proper air / improper body (leaky into body)", "I/I": "improper / improper"}
TISSUE_FILL = {"air": "#ffffff", "skin": "#f3cdb9", "fat": "#f7e6a8", "muscle": "#e2a3a3",
               "temporalis muscle": "#e2a3a3", "bone": "#ece6d6", "cortical bone": "#ece6d6",
               "cancellous bone": "#ddd3bb", "csf": "#c7e3f3", "dura": "#dccbe6",
               "grey matter": "#cfc8d6", "white matter": "#ebe6ef", "brain": "#d8d1de",
               "scalp": "#f0c7b3", "galea": "#e8b9a5"}


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": False,
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "legend.frameon": False, "mathtext.fontset": "dejavusans",
    })


def fill_for(name):
    key = name.lower()
    for k, v in TISSUE_FILL.items():
        if k in key:
            return v
    return "#eeeeee"


def eps_to_sigma(eps, f):
    return -np.imag(eps) * 2 * np.pi * f * EPS0


# ----------------------------------------------------------------------------- structure
def draw_stack(ax, stack, prof=None, field="Hy", above_mm=6, below_mm=8, label_side="right"):
    """Layer cross-section (depth axis vertical, mm, 0 = skin surface) with an optional
    mode profile overlay."""
    edges = [0.0]
    for L in stack.layers:
        edges.append(edges[-1] - L.thickness * 1e3)
    top, bot = above_mm, edges[-1] - below_mm
    ax.add_patch(MRect((0, 0), 1, top, fc=fill_for(stack.top_name), ec="none"))
    for i, L in enumerate(stack.layers):
        ax.add_patch(MRect((0, edges[i + 1]), 1, edges[i] - edges[i + 1], fc=fill_for(L.name), ec="none"))
    ax.add_patch(MRect((0, bot), 1, edges[-1] - bot, fc=fill_for(stack.bot_name), ec="none"))
    for e in edges:
        ax.axhline(e, color=MUTED, lw=0.6)
    f = stack.freq

    def lab(y, name, eps, d=None):
        nm = name.replace(" muscle", "").replace(" (half-space)", " (½-space)")
        s = f"{nm}  " + rf"{eps.real:.1f}, {abs(eps_to_sigma(eps, f)):.2f} S/m"
        if d is not None:
            s += f", {d:.1f} mm"
        ax.text(1.04 if label_side == "right" else -0.02, y, s, va="center",
                ha="left" if label_side == "right" else "right", fontsize=7.6, color=INK,
                transform=ax.get_yaxis_transform())

    lab(top / 2, stack.top_name, complex(stack.eps_top))
    for i, L in enumerate(stack.layers):
        lab((edges[i] + edges[i + 1]) / 2, L.name, complex(L.eps), L.thickness * 1e3)
    lab((edges[-1] + bot) / 2, stack.bot_name + " (half-space)", complex(stack.eps_bot))
    if prof is not None:
        x = prof["x"] * 1e3
        sel = (x <= top) & (x >= bot)
        y = np.abs(prof[field])[sel]
        ax.plot(0.05 + 0.9 * y / y.max(), x[sel], color=SERIES[0], lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(bot, top)
    ax.set_xticks([])
    ax.set_ylabel("depth below skin surface (mm)")
    ax.spines["bottom"].set_visible(False)
    return edges


def equation_panel(ax):
    ax.axis("off")
    lines = [
        (r"$\mathbf{TM\ fields}\ \ H_y(x)\,e^{j(\omega t-\beta z)},\quad n=\beta/k_0,\quad \gamma_j=\sqrt{n^2-\varepsilon_j}$", 12),
        (r"$\mathbf{u}=\left[H_y,\ \frac{1}{\varepsilon}\frac{dH_y}{d(k_0x)}\right]^T$ is continuous at every interface", 10.5),
        (r"Layer $j$ (thickness $d_j$, $\phi_j=\gamma_j k_0 d_j$):", 10.5),
        (r"$\mathbf{u}_{top}=M_j\,\mathbf{u}_{bot},\qquad M_j=\left[\cosh\phi_j,\ \ \varepsilon_j\frac{\sinh\phi_j}{\gamma_j}\ \ ;\ \ \frac{\gamma_j\sinh\phi_j}{\varepsilon_j},\ \ \cosh\phi_j\right]$", 10.5),
        (r"(even in $\gamma_j$  $\Rightarrow$ interior layers add no branch cuts)", 9.5),
        (r"Body half-space: $\mathbf{u}=[1,\ \gamma_b/\varepsilon_b]^T$;   air: $H_y\propto e^{-\gamma_t k_0 x}$", 10.5),
        (r"$F(n)=\frac{\gamma_t}{\varepsilon_t}\,u_0+u_1=0,\qquad \mathbf{u}=M_1M_2\cdots M_N\,[1,\ \gamma_b/\varepsilon_b]^T$", 12),
        (r"$\gamma_t,\gamma_b$ double-valued $\Rightarrow$ 4 Riemann sheets.  Uniformise with $\gamma_t^2-\gamma_b^2=\varepsilon_b-\varepsilon_t\equiv C$:", 10.5),
        (r"$\gamma_t=\frac{\sqrt{C}}{2}\left(t+\frac{1}{t}\right),\quad \gamma_b=\frac{\sqrt{C}}{2}\left(t-\frac{1}{t}\right),\quad z=\log t,\ |\mathrm{Im}\,z|\leq\pi$", 12),
        (r"proper (decaying) side $\Leftrightarrow\ \mathrm{Re}\,\gamma>0$;   loss: $\alpha$[dB/cm] $=-8.686\,k_0\,\mathrm{Im}\,n$/100", 10.5),
    ]
    y = 0.96
    for s, fs in lines:
        ax.text(0.0, y, s, fontsize=fs, color=INK, va="top", transform=ax.transAxes)
        y -= 0.092
    ax.text(0.0, 0.0, "Refs: Chew (1990) ch.2; Smith, Houde-Walter & Forbes, IEEE JQE 28, 1520 (1992);\n"
            "Anemogiannis & Glytsis, JLT 10, 1344 (1992); Kocabas et al., PRB 79, 035120 (2009)",
            fontsize=7.8, color=INK2, va="bottom", transform=ax.transAxes)


# ----------------------------------------------------------------------------- t-plane
def sheet_map(stack, zr, zi, n=500):
    tp = TPlane(stack)
    X, Y = np.meshgrid(np.linspace(*zr, n), np.linspace(*zi, n))
    gt, gb = tp.gammas(np.exp(X + 1j * Y))
    code = (gt.real < 0).astype(int) + 2 * (gb.real < 0).astype(int)  # 0 PP,1 IP,2 PI,3 II
    return X, Y, code


def plot_tplane(ax, stack, modes, zr=(-1.6, 1.6), zi=(-np.pi, np.pi), pol="TM", n=420, contours=True):
    X, Y, code = sheet_map(stack, zr, zi, n)
    cmap = mpl.colors.ListedColormap([mpl.colors.to_rgba(c, 0.18) for c in SERIES[:4]])
    ax.pcolormesh(X, Y, code, cmap=cmap, vmin=-0.5, vmax=3.5, shading="auto", rasterized=True)
    ax.contour(X, Y, code, levels=[0.5, 1.5, 2.5], colors=MUTED, linewidths=0.5)
    if contours:
        F, S = dispersion_scaled(X + 1j * Y, stack, pol)
        L = np.log10(np.abs(F) + 1e-300) + S / np.log(10)
        ax.contour(X, Y, L, levels=np.linspace(np.percentile(L, 3), np.percentile(L, 70), 14),
                   colors=INK2, linewidths=0.35, alpha=0.6)
    for m in modes:
        if zr[0] <= m.z.real <= zr[1]:
            ax.plot(m.z.real, m.z.imag, SHEET_MARK[m.sheet], ms=6.5, mfc=SHEET_COLOR[m.sheet],
                    mec=SURF, mew=1.5)
    ax.set_xlim(*zr)
    ax.set_ylim(*zi)
    ax.set_yticks(np.pi * np.array([-1, -0.5, 0, 0.5, 1]))
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel(r"Re $z$ = ln|t|")
    ax.set_ylabel(r"Im $z$ = arg t")


def sheet_legend(ax, loc="upper left", **kw):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker=SHEET_MARK[s], ms=7, mfc=SHEET_COLOR[s], mec=SURF, label=SHEET_LABEL[s])
         for s in SHEET_ORDER]
    ax.legend(handles=h, loc=loc, fontsize=8, **kw)


def plot_nplane(ax, modes, nmax=4.0, imax=3.0):
    for m in sorted(modes, key=lambda m: SHEET_ORDER.index(m.sheet), reverse=True):  # P/P drawn last
        if abs(m.n_eff.real) <= nmax and abs(m.n_eff.imag) <= imax:
            ax.plot(m.n_eff.real, m.n_eff.imag, SHEET_MARK[m.sheet], ms=6.5, mfc=SHEET_COLOR[m.sheet],
                    mec=SURF, mew=1.5)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlim(0, nmax)
    ax.set_ylim(-imax, imax)
    ax.set_xlabel(r"Re $n_{eff}$")
    ax.set_ylabel(r"Im $n_{eff}$   (< 0: decays along +z)")


def _dipole(ax, x, y, angle_deg, L=0.045, color=INK):
    a = np.deg2rad(angle_deg)
    dx, dy = L / 2 * np.cos(a), L / 2 * np.sin(a)
    ax.plot([x - dx, x - 0.12 * dx], [y - dy, y - 0.12 * dy], color=color, lw=2.6, solid_capstyle="round")
    ax.plot([x + 0.12 * dx, x + dx], [y + 0.12 * dy, y + dy], color=color, lw=2.6, solid_capstyle="round")


def scenario_schematic(fig, H=1.6, D=0.145, R=0.085):
    """Three panels: (A) ground reflection (front view), (B) through the head,
    (C) guided/creeping path around the back (top view of a layered head)."""
    from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
    axA, axB, axC = fig.subplots(1, 3, gridspec_kw=dict(width_ratios=[0.8, 1, 1]))
    # (A) front view, not to scale vertically
    ax = axA
    ax.set_aspect("equal")
    ax.add_patch(Circle((0, H), 0.11, fc=TISSUE_FILL["skin"], ec=MUTED, lw=0.8))
    ax.add_patch(FancyBboxPatch((-0.2, 0.9), 0.4, 0.55, boxstyle="round,pad=0.02", fc=TISSUE_FILL["skin"],
                                ec=MUTED, lw=0.8))
    ax.add_patch(MRect((-0.17, 0.0), 0.14, 0.9, fc=TISSUE_FILL["skin"], ec=MUTED, lw=0.8))
    ax.add_patch(MRect((0.03, 0.0), 0.14, 0.9, fc=TISSUE_FILL["skin"], ec=MUTED, lw=0.8))
    ax.axhline(0, color=INK2, lw=1.2)
    ax.text(0.33, -0.08, "floor (concrete / wood / ground, ITU-R P.2040)", fontsize=7.5, color=INK2, ha="right")
    xl, xr = -D / 2, D / 2
    ax.plot([xl, 0, xr], [H, 0, H], color=SERIES[1], lw=1.6)
    for x in (xl, xr):
        ax.plot(x, H, "o", ms=6, mfc=SERIES[1], mec=SURF)
    ax.text(0.0, H + 0.2, "(A) floor reflection\nhorizontal dipoles, front–back\n(broadside toward the floor)",
            ha="center", fontsize=8.5, color=INK)
    ax.text(0.2, 0.55, "specular ray passes\nthrough the torso:\nbody-transparent\nupper bound", fontsize=7.5,
            color=INK2)
    ax.set_xlim(-0.45, 0.55)
    ax.set_ylim(-0.15, H + 0.5)
    ax.axis("off")
    # (B) and (C): top view of a layered head
    for ax, title in ((axB, "(B) direct, through the skull & brain\ntangential dipoles, broadside across the head"),
                      (axC, "(C) loss-guided / surface mode around the back\ndipoles normal to the skin")):
        ax.set_aspect("equal")
        rings = [(R, "skin"), (R - 0.004, "fat"), (R - 0.007, "muscle"), (R - 0.0115, "bone"),
                 (R - 0.0165, "csf"), (R - 0.0185, "brain")]
        for r, name in rings:
            ax.add_patch(Ellipse((0, 0), 2 * r * 0.92, 2 * r * 1.1, fc=fill_for(name), ec=MUTED, lw=0.4))
        ax.add_patch(Ellipse((-R * 0.92 - 0.006, 0), 0.012, 0.03, fc=TISSUE_FILL["skin"], ec=MUTED, lw=0.6))
        ax.add_patch(Ellipse((R * 0.92 + 0.006, 0), 0.012, 0.03, fc=TISSUE_FILL["skin"], ec=MUTED, lw=0.6))
        ax.text(0, R * 1.1 + 0.012, "front", ha="center", fontsize=7.5, color=INK2)
        ax.text(-0.115, -R * 1.1, "back", ha="left", fontsize=7.5, color=INK2)
        ax.set_xlim(-0.13, 0.13)
        ax.set_ylim(-0.13, 0.13)
        ax.axis("off")
        ax.set_title(title, fontsize=8.5, color=INK, loc="center")
    x0 = R * 0.92 + 0.016
    axB.annotate("", (x0 - 0.012, 0), (-x0 + 0.012, 0), arrowprops=dict(arrowstyle="-|>", color=SERIES[0], lw=1.6))
    _dipole(axB, -x0, 0, 90, L=0.03)
    _dipole(axB, x0, 0, 90, L=0.03)
    axB.text(0, 0.01, "brain ≈ 11–12 cm", ha="center", fontsize=7.5, color=INK2)
    th = np.linspace(np.pi, 2 * np.pi, 100)
    axC.plot((R * 0.92 + 0.004) * np.cos(th), (R * 1.1 + 0.004) * np.sin(th), color=SERIES[2], lw=1.8)
    axC.plot((R * 0.92 - 0.012) * np.cos(th), (R * 1.1 - 0.012) * np.sin(th), color=SERIES[3], lw=1.8, ls=(0, (1, 1.5)))
    _dipole(axC, -x0, 0, 0, L=0.03)
    _dipole(axC, x0, 0, 0, L=0.03)
    axC.text(0, -0.122, "air-side surface / creeping wave (solid)", ha="center", fontsize=7.5, color=INK)
    axC.text(0, -0.045, "skull-guided mode (dotted)", ha="center", fontsize=7.5, color=INK)
    return axA, axB, axC


def plot_apm_boxes(ax, log, max_depth_color=True):
    for rect, n in log.boxes:
        lw = 0.4 if n <= 1 else 0.8
        ax.add_patch(MRect((rect.x0, rect.y0), rect.x1 - rect.x0, rect.y1 - rect.y0, fc="none",
                           ec=INK if n > 1 else MUTED, lw=lw))
