"""Argument Principle Method (APM) root finder.

A Python port of the Mathematica "APM Implementation Package" (numZeros / GetSubRegions
/ APM / GetSolutions), generalised to the scaled multilayer dispersion function.

Method [Delves & Lyness, Math. Comput. 21, 543 (1967); Anemogiannis & Glytsis,
J. Lightwave Technol. 10, 1344 (1992); Kravanja & Van Barel, Computing 63, 69 (1999)]:

  1. number of zeros inside a closed contour Gamma
         N = (1/2 pi j) \\oint_Gamma F'(z)/F(z) dz = (change of arg F along Gamma) / 2 pi
  2. if N > 1, split the rectangle into four and recurse (as GetSubRegions did);
  3. if N == 1, the root is the first moment
         z_0 = (1/2 pi j) \\oint_Gamma z F'(z)/F(z) dz
     (for N > 1 in a tiny box: moments s_k, k = 1..N, + Newton's identities);
  4. polish with a secant/Muller iteration on the unscaled function (mpmath).

Differences from the notebook: instead of NIntegrate of the symbolic F'/F we integrate
d(log F) directly along an adaptively refined polyline (derivative-free, cf. Kravanja &
Van Barel 1999). Since log F = log F_scaled + S exactly, the scaling used for overflow
safety does not bias either the count or the moments.

Robustness. Counting by phase increments fails silently if F turns by a multiple of 2 pi between
two samples: angle() then reports a small step. Near |Re z| ~ 3 the layer phases grow like t, so
a long edge can turn 4 pi per segment. One rule handles it: two counts of the same region must
agree, and when they do not, the finer one wins.
  - a contour segment is accepted only once its midpoint has been evaluated and both halves
    have small steps (catches single 2 pi turns; every evaluated point stays in the polyline);
  - the four children of a split trace the parent's boundary at twice the sampling density
    (their inner edges cancel), so their total is a refined count of the parent and wins;
  - a leaf uses the winding number of its fine contour, never dividing by a measured m0 that
    disagrees with the coarse count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mpmath as mp


class RootOnContour(RuntimeError):
    pass


@dataclass
class Rect:
    x0: float
    x1: float
    y0: float
    y1: float

    def corners(self):
        return [complex(self.x0, self.y0), complex(self.x1, self.y0),
                complex(self.x1, self.y1), complex(self.x0, self.y1)]

    def contains(self, z, pad=0.0):
        dx, dy = (self.x1 - self.x0) * pad, (self.y1 - self.y0) * pad
        return (self.x0 - dx <= z.real <= self.x1 + dx) and (self.y0 - dy <= z.imag <= self.y1 + dy)

    @property
    def size(self):
        return max(self.x1 - self.x0, self.y1 - self.y0)

    def split(self, fx=0.5, fy=0.5):
        xm = self.x0 + fx * (self.x1 - self.x0)
        ym = self.y0 + fy * (self.y1 - self.y0)
        return [Rect(self.x0, xm, self.y0, ym), Rect(xm, self.x1, self.y0, ym),
                Rect(xm, self.x1, ym, self.y1), Rect(self.x0, xm, ym, self.y1)]


@dataclass
class APMLog:
    """Record of the subdivision for visualisation."""
    boxes: list = field(default_factory=list)   # (Rect, count)


def _logf(f, z):
    F, S = f(z)
    return F, S


def _small_steps(Fa, Sa, Fb, Sb, dphi_max):
    return (np.abs(np.angle(Fb / Fa)) <= dphi_max) & (np.abs(np.log(np.abs(Fb / Fa)) + Sb - Sa) <= 1.0)


def contour_logF(f, rect: Rect, n0=48, max_pts=200_000, dphi_max=0.35, min_seg=1e-11):
    """Adaptive polyline along the rectangle; returns z, F_scaled, S (closed).

    Each unchecked segment is split at its midpoint; it is accepted when both halves have phase
    steps <= dphi_max and log-modulus steps <= 1, otherwise both halves are checked again. A
    turn of ~2 pi between two samples, invisible to angle(F_b / F_a), shows up as two ~pi halves.
    """
    c = rect.corners() + [rect.corners()[0]]
    z = np.concatenate([np.linspace(c[i], c[i + 1], n0, endpoint=False) for i in range(4)] + [[c[0]]])
    F, S = f(z)
    ok = np.zeros(len(z) - 1, dtype=bool)
    while True:
        if not np.all(np.isfinite(F)) or np.any(F == 0):
            raise RootOnContour("non-finite or zero value on contour")
        if ok.all():
            return z, F, S
        idx = np.nonzero(~ok)[0]
        if np.any(np.abs(z[idx + 1] - z[idx]) < min_seg):
            raise RootOnContour("cannot resolve phase: root (or pole) on/near contour")
        if len(z) + len(idx) > max_pts:
            raise RootOnContour("too many contour points")
        zm = (z[idx] + z[idx + 1]) / 2
        Fm, Sm = f(zm)
        good = (_small_steps(F[idx], S[idx], Fm, Sm, dphi_max)
                & _small_steps(Fm, Sm, F[idx + 1], S[idx + 1], dphi_max))
        z = np.insert(z, idx + 1, zm)
        F = np.insert(F, idx + 1, Fm)
        S = np.insert(S, idx + 1, Sm)
        ok = np.insert(ok, idx + 1, good)
        ok[idx + np.arange(len(idx))] = good        # first halves, at their shifted positions


def contour_moments(z, F, S, kmax=1):
    """(1/2 pi j) \\oint z^k dlogF for k = 0..kmax, midpoint rule on the polyline."""
    dlog = np.log(np.abs(F[1:] / F[:-1])) + (S[1:] - S[:-1]) + 1j * np.angle(F[1:] / F[:-1])
    zm = (z[1:] + z[:-1]) / 2
    return np.array([np.sum(zm ** k * dlog) for k in range(kmax + 1)]) / (2j * np.pi)


def count_zeros(f, rect: Rect, **kw):
    z, F, S = contour_logF(f, rect, **kw)
    m0 = contour_moments(z, F, S, 0)[0]
    n = int(round(m0.real))
    if abs(m0.real - n) > 0.05 or abs(m0.imag) > 0.05 * max(1, abs(n)):
        raise RootOnContour(f"non-integer winding {m0}")
    return n, (z, F, S)


def _newton_identities(s):
    """Polynomial coefficients from power sums s[1..n] (s[0] = n)."""
    n = int(round(s[0].real))
    e = [1.0 + 0j]
    for k in range(1, n + 1):
        e.append(sum((-1) ** (i - 1) * e[k - i] * s[i] for i in range(1, k + 1)) / k)
    coeffs = [(-1) ** k * e[k] for k in range(n + 1)]
    return np.roots(coeffs)


def find_roots(f, rect: Rect, polish=None, max_depth=14, min_box=1e-6, log: APMLog | None = None,
               _depth=0, _n=None):
    """Return list of (z_root, rect, multiplicity_estimate). `_n`: zero count already known."""
    log = log if log is not None else APMLog()
    if _n is not None:
        n = _n
    else:
        try:
            n, _ = count_zeros(f, rect)
        except RootOnContour:
            # nudge the box outward asymmetrically, like the notebook's ModifySearchRegion
            g = 1e-3 * rect.size * (1 + 0.37 * _depth)
            rect = Rect(rect.x0 - g, rect.x1 + 1.3 * g, rect.y0 - 0.7 * g, rect.y1 + g)
            n, _ = count_zeros(f, rect)
    log.boxes.append((rect, n))
    if n <= 0:
        return []
    if n == 1 or rect.size < min_box or _depth >= max_depth:
        # finer polyline for accurate moments (midpoint rule error ~ h^2)
        data = contour_logF(f, rect, n0=96, dphi_max=0.04)
        m = contour_moments(*data, kmax=n)
        n_fine = int(round(m[0].real))
        if abs(m[0] - n_fine) > 0.05:
            raise RootOnContour(f"non-integer winding {m[0]} on the fine contour")
        if n_fine != n:                             # the finer integral wins
            return find_roots(f, rect, polish, max_depth, min_box, log, _depth, _n=n_fine)
        zs = [m[1] / n] if n == 1 else _newton_identities(np.concatenate([[n], m[1:]]))
        out = []
        for z0 in zs:
            zr = polish(z0, rect) if polish else None
            if zr is None:
                zr = z0
            out.append((complex(zr), rect, 1))
        return out
    for attempt in range(4):
        fx, fy = 0.5 + 0.0137 * attempt, 0.5 - 0.0113 * attempt
        sub = rect.split(fx, fy)
        try:
            counts = [count_zeros(f, r)[0] for r in sub]
            out = []
            for r, k in zip(sub, counts):
                out += find_roots(f, r, polish, max_depth, min_box, log, _depth + 1, _n=k)
            return out
        except RootOnContour:
            continue
    raise RootOnContour(f"no subdivision of {rect} counts cleanly")


def make_polisher(f_mp, dps=30, tol=1e-24, maxit=60):
    """Damped Newton polishing (mpmath, numerical derivative) confined to the APM box.

    The APM box is known to contain exactly one simple zero, so any iterate leaving
    the (slightly padded) box is rejected; this prevents the jump-to-a-neighbouring-root
    failure of unconstrained secant/Muller iterations.
    """
    def polish(z0, rect: Rect | None = None):
        with mp.workdps(dps):
            z = mp.mpc(z0)
            fz = f_mp(z)
            for _ in range(maxit):
                d = mp.diff(f_mp, z)
                if d == 0:
                    return None
                step = fz / d
                lam = mp.mpf(1)
                while lam > 1e-6:
                    zn = z - lam * step
                    fn = f_mp(zn)
                    if abs(fn) < abs(fz) and (rect is None or rect.contains(complex(zn), pad=0.05)):
                        break
                    lam /= 2
                else:
                    break
                z, fz = zn, fn
                if abs(lam * step) < tol * (1 + abs(z)):
                    break
            if rect is not None and not rect.contains(complex(z), pad=0.05):
                return None
            return complex(z)
    return polish
