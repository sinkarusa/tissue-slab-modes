# Modal excitation (residue) table

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

## Checks

### a. Residue formula vs. numerical contour integral

(1/2&pi;j) &oint; G(n) dn on a small circle around the pole, upper-arm stack, source at mid-fat, observer at the skin surface (x_o = 0+); both source orientations.

| pole | sheet | orientation | radius \|n-n_m\| | contour integral | closed-form residue | relative error |
|---|---|---|---|---|---|---|
| lossy-wall fat mode | P/P | normal | 0.02 | +8.445275e-02-1.090572e-01j | +8.445275e-02-1.090572e-01j | 6.872e-11 |
| lossy-wall fat mode | P/P | tangential | 0.02 | -4.078015e-02-2.064346e-02j | -4.078015e-02-2.064346e-02j | 6.872e-11 |
| its leaky twin | I/P | normal | 0.02 | -1.274365e-01+7.365770e-02j | -1.274365e-01+7.365770e-02j | 3.528e-11 |
| its leaky twin | I/P | tangential | 0.02 | +1.558600e-02+5.068708e-02j | +1.558600e-02+5.068708e-02j | 3.529e-11 |

Worst relative error over these 4 pole/orientation combinations (one proper P/P pole, one improper I/P pole): **6.87e-11** (&le; 1e-6 required).

### b. Reciprocity

The source jump conditions (sec. 1 above) make the normal-current Green's function a classic shunt-current-source transfer admittance on the TM transmission line; standard telegrapher's-equation (Z21 = Z12) reciprocity then holds **after weighting by the permittivity at the source**, not for the bare field ratio: eps(x_s)&middot;G(x_o; x_s) = eps(x_o)&middot;G(x_s; x_o). (The tangential/series-type source does *not* satisfy the same-component swap -- as expected from general dyadic EM reciprocity, H-due-to-J pairs with E-due-to-M, not with itself; verified numerically below to fail for both the plain and eps-weighted forms, so the normal orientation is the one checked quantitatively.) Source/observer: skin surface (air, eps=1) and mid-fat (eps &asymp; 5.3-0.77j), upper-arm stack, normal orientation.

| pole | sheet | G(surface; mid-fat) | G(mid-fat; surface) | eps(mid-fat)&middot;G(surf;fat) / eps(surf)&middot;G(fat;surf) |
|---|---|---|---|---|
| lossy-wall fat mode | P/P | +8.445275e-02-1.090572e-01j | +3.623732e-01-6.407000e-01j | 1.000000000+2.686e-11j |
| its leaky twin | I/P | -1.274365e-01+7.365770e-02j | -6.165192e-01+4.867139e-01j | 1.000000000-8.220e-11j |

Worst deviation of the weighted ratio from 1 over the proper and improper pole: **8.25e-11**.

### c. Cross-check vs. `tissue_slab_modes.sommerfeld.residue_R`

For a source in air, the core's `R_TM(q, stack)` is exactly the stack's TM reflection coefficient seen from the top; its denominator `_D(q,stack) = gamma0/eps_top + Y` equals F_raw(n=q)/H_B(0) (identifying q with n -- both are beta/k0 along the stack's own propagation axis). Tracking the derivation through (sec. 1,3 above) for a **tangential** source at height h -> 0 gives the exact relation Res_n[G_tangential] = -(1/2)&middot;exp(-gamma_top&middot;k0 h)&middot;residue_R(n_m, stack), i.e. a mode-independent common factor of -1/2 in the h -> 0 limit. Upper-arm stack, two P/P modes (proper-top, matching residue_R's own branch convention), h = 0.0196 &micro;m:

| pole | n_eff | residue (mine) | residue_R (core) | common factor (mine / residue_R) |
|---|---|---|---|---|
| lossy-wall fat mode | 1.8321-0.6667j | -4.322503e-02-7.838469e-02j | +8.645031e-02+1.567696e-01j | -0.499999-3.855e-07j |
| other P/P mode | 0.3750-6.1728j | +3.006925e-02+5.056042e-03j | -6.013858e-02-1.011171e-02j | -0.500000-3.127e-06j |

Ratio of residues between the two modes, mine: -1.824264-2.300062j; core's -(1/2)exp(-gamma_top h) residue_R ratio: -1.824264-2.300062j (match 3.05e-09 relative). Both common factors are -0.499999-0.000000j and -0.500000-0.000003j (&rarr; -0.5 as h &rarr; 0): **up to the common factor -1/2, the two conventions match.**

### d. Which poles are in the field

A residue only matters if the pole is *captured* -- actually reached when the exact real-n spectral integral for the field is closed onto vertical branch cuts (Tamir & Oliner 1963; `in_region`/`captured`, Method point 5). Checked directly: summing the modal term of every captured pole plus the two vertical branch-cut integrals (top, bottom half-space) must reproduce the exact real-axis integral for H_y(z) at the skin surface. Buried sources (mid-fat, mid-diploe, mid-temporalis), z = 6 cm, both orientations; the real-axis integral is split with extra breakpoints near any near-real-axis pole (e.g. the temple's P/P Zenneck root) to keep quadrature accurate past it.

| stack | orientation | pole set | relative mismatch |
|---|---|---|---|
| upper arm | normal | captured set (14 poles) | 1.27e-10 |
| upper arm | normal | captured + Zenneck-type pole (named OUT) | 1.83e+00 |
| upper arm | normal | captured + its leaky twin (named OUT) | 2.76e-01 |
| upper arm | normal | captured - lossy-wall fat mode (named IN) | 4.78e-01 |
| upper arm | tangential | captured set (14 poles) | 4.77e-10 |
| upper arm | tangential | captured + Zenneck-type pole (named OUT) | 3.01e+00 |
| upper arm | tangential | captured + its leaky twin (named OUT) | 1.07e-01 |
| upper arm | tangential | captured - lossy-wall fat mode (named IN) | 1.71e-01 |
| occipital | normal | captured set (19 poles) | 1.83e-10 |
| occipital | normal | captured + Zenneck-type pole (named OUT) | 1.38e+00 |
| occipital | normal | captured - lossy-wall skull mode (named IN) | 1.77e-01 |
| occipital | tangential | captured set (19 poles) | 1.20e-11 |
| occipital | tangential | captured + Zenneck-type pole (named OUT) | 1.29e+00 |
| occipital | tangential | captured - lossy-wall skull mode (named IN) | 2.34e-02 |
| temple | normal | captured set (18 poles) | 5.19e-10 |
| temple | normal | captured + Zenneck-type pole (named OUT) | 1.29e+00 |
| temple | normal | captured + brain-side leaky wave (named OUT) | 1.36e+00 |
| temple | normal | captured - temporalis mode (named IN) | 3.75e-02 |
| temple | tangential | captured set (18 poles) | 3.09e-11 |
| temple | tangential | captured + Zenneck-type pole (named OUT) | 1.43e+00 |
| temple | tangential | captured + brain-side leaky wave (named OUT) | 7.21e-02 |
| temple | tangential | captured - temporalis mode (named IN) | 1.91e-03 |

Worst baseline (full captured-pole-set) mismatch over the 3 stacks x 2 orientations at z = 6 cm: **5.19e-10** (&le; 1e-8 required); every add/remove variant above breaks closure by orders of magnitude, confirming the captured set is exactly right, not merely sufficient. (Air sources -- 1 mm above skin -- have a slower-decaying spectral tail; closing to this precision for them would need n_max beyond what the unscaled transfer matrices evaluate without overflowing, so the table below reports their field shares at the quadrature-limited ~1e-3 level instead.)

## Excitation table at 2.44 GHz

Observer: H<sub>y</sub> at the skin surface (x_o = 0+), as a function of lateral distance z. Rows are the same named poles the catalog highlights for each stack; 'in the surface field?' is the check d capture test -- only a captured pole's residue corresponds to anything a real source actually launches, so only captured rows carry numbers.

### Upper arm (skin 1.5 mm | fat 10 mm | muscle)

Legend: **sh** = share of the exact H<sub>y</sub>(z) at the skin surface carried by this mode's own term alone, 20log<sub>10</sub>(|modal term| / |exact field|), at z = 2 cm / 6 cm; **res** = |residue| in dB relative to this mode's own strongest source/orientation below (the 3-D point-dipole factor &times;sqrt(n<sub>m</sub>) is the same for every source of one mode, so it cancels here). Both are shown only for captured (physically present) poles -- see 'in the surface field?'.

| pole | sheet | n_eff | loss (dB/cm) | in the surface field? | 1 mm above skin (air), normal | 1 mm above skin (air), tangential | mid-fat, normal | mid-fat, tangential |
|---|---|---|---|---|---|---|---|---|
| Zenneck-type pole | I/P | 0.985 +0.063j | -0.28 | no (grows along z (Im n > 0)) | -- | -- | -- | -- |
| lossy-wall fat mode | P/P | 1.832 -0.667j | 2.96 | yes (captured) | sh -17.6/-19.5; res -3.6 | sh -5.3/-13.6; res -4.5 | sh -1.3/-6.4; res +0.0 | sh -9.7/-15.4; res -9.6 |
| its leaky twin | I/P | 1.797 -0.866j | 3.85 | no (improper, slow wave (Re n > 1)) | -- | -- | -- | -- |

### Occipital (back of the head, 8.4 mm bone)

Legend: **sh** = share of the exact H<sub>y</sub>(z) at the skin surface carried by this mode's own term alone, 20log<sub>10</sub>(|modal term| / |exact field|), at z = 2 cm / 6 cm; **res** = |residue| in dB relative to this mode's own strongest source/orientation below (the 3-D point-dipole factor &times;sqrt(n<sub>m</sub>) is the same for every source of one mode, so it cancels here). Both are shown only for captured (physically present) poles -- see 'in the surface field?'.

| pole | sheet | n_eff | loss (dB/cm) | in the surface field? | 1 mm above skin (air), normal | 1 mm above skin (air), tangential | mid-diploe, normal | mid-diploe, tangential |
|---|---|---|---|---|---|---|---|---|
| Zenneck-type pole | I/P | 0.989 +0.015j | -0.07 | no (grows along z (Im n > 0)) | -- | -- | -- | -- |
| lossy-wall skull mode | P/P | 2.949 -1.246j | 5.53 | yes (captured) | sh -34.3/-49.1; res -3.2 | sh -14.9/-33.3; res -3.5 | sh -6.1/-15.0; res +0.0 | sh -17.9/-32.6; res -3.0 |

### Temple (1.5 mm squama)

Legend: **sh** = share of the exact H<sub>y</sub>(z) at the skin surface carried by this mode's own term alone, 20log<sub>10</sub>(|modal term| / |exact field|), at z = 2 cm / 6 cm; **res** = |residue| in dB relative to this mode's own strongest source/orientation below (the 3-D point-dipole factor &times;sqrt(n<sub>m</sub>) is the same for every source of one mode, so it cancels here). Both are shown only for captured (physically present) poles -- see 'in the surface field?'.

| pole | sheet | n_eff | loss (dB/cm) | in the surface field? | 1 mm above skin (air), normal | 1 mm above skin (air), tangential | mid-temporalis, normal | mid-temporalis, tangential |
|---|---|---|---|---|---|---|---|---|
| Zenneck-type pole | P/P | 0.973 -0.023j | 0.10 | no (proper, hidden under the air branch cut) | -- | -- | -- | -- |
| temporalis mode | P/I | 5.151 -1.331j | 5.91 | yes (captured) | sh -55.3/-72.2; res -9.4 | sh -39.7/-59.0; res -9.5 | sh -12.1/-28.5; res +0.0 | sh -36.9/-54.4; res -9.2 |
| brain-side leaky wave | P/I | 4.808 -0.292j | 1.30 | no (above the gray matter branch point) | -- | -- | -- | -- |

## Interpretation

- **Of the 8 named poles across the three stacks, only the 3 directly-named tissue modes are captured -- the fat mode, the skull mode, the temporalis mode.** Every 'Zenneck-type' pole and every secondary leaky partner (the fat mode's I/P twin, the temple's brain-side P/I leaky wave) fails the capture test, each for a reason check d confirms independently: the arm's and occiput's Zenneck-type poles simply grow along z (Im n > 0, the giveaway of an improper pole that was never reachable); the fat mode's twin is an improper slow wave (Re n > 1); the temple's brain-side leaky wave sits above the gray-matter branch point; and the temple's own Zenneck-type pole -- a textbook bound (P/P), positive-loss mode -- is simply **hidden under the air branch cut**, the classic Collin (2004) result that a lossy half-space's Zenneck pole is not on the sheet a real source's field ever reaches. The sheet label does not decide it: the temporalis mode is improper (P/I) yet captured, and the temple's Zenneck pole is proper yet hidden.
- **Even the one captured mode per stack is usually a minority contributor to the real surface field, and its share shrinks with distance.** At its very best -- source inside the mode's own layer, normal orientation, z = 2 cm -- the named mode's own term carries about 86% of the field amplitude for the arm (sh -1.3 dB) but only about 50% for the occiput (-6.1 dB) and 25% for the temple (-12.1 dB); by z = 6 cm every stack's share has dropped further, to -6.4 dB (arm), -15.0 dB (occiput), -28.5 dB (temple). The rest of the field is the sum of the many other captured poles (14-19 of them per stack, check d) plus the two branch-cut integrals, and that remainder decays only algebraically with z while any one pole's term decays exponentially -- so no matter how low-loss the named mode is, distance eventually hands the field to the branch-cut (lateral/"Norton-wave") term instead, exactly the competition Collin (2004) describes.
- **Put the source inside the guiding layer.** For every captured mode the strongest residue comes from a current normal to the layers at mid-layer (the 0 dB reference). Turning it tangential costs 9.6 dB (arm), 3.0 dB (occiput) and 9.2 dB (temple). Moving it to 1 mm above the skin costs only 3-10 dB of residue, but the mode's share of the surface field collapses, because the same source now drives the branch-cut waves far harder. At 2 cm the share falls to -17.6 dB (arm), -34.3 dB (occiput) and -55.3 dB (temple) for a normal current, and to -5.3, -14.9 and -39.7 dB for a tangential one.

## Convention caveats and surprises

- **Source model.** 'Normal' and 'tangential' are unit-amplitude electric line currents (infinite filaments along y, i.e. 2-D line sources), not the core's own line/point-dipole sources (`linesource.py`'s M_z magnetic line current, or the VED/HED of `sommerfeld.py` / `cylinder3d.py`). Check c above shows the precise relationship to `residue_R` for a tangential source in air; no such clean relationship was derived for the normal orientation or for a magnetic-current source, so only *relative* comparisons (dB within a row, across source positions and orientations) should be read quantitatively, not raw residue magnitudes across source types.
- **Reciprocity only holds in its eps-weighted form, and only for the normal/shunt-type source** (check b): G(x_o; x_s) and G(x_s; x_o) differ by exactly eps(x_s)/eps(x_o) for a normal source; the tangential/series-type source's natural reciprocal partner is a *different* field component (H-due-to-J is, in general, reciprocal to E-due-to-M, not to itself) and was numerically confirmed *not* to satisfy either the plain or eps-weighted swap.
- **A residue is not a presence.** Whether a catalog mode appears in a real field is decided by whether its pole is captured when the exact field integral is deformed onto vertical branch cuts (Tamir & Oliner 1963; Method point 5, check d), not by the size of its residue and not by its sheet label. Five of the eight named poles here have finite, sometimes large residues and still never appear. Check d closes the exact spectral integral at z = 6 cm for every buried source and both orientations using only the captured poles (14-19 per stack) plus the two branch-cut integrals, to a worst relative mismatch of 5.19e-10; adding any named OUT pole, or removing the named IN pole, breaks that closure by 7 to 11 orders of magnitude.
- **Air-positioned sources are the hard case for this kind of closure test, for a mundane numerical reason.** Their spectral integrand decays with n much more slowly than a buried source's, so matching the real-axis integral to 1e-8-level precision would need an n_max well beyond where this module's unscaled transfer matrices still evaluate without overflowing (the catalog's own `find_modes` sidesteps this with mpmath's arbitrary precision; this residue code does not). Check d is therefore run on the buried sources only; the excitation table's air-source columns are still the exact closed-form residues (not approximations), but their 'sh' (share-of-exact-field) entries carry the coarser ~1e-3 quadrature precision noted under check d, not the 1e-8 the buried-source entries have.

