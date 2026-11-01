# Appendix: mode catalog

Planar (1D) layered-tissue stacks, TM unless noted, 2.44 GHz (Bluetooth) unless noted. Every mode is a zero of the multilayer dispersion function F, which is the same thing as a pole of the stack's spectral response to a source. Modes are found with the argument-principle solver in [tissue-slab-modes](../README.md) and classified by Riemann sheet: **P/P** = proper on both sides (Re γ > 0), **I/P** = grows into the air (leaky to air), **P/I** = grows into the body side (leaky into the body), **I/I** = improper on both sides. Tissue permittivities use the 4-pole Cole-Cole model of Gabriel et al. (1996). Regenerate the figures with `python scripts/make_figures.py [--eps]`.

The post gives the physical argument and representative results. This appendix holds the full derivations, stack specifications, frequency and threshold details, source-capture rule, three-dimensional case results, and prior-art inventory. The mode-family overview is [Figure 2 in the post](README.md#4-modes-in-the-body).

---

## The equations we solve

The layers are stacked along x, the wave travels along z as exp(j(ωt − k₀nz)), lossy media have Im ε < 0, and lengths are scaled by k₀ (ξ = k₀x). In a layer of permittivity εⱼ the transverse field is A exp(γⱼξ) + B exp(−γⱼξ) with γⱼ = √(n² − εⱼ). The state vector u = [H_y, (1/ε) dH_y/dξ] (TM) or u = [E_y, dE_y/dξ] (TE) is continuous at every interface, and a layer of thickness dⱼ carries it from its bottom to its top by


```math
M_j=\begin{pmatrix}\cosh\phi_j & p_j\,\sinh\phi_j/\gamma_j\\ \gamma_j\,\sinh\phi_j/p_j & \cosh\phi_j\end{pmatrix},\qquad \phi_j=\gamma_j k_0 d_j,\quad p_j=\varepsilon_j\ (\mathrm{TM})\ \text{or}\ 1\ (\mathrm{TE}).
```

The complex effective index is n = (β − jα)/k₀: Re n sets the phase speed c/Re n, while Im n < 0 gives α = −k₀ Im n and a power loss of 8.686α = −8.686k₀ Im n in dB per unit length.

In the bottom half-space the field is exp(+γ_b ξ), so u starts as (1, γ_b/p_b); above the top interface it is exp(−γ_t ξ). Carrying u up through all N layers and matching it to the top half-space gives the dispersion function

```math
F(n)=\frac{\gamma_t}{p_t}\,u_1+u_2=0,\qquad \begin{pmatrix}u_1\\u_2\end{pmatrix}=M_N\cdots M_1\begin{pmatrix}1\\ \gamma_b/p_b\end{pmatrix}.
```

Every entry of Mⱼ is an even function of γⱼ, so the interior layers add no branch points; only γ_t and γ_b are double-valued, and the sign choices Re γ ≷ 0 on each side give the four sheets P/P, I/P, P/I and I/I. Because γ_t² − γ_b² = ε_b − ε_t = C is constant, the substitution γ_t = (√C/2)(t + 1/t), γ_b = (√C/2)(t − 1/t) makes F single-valued in t. The solver finds every zero inside its search window with the argument principle on the plane z = ln t. The four-sheeted relation follows Smith, Houde-Walter & Forbes, *IEEE J. Quantum Electron.* 28(6):1520–1526 (1992), [doi:10.1109/3.135305](https://doi.org/10.1109/3.135305); the single-valued unfolding follows Smith, Forbes & Houde-Walter, *IEEE J. Quantum Electron.* 29(4):1031–1034 (1993), [doi:10.1109/3.214486](https://doi.org/10.1109/3.214486); the argument-principle search follows Anemogiannis & Glytsis, *J. Lightwave Technol.* 10(10):1344–1351 (1992), [doi:10.1109/50.166774](https://doi.org/10.1109/50.166774). The implementation builds on Şükrü Ekin Kocabaş's argument-principle [Mathematica notebook](https://kocabas.gitlab.io/publications/code/ModalShapes_Plasmon.nb) for metal-insulator-metal guides, Kocabaş, Veronis, Miller & Fan, *Phys. Rev. B* 79:035120 (2009), [doi:10.1103/PhysRevB.79.035120](https://doi.org/10.1103/PhysRevB.79.035120).

### Two useful special cases

With no finite layer between half-spaces ε_t and ε_b, the TM condition becomes

```math
\frac{\gamma_t}{\varepsilon_t}+\frac{\gamma_b}{\varepsilon_b}=0\quad\Longrightarrow\quad n^2=\frac{\varepsilon_t\,\varepsilon_b}{\varepsilon_t+\varepsilon_b}.
```

This is the surface-plasmon and Zenneck formula used in the post. For a symmetric slab, an even TM field is cos(κk₀x) in a core ε_c of thickness d and cos(κk₀d/2) exp(−γ_w k₀(|x| − d/2)) in the walls ε_w. Continuity at either wall gives

$$\frac{\kappa}{\varepsilon_c}\tan\left(\frac{\kappa k_0 d}{2}\right)=\frac{\gamma_w}{\varepsilon_w},\qquad \kappa=\sqrt{\varepsilon_c-n^2},\quad \gamma_w=\sqrt{n^2-\varepsilon_w}.$$

Figure 1 in the post plots this exact solution, with rays at arctan(Re κ/Re n) from the guide axis in the core and arctan(Im γ_w/Re n) in the wall: 71° in panel b, or 19° from the wall normal. Chew, *Waves and Fields in Inhomogeneous Media* (1990; IEEE Press reprint), chapter 2, [doi:10.1109/9780470547052](https://doi.org/10.1109/9780470547052), derives the general multilayer form; the implementation is in `tissue_slab_modes/dispersion.py`.

## The spectral integral and capture rule

A small source's exact field is an integral over all real n of the stack response,

```math
H_y(x,z)\propto\int\frac{N(x,n)}{F(n)}\,\exp(-jk_0nz)\,dn.
```

Each mode is a pole because F is in the denominator. Deforming the path around the two branch cuts, which start at the branch points n = √ε of air and of the deepest tissue, splits the field into residues of the swept poles plus two continuous-spectrum terms: the space and Norton ground waves in air, and a lateral wave in the body. Tamir & Oliner (1963) gave the capture rule. In each half-space a pole must lie on the sheet reached by the deformed contour and decay along z (Im n < 0); more precisely, an improper pole counts only if it lies between that half-space's Sommerfeld cut and vertical cut, while a proper pole in that region is hidden.

[`results/residues.md`](results/residues.md) supplies the numerical check. For buried sources in all three stacks, the swept poles—14–19 per stack—plus the two cut integrals reproduce the exact field at z = 6 cm to 5.2 × 10⁻¹⁰. Adding any named uncaptured pole, or removing the named guided mode, breaks the match by seven to eleven orders of magnitude.

### Figure 8: the full pole map

![Left: a phase-colored map of the dispersion function with color wheels at each zero and black curves separating four regions. Right: the same zeros as effective index, with the fat mode near 1.83 − 0.67j and its twin just below it.](figures/02_pole_map.png)

**Figure 8.** Zeros of the upper arm's TM dispersion function for skin 1.5 mm | fat 10 mm | muscle at 2.44 GHz. Left: F on the plane z = ln t, where the uniformizing variable t makes F single-valued and unfolds the four Riemann sheets onto one rectangle. Black curves separate the sheets, color wheels mark zeros, and marker shapes give their sheets. Right: the same zeros as effective index n.

## Tissues, stacks, and scale

The complete 2.44 GHz material values used here are:

| tissue | ε at 2.44 GHz | index Re √ε | wavelength (cm) | plane-wave loss (dB/cm) |
|---|---:|---:|---:|---:|
| air | 1 | 1.0 | 12.3 | 0 |
| fat | 5.3 − 0.8j | 2.3 | 5.3 | 0.74 |
| cortical bone | 11.4 − 2.9j | 3.4 | 3.6 | 1.89 |
| cancellous bone | 18.6 − 5.9j | 4.4 | 2.8 | 3.01 |
| skin | 38.0 − 10.7j | 6.2 | 2.0 | 3.83 |
| dura | 42.0 − 12.2j | 6.6 | 1.9 | 4.15 |
| tendon | 43.1 − 12.4j | 6.6 | 1.9 | 4.14 |
| muscle | 52.7 − 12.8j | 7.3 | 1.7 | 3.87 |
| gray matter | 48.9 − 13.3j | 7.1 | 1.7 | 4.17 |
| CSF | 66.3 − 25.4j | 8.3 | 1.5 | 6.81 |

The five standard flat stacks run from air down into the deepest tissue half-space:

- **Upper arm:** skin 1.5 mm | fat 10 mm | muscle.
- **Abdomen:** skin 2.0 mm | fat 20 mm | muscle.
- **Volar wrist:** skin 1.3 mm | fat 3.0 mm | tendon 3.0 mm | muscle.
- **Back of the head (occiput):** skin 2.0 mm | fat 3.0 mm | 8.4 mm of three-layer skull bone (1.5 mm outer cortical table | 5.6 mm cancellous diploë | 1.3 mm inner cortical table) | dura 0.5 mm | CSF 2.0 mm | gray matter.
- **Temple:** skin 1.5 mm | fat 2.0 mm | temporalis muscle 8.1 mm (Pirri et al. 2021) | cortical temporal squama 1.5 mm (measured 1.52 ± 0.50 mm in 72 adults, Bhushal et al. 2025) | dura 0.5 mm | CSF 2.0 mm | gray matter.

The assumed limb stack is air | skin 1.5 mm | fat 10 mm | muscle 25 mm | cortical bone 6 mm | yellow marrow.

In the first four stacks, the low-index fat or bone layer sits between wet, high-index tissues, an antiguide by the fiber rule. At the temple, high-index temporalis muscle is itself the core.

At 2.44 GHz, the 0.5–25 mm tissue layers are a few hundredths of a wavelength to about one and a half wavelengths across: thin-film territory, where a ray picture can illustrate a mode but only the wave equations compute it.

Tubular body parts are the nearest analogs of a closed metal pipe. Treating the wet tissue around a core as a perfectly conducting cylinder gives a lowest-mode TE₁₁ cutoff of 2.0–3.0 GHz for a femur shaft, 2.5–3.8 GHz for a tibia, 7.0–13.5 GHz for the trachea (adult male coronal diameter 13–25 mm, Lawrence et al. 2014), and 22–29 GHz for the ear canal. In these first-order screens with assumed bone and canal geometries, the femur shaft sits at cutoff near 2.44 GHz, while the trachea and ear canal are too narrow to carry a pipe mode there. A flat layer has no cutoff for TM₀, although thin fat turns that mode leaky.

Index guiding, lossy reflecting walls, and branch-cut waves have familiar relatives. In the eye, Müller cells span the retina with a higher refractive index than the surrounding tissue and pipe light to the photoreceptors (Franze et al. 2007). Very-low-frequency radio at 3–30 kHz travels between the ground and lower ionosphere (Wait 1957); both imperfect walls reflect more than they absorb. Medium-wave AM over flat ground includes the Norton ground wave (Norton 1936, 1937), a branch-cut wave rather than a mode.

## Travel distance

The full distance comparison behind the post's shorter table is:

| mode | loss (dB/cm) | wavelengths per e-fold | bulk loss of core; walls (dB/cm) |
|---|---:|---:|---|
| limb muscle, truly bound (TE) | 4.0 | 0.63 | muscle 3.9 (0.67 wavelengths per e-fold) |
| CSF, truly bound (TE), 7.99 GHz | 22.9 | 0.34 | CSF 24.7 (0.36) |
| fat, upper arm, "bound" | 3.0 | 0.22 | fat 0.74; skin and muscle 3.8–3.9 |
| fat, abdomen (20 mm), "bound" | 1.7 | 0.44 | fat 0.74; skin and muscle 3.8–3.9 |
| skull, occiput, "bound" | 5.5 | 0.19 | bone 1.9–3.0; fat 0.74 above; dura, CSF and gray matter 4.2–6.8 below |
| temporalis over 1.5 mm of bone, leaky | 5.9 | 0.31 | muscle 3.9; fat 0.74 above, bone 1.9 below |

The ratio β/(2α) = Re n/|2 Im n| is the radians of phase per e-fold, or 4.34 dB, of power decay. A mode travels (β/2α)/2π guide wavelengths over that interval. Across the captured tissue modes, the range is 0.04 on the wrist to 0.63 in the limb. At 2.44 GHz an unguided plane wave travels only 0.4–0.7 wavelengths per e-fold in wet tissue; fat reaches 1.1.

---

## 1. Zenneck-type surface wave

**What it is.** A TM wave "bound" to the skin surface and carried almost entirely in the air: 99 % of its power is outside the body. Over a plain skin half-space (n = 0.988 − 0.003j, 0.01 dB/cm) the field falls to 1/e at about 93 cm into the air (7.6 free-space wavelengths); over the layered temple stack (n = 0.973 − 0.023j, 0.10 dB/cm) it reaches only about 22 cm (1.8 wavelengths), still with 99 % of its power in the air. Re n < 1, so its phase speed along the surface exceeds c because its wavefront tilts into the tissue.

**Why it's cool.** It is the same pole formula as a surface plasmon: n = √(ε/(1+ε)) for both. The sign of Re ε decides which one you get. Gold at optical frequencies (Re ε < −1) gives a slow, truly bound plasmon pressed against the metal at visible wavelengths; skin at 2.4 GHz (Re ε > 0, large |ε|) gives a fast, loosely "bound" Zenneck wave. Over a plain skin half-space the field reaches 7.6 free-space wavelengths, comparable to a gold plasmon at 10 µm (11.5). The Zenneck wave has the lowest loss of any mode here, but a small source on or near skin does not launch it; a large aperture shaped to its profile launches a Zenneck-like field only near the aperture (Hill & Wait 1980). On the temple stack the pole is proper (P/P) but lies under the air's Sommerfeld cut and is excluded from the deformed source contour.

**Prior art.** Zenneck vs plasmon: Michalski & Mosig, *IEEE Antennas Propag. Mag.* 64(6):70–80 (2022). Patented launcher designs (their physics is disputed): CPG Technologies US 10,062,944 B2 and US 9,923,385 B2; experiment: Oruganti et al., *Sci. Rep.* 10:925 (2020) (over metal); theory: Hill & Wait 1980 (10 MHz over ground). Ear-to-ear hearing aids deliberately polarize normal to the head to couple into the around-head path: Starkey US 10,051,388 B2, GN ReSound EP 2871861. Tesla's patent for transmitting power through the ground and air, US 787,412 (1905, filed 1900), is a precursor that CPG's patents cite, though it describes stationary waves in the earth rather than a surface wave.


## 2. The fat mode: lossy-wall loss guiding (the headline)

**What it is.** Fat (ε′ ≈ 5.3) is the *low*-index layer between skin (38) and muscle (53). That is an antiguide: total internal reflection cannot hold a wave in it. Instead, the large permittivity contrast with the wet skin and muscle walls reflects part of a grazing TM wave at each bounce, somewhat like a leaky parallel-plate waveguide—on the upper-arm stack it refracts into the muscle wall at 14.5° from the wall normal at each bounce—while the walls' **loss** absorbs the wave that refracts into them, moving the pole across the branch cut from leaky (improper) to mathematically "bound" (proper, P/P), with 82 % of its power in the fat and 3.0 dB/cm of loss on the 10 mm upper-arm stack. Following the Riemann-sheet classification, we call this **lossy-wall loss guiding** (or **"bound" through wall loss**). Nagel, Blair & Scarpulla (2011) used "loss-guided" for the reverse geometry, a lossy core between lossless walls; the tissue case puts the high permittivity and the loss in the walls.

**Why it's cool.**
- **It is the Riemann-sheet mirror image of Siegman's gain guiding, exact for TE and nearly so for TM.** Take a symmetric 8 mm slab of *lossless* fat (ε = 5.3) in muscle (Figure 9). With lossless walls the mode is leaky. Scale the muscle loss by s: the pole crosses the branch cut at s = 0.26 and real muscle (s = 1) holds it "bound" (P/P) at 3.0 dB/cm and n = 2.458 − 0.672j. Adding *gain* to the core instead, as in Siegman's fibers, crosses at s = 0.27. For TE the two paths coincide exactly because only Siegman's complex contrast enters, V² = (k₀d/2)²(ε_core − ε_wall) = ΔN + jG for a core of thickness d, where ΔN < 0 marks an antiguide and G > 0 means the walls are lossier than the core or the core has gain (TM differs slightly because ε also enters the boundary condition). TE needs s = 2.40 at 8 mm, though: its field must nearly vanish at each high-permittivity wall, as between metal plates, so 8 mm is too thin to reach "bound" at real tissue loss—the crossing happens only above about 13 mm of fat (with fat's own loss included, s_th = 2.46, 1.21, 1.03, 0.89 at d = 8, 12, 13, 14 mm). The abdomen's 20 mm fat layer does carry a "bound" TE mode (2.1 dB/cm, 52 % of its power in the fat and 31 % in the air), but skin and air hold it from above, not a second wet wall; the symmetric 20 mm slab's TE₀ is "bound" yet overdamped (8.4 dB/cm).
- **High permittivity does most of the holding; lossier walls lower the loss.** Restoring fat's own 0.74 dB/cm loss (ε = 5.3 − 0.8j) in the symmetric 8 mm slab (Figure 4 in the post), even lossless muscle walls (s = 0) already hold the leaky mode to 3.91 dB/cm through partial grazing reflection. Turning on real muscle loss (s = 1) crosses the branch cut at s = 0.32 (against 0.26 for Figure 9's lossless fat) and makes the mode proper at n = 2.420 − 0.848j and 3.77 dB/cm, refracting into the muscle wall at 19.3° from its normal at each bounce. Raising the wall loss further lowers the modal loss: 3.91 → 3.77 → 3.39 → 2.45 → 1.71 dB/cm for s = 0, 1, 3, 10, 30, heading toward the perfect-conductor limit of 0.74 dB/cm (Figure 4 in the post, right). This coal-seam effect was described in 1970s–80s mining radio: a more conductive wall reflects more and keeps the field out, as a metal plate would.
- **Thinner fat behaves like a metal gap.** Re n rises from 2.29 (25 mm) to 4.42 (1 mm) as the gap closes, but the loss rises with it too, to 12.8 dB/cm (β/2α = 0.77): close, resistive walls slow the wave and damp it together—a lossy parallel-plate line, not the reactive slowing of a true metal-insulator-metal gap plasmon, since tissue never reaches Re ε < 0. Below about 4 mm the mode is heavily damped, and below about 2 mm it turns leaky (Figure 4 in the post, left and middle). The real upper-arm stack shares only the loss trend: from 10 to 8 mm of fat the loss rises from 3.0 to 3.6 dB/cm, but Re n falls (1.83 → 1.70) and the refraction angle in the muscle moves toward the wall normal (14.5° → 13.4°).
- **There is a best frequency.** In the abdomen (20 mm fat) the loss has a minimum of about 1.6 dB/cm near 1.7 GHz and stays below 1.75 dB/cm from 1.5 to 2.5 GHz. At 2.44 GHz, the abdomen loses 1.72 dB/cm (β/2α = 2.74), the upper arm 3.0 dB/cm (β/2α = 1.37), and the volar wrist 11.5 dB/cm (β/2α = 0.25). At low frequency the roots survive but are overdamped, β/(2α) < 1, as the faded curves in Figure 10 show. The modes become wave-like above about 1.2 GHz in the abdomen, 2.0 GHz in the upper arm, and 5.0 GHz at the wrist, each 0.5–0.6 GHz below its loss minimum.

**Prior art.** The subcutaneous fat channel itself is modeled and measured as a microwave dielectric waveguide by the Uppsala Fat-IBC group: Asan et al., *Healthc. Technol. Lett.* 4:115 (2017) and *Sensors* 18:2752 (2018); Rangaiah et al., *IEEE TBME* 71(1):89–96 (2024); Engstrand et al., EuMC 2025; Engstrand, PhD thesis, Uppsala (2026). Sensing patents and applications: Probingon AB US 12,357,187 B2; Meaney, Augustine et al. (Uppsala) EP 4 044 914 A1 (withdrawn); Samsung US 2016/0143558 A1 (UWB fat-layer reflectometry). Coal-seam modes in mining geophysics: Emslie & Lagace, *Radio Sci.* 11:253 (1976); Hill, *J. Res. NBS* 89:385 (1984); Wait, *Radio Sci.* 11:263 (1976). Gain guiding: Siegman, *JOSA A* 20:1617 (2003), *JOSA B* 24:1677 (2007); US 7,668,211 B1. Loss-gain-loss non-Hermitian guiding: Savoia, Castaldi & Galdi, *PRA* 94:043838 (2016). We take the term "loss-guided" from Nagel, Blair & Scarpulla, *Opt. Express* 19:20159–20171 (2011), who use it, following Siegman (2003), for the reverse geometry (lossy core, lossless walls); we credit and distinguish it. While Fat-IBC models fat as a dielectric waveguide channel, we found no prior work giving the Riemann-sheet synthesis that links fat and skull modes to Siegman's gain-guided antiguides, coal-seam modes, and Nagel's loss-guided thin films.

### Figure 9: loss versus gain guiding

![Two phase-colored maps with two paths each, a solid orange path labeled add loss to the walls and a dashed blue path labeled add gain to the core, both crossing a horizontal black line that separates leaky from proper.](figures/03_loss_vs_gain_guiding.png)

**Figure 9.** Pole paths for the lossless-fat slab, TM₀ (left) and TE₀ (right), on the uniformized z = ln γ_clad plane. The horizontal black line separates leaky from proper. Solid: wall loss; dashed: core gain; both are scaled by s relative to real muscle loss. Squares mark the lossless antiguide, circles the branch-cut crossing, and stars real muscle loss.

Figure 4 in the post gives the companion sweep with fat's real loss restored, including the thickness dependence and coal-seam limit.

### Figure 10: frequency and overdamping

![Left: modal loss against frequency from 0.3 to 10 GHz for abdomen, upper arm, wrist, and the skull mode and its fat-dominated second branch, faded where overdamped. Right: the ratio of phase to decay against frequency, with an overdamped zone below 1.](figures/06_frequency.png)

**Figure 10.** Modal loss (left) and β/(2α) (right) against frequency for the TM fat modes of the abdomen, upper arm, and volar wrist, and for the occipital TM skull mode and its second, fat-dominated branch. Curves are faded where overdamped and dotted where leaky; vertical dotted lines mark medical and wireless bands.

### Figure 11: transverse mode profiles

The transverse profile of every mode in [`results/modes.md`](results/modes.md), with one PNG per mode in `figures/modes/<id>.png`, forms **Figure 11**.

## 3. The leaky twin: a zero that is not a wave

**What it is.** A second zero next to the fat mode, on the I/P sheet (n = 1.797 − 0.866j, 3.85 dB/cm): the same fat-guided profile, but it grows into the air.

**Why it's cool.** It makes the four Riemann sheets concrete. On the uniformized plane (Figure 8) every color wheel is a zero, and the black curves separating the sheets pass right between the "bound" mode and its twin. It is also the cleanest example that a zero is not automatically a wave. The twin is slower than light in air (Re n = 1.80 > 1), and a slow wave cannot leak into air, so no source excites it. The exact field of a source in the fat matches the captured poles plus the branch-cut waves, with no twin term. The sheet label alone cannot tell you this; where the pole sits relative to the branch cuts can.

## 4. The skull mode: lossy-wall loss guiding (back of the head / occiput)

**What it is.** The same mechanism in bone. At the back of the head (occiput), the skull is 8.4 mm thick: 1.5 mm outer cortical table, 5.6 mm cancellous diploë, and 1.3 mm inner cortical table. Bone (ε′ ≈ 11–19), under 3 mm of fat, forms a low-index layer between the scalp and dura (42), CSF (66), and brain (49). The mode is "bound" (P/P), with n = 2.949 − 1.246j; 46 % of its power is in bone, and it loses 5.5 dB/cm.

**Why it's cool.** It is a bone-guided radio mode, the electromagnetic counterpart of what bone-anchored hearing aids do acoustically. The field peaks at the inner table, against the dura/CSF, refracting into the gray matter at 24.5° from the normal at each bounce. β/(2α) ≈ 1.2 at 2.44 GHz, so it is only just wave-like. Its loss is lowest (about 5.1 dB/cm) near 1.8 GHz, just inside its overdamped range; it becomes wave-like only above about 2 GHz. On this branch the loss reaches 11.4 dB/cm at 5.8 GHz. A second "bound" mode, overdamped at Bluetooth, loses less than the first branch above 4.0 GHz (8.2 dB/cm at the crossing, 7.6 dB/cm at 5.8 GHz; dashed in Figure 10) and, still overdamped at its own roughly 4.0 GHz crossing (β/2α = 0.83), becomes wave-like in its own right only above about 4.3 GHz—but it is carried mostly by the fat over the bone (56 % fat, 37 % bone at 5.8 GHz), not by bone like the first branch (63–88 % bone from 4.0 to 7.99 GHz).

**Prior art.** None found for an EM skull-guided mode as such. Data links through the skull bone between hearing devices use vibration instead: Cochlear US 9,191,759 B2 (bone-conduction data link). RF links into the brain through the skull are being studied directly (Särestöniemi et al. 2022); a related but distinct application delivers wireless neuromodulation through an implanted microwave split-ring resonator rather than a data link (Boston University, US 2023/0140692 A1).

**3D check.** Earlier runs of cases A and B used lossless tissue and their numbers were wrong. The corrected, lossy-tissue results—the skull mode launched alone at the back of the head toward the right ear (Figure 6 in the post)—are in the post's section 5; the raw numbers are in `results/case_a_mode/case_a_mode_res2.json` and `results/case_a/case_a_res2.json`. Right at Ernie's posterior midline launch ray, the 9.2 mm three-layer bone has much thicker cortical tables and a thinner diploë than the generic stack, which, together with Ernie's thicker skin and thinner fat, lowers the 1D loss to 4.4 dB/cm (n = 2.917 − 0.984j) against 5.5 dB/cm for the generic occipital stack; it also sits over the venous sinus (`blood`) rather than CSF and brain—replacing the sinus with a generic dura/CSF/gray-matter base under Ernie's own bone gives 4.2 dB/cm, so the sinus itself adds only about 0.2 dB/cm. In 3D the mode survives along the curving skull, losing 3.2–3.7 dB/cm after beam-spreading correction (16–28 % below the 4.4 dB/cm 1D prediction). That correction assumes a Gaussian beam with a 3 cm Rayleigh range and is worth 0.62/0.68/0.44 dB/cm over 1–2.5/2.5–6/6–12 cm. Uncorrected, the mid-bone field falls 4.2 dB/cm over 1–6 cm (21.1 dB total), matching the 1D rate, so over 1–6 cm the gap mostly reflects the assumed spreading; over 6–12 cm the field decays more slowly than 1D even uncorrected (3.6 dB/cm). A point-dipole source at the same spot, without the mode-matched launch sheet (`case_a`), cross-checks at 2.96–4.0 dB/cm after removing an assumed cylindrical spreading (worth 0.5–2.6 dB/cm). The local mid-bone phase index is 2.5–2.6 over 1–6 cm (12–15 % below the 1D value), rising to 2.95 (about 1 % above it) over 6–12 cm, where the bone thins from 6.6 to 3.5 mm, although in 1D thinner bone lowers the index (about 2.45 at 3.5 mm). "Bound" (P/P) here means the interior tissue absorbs the refracted leakage, not that the bone seals the power in.

## 5. Temporalis mode: "bound" to leaky as the temple thins

**What it is.** At the temple |H_y| peaks not in the bone but in the temporalis muscle (8.1 mm, index 7.3), which is the high-index layer between fat (2.3) and bone (3.4), so this is ordinary index guiding with a lossy core, though with 6 mm of bone about 36 % of the power still flows in the bone (40 % in the temporalis). The thin squama is only a low-index barrier between the muscle and the CSF/brain. Because gray matter's index (7.1) and CSF's (8.3) are above the mode's (~4.5–5.2), the wave tunnels through the bone into the brain, more strongly the thinner the bone: frustrated total internal reflection.

**Why it's cool.** The anatomy sits right on the threshold. The mode is "bound" (P/P) down to 1.64 mm of bone, and the measured adult squama is 1.52 ± 0.50 mm (Bhushal et al. 2025); loss and index change smoothly through that thickness, and nothing physically switches on there—it is only where the flat-layer sheet label flips. At 6 mm it is "bound" (4.49 − 0.97j, 4.3 dB/cm); at 1.5 mm it leaks into the brain (5.15 − 1.33j, 5.9 dB/cm; Figure 5 in the post). The same stack also holds a second P/I root at 4.81 − 0.29j (1.3 dB/cm) whose field doubles within 14 mm of brain. Its low dB/cm is irrelevant: the root sits above the brain's branch point, outside the region the field integral sweeps, so no source excites it. The temporalis mode, by contrast, is a genuine captured leaky wave, though only just: at its Im n the capture region begins at Re n = 6.633/1.331 = 4.984, and the mode sits at 5.151, a margin of 0.17.

**3D check.** At `case_c_left`'s launch point the bone reads 2.05 mm along the bone normal (the 1D prep stack) but 1.9 mm along the launch ray, both off the 0.5 mm voxel grid; `case_c_right`'s 3.0 mm matches exactly. The left margin over the "bound"/leaky threshold (1.70–1.75 mm) is 0.30–0.35 mm on the prep stack and about 0.15–0.20 mm along the ray, both below the 0.5 mm voxel, so the 3D grid cannot tell which side of the threshold the left launch spot is on; on this stack Re(n² − ε_bone) ≈ −1.6, so the bone is not even an evanescent barrier there.

**Prior art.** The around-head hearing-aid patents above. Nothing found on temporalis-guided or skull-tunneling modes.


## 6. Truly bound: muscle in a limb

**What it is.** Muscle between fat and cortical bone/yellow marrow is truly bound—evanescent on both sides (Re(n² − ε) > 0)—by ordinary fiber-style index guiding. For 25 mm of muscle over 6 mm of cortical bone and the low-water yellow marrow, the TE mode at 2.44 GHz has n = 7.04 − 0.89j, keeps 98 % of its power in muscle, and loses 4.0 dB/cm; at 0.9 GHz it loses 2.1 dB/cm, and at 0.4 GHz it loses 1.5 dB/cm, though there it is barely a wave (β/2α = 1.2). The other truly bound mode here is the occipital CSF mode at 7.99 GHz: TE, n = 6.66 − 1.58j, 47 % in CSF and 46 % in gray matter, 22.9 dB/cm.

**Why it's cool.** None of the five standard flat stacks elsewhere in this catalog has a truly bound mode at or below 6.5 GHz, because their deepest half-space is high-index muscle or brain (and the 2 mm CSF layer is below its asymmetric-slab cutoff until about 6.65–6.9 GHz); the limb is the exception, with low-index yellow marrow inside the bone so no wetter layer beyond the muscle steals the mode. A truly bound mode rides in a wet layer and loses about that layer's own bulk loss (4.0 dB/cm against 3.87 dB/cm for muscle): low-loss layers are cheaper in bulk, but a mode held in them by wet walls pays mostly for leakage into those walls (1.7–3.0 dB/cm for the abdomen and arm fat modes, 5.5 for the skull, 11.5 at the wrist).

**Prior art.** A short, time-boxed search (five queries: limb/arm guided modes, muscle slab waveguide, muscle channel IBC at 2.4 GHz, arm as lossy cylinder, intramuscular/limb channel) found no paper or patent that frames limb muscle between fat and bone as an index-guided RF waveguide mode. The closest adjacent work treats muscle as a galvanic-coupling intra-body channel at 1 kHz–10 MHz (Qin et al., *Open Biomed. Eng. J.* 11:1 (2017), PubMed 28567127), not a guided mode.

## 7. The reverse geometry: a lossy core

Nagel, Blair & Scarpulla's (2011) own geometry swaps where the loss sits: a lossy core between lossless walls. Their "loss-guided" modes exist only because the film is lossy, and their effective index is below the walls', so total internal reflection does not hold them; the paper also finds an anti-guided mode in a lossy film of lower index than its walls. Tissue permittivities give two versions of a lossy core, at two frequencies.

At 7.99 GHz, muscle—the high-index layer—forms a 25 mm lossy core inside lossless fat walls. Its fundamental total-internal-reflection TM mode rides in the core and pays about muscle's own loss (n = 6.83 − 1.28j, 18.6 dB/cm). A different proper TM root loses far less: n = 2.089 − 0.032j, 0.47 dB/cm, with 99 % of its power in the lossless fat and only 1 % in the core. Its Re n is below fat's index (2.18), so the field in the fat is a wave heading toward the core, not an evanescent tail: the root is one of two coupled Zenneck-type surface waves on the fat | muscle faces, proper only because the core is lossy—the tissue version of Nagel et al.'s loss-guided modes. Restoring fat's own real loss (ε = 4.76 − 1.00j, 3.30 dB/cm in bulk) turns this pole improper at about 0.8–0.85 of that loss, as the single-interface Zenneck formula predicts: 0.47, 1.19, 1.91 dB/cm (still proper) at ×0, ×0.25, ×0.5, then improper at ×1 (3.34 dB/cm, close to fat's own bulk rate), where the field grows into the fat while its power still flows toward the muscle—wall loss *breaking* the proper mode here, the mirror image of the main thesis.

At 2.44 GHz the same question can be asked of fat itself, as an antiguide core: how much loss would fat need, alone, before a lossless-muscle | fat | lossless-muscle sandwich carries a proper TM mode, with no help from wall loss? The threshold core loss ε″ is 5.39 at 5 mm of fat, 3.50 at 8 mm and 2.82 at 10 mm of fat—a product ε″·d of about 28 mm across all three—7.0, 4.6 and 3.7 times real fat's own ε″ ≈ 0.77. The root that turns proper there is a new, Nagel-type loss-guided root fed from the walls, not the leaky fat mode made proper (that one stays improper up to at least ε″ = 8). Real fat's own loss is far from enough on its own; in the tissue's actual geometry it is wall loss, not core loss, that moves the fat mode onto the proper sheet.

## Full three-dimensional case results

The detailed FDTD cases use MEEP on the SimNIBS Ernie Extended head, segmented from MRI and CT (Van Hoornweder et al. 2024), with lossy Gabriel tissues on a 0.5/0.25 mm grid (anatomy at 0.5 mm) at 2.44/7.99 GHz. In mode-launch cases, a Huygens sheet excites the 1D profile alone; summary data and decay fits are in [`results/case_*/`](results/).

**Occipital skull mode.** The local-stack, venous-sinus, point-dipole, and phase-index details are in the 3D check under §4 above; Figure 6 in the post shows the field.

Figure 6 in the post shows steady-state radial electric field over one period on a log scale from +20 to −30 dB relative to the mid-bone field 1 cm from the source. Red points out of the head and blue into it; the axial plane passes through the source, and the sagittal plane lies 3 cm downstream.

**Temporalis mode.** At the thinnest spot of each temporal squama (`case_c_left`, `case_c_right`), Ernie's temporalis muscle is 4.05 mm thick over 2.05 mm of bone on the left (1D mode: n = 3.55 − 1.68j, 7.4 dB/cm, 83 % of power in the brain) and 5.05 mm over 3.0 mm of bone on the right (1D mode: n = 3.84 − 1.35j, 6.0 dB/cm, 63 %). On these local stacks, the 1D mode turns leaky below 1.70–1.75 mm of bone on the left and 1.95–2.00 mm on the right, so both launch spots are on the "bound" side. The left margin, 0.3–0.35 mm, is below the 0.5 mm grid, and with Re(n² − ε_bone) ≈ −1.6 the left bone is not even an evanescent barrier, so the 3D run can neither confirm nor refute that side's flat-layer "bound" label. In 3D, the left bone stays at 1.9–2.5 mm for about 1.4 cm before passing 3 mm; the right thickens from 3.0 to 3.4–4 mm within about 1 cm. The thinner left temple sheds more wave into the brain than the right ([`results/case_c_left/`](results/case_c_left/), [`results/case_c_right/`](results/case_c_right/)).

**Head modes at 7.99 GHz.** Both are short-lived. The skull case (`case_d_skull`) launches the bone-dominated branch, which sheds much less into the brain than at 2.44 GHz; with spreading removed it loses **10.5–11.3 dB/cm** over 1–6 cm (1D: 13.7 dB/cm) and falls **24 dB between 1 and 3 cm**. A TM CSF mode (`case_d_csf`), launched at Ernie's thickest occipital CSF pocket, decays faster: beyond about 1.5 cm the wave has left the small pocket, and it loses **19.0–20.2 dB/cm** (1D: 25.3 dB/cm) and falls **43 dB between 1 and 3 cm**.

**Around the head from an external dipole.** From a point dipole 3 mm outside the skin behind and above the left ear (`case_b`), the field creeping around the back of the head reaches the mirror point behind the right ear, about 17 cm away along the head, only 42.1 dB below its level 1 cm from the source. Just inside the skin, with cylindrical spreading removed, it decays at only about 1–2 dB/cm, slower than any head tissue mode computed here, about 3.2 dB/cm or more. The air outside feeds that field rather than the skin guiding it.

---

## Which source launches which mode

The full launch-share table—how much of the surface field |H_y| each mode's own residue term carries at 2 cm and 6 cm from a small electric line source, by source depth and orientation, plus the exactness checks—is in [`results/residues.md`](results/residues.md) and summarized in the post's section 3. A current inside the guiding layer and normal to it couples into the named mode most strongly, carrying −1.3 / −6.4 dB of the surface field at 2 / 6 cm on the upper arm, −6.1 / −15.0 dB at the occiput, and −12.1 / −28.5 dB at the temple. Turning it tangential, or moving it 1 mm into the air above the skin, costs only 3–10 dB more of coupling into the mode itself. What collapses the mode's *share* of the surface field from air—down to −5.3 to −55.3 dB at 2 cm—is that an external source drives the air-side branch-cut waves much harder: with the source 1 mm above the skin and normal to the layer, that share is −17.6 dB on the arm, −34.3 dB at the occiput, and −55.3 dB at the temple.

Figure 3 in the post shows the static launch overview. The limb, fat, skull, and temporalis maps span 10 cm; the Zenneck row spans four guide wavelengths, or 50 cm. The [animated overview](figures/modes/launch_maps.gif) and one GIF per mode remain available.

## Patent and prior-art inventory

This inventory records adjacent patents and applications; it is not a priority claim for the calculations in the post.

- **Fat channel and sensing:** Probingon AB US 12,357,187 B2; Meaney, Augustine et al. EP 4 044 914 A1 (withdrawn); Samsung US 2016/0143558 A1.
- **Surface-wave launchers and around-head links:** CPG Technologies US 10,062,944 B2 and US 9,923,385 B2; Starkey US 10,051,388 B2; GN ReSound EP 2 871 861; Tesla US 787,412.
- **Skull links and implants:** Cochlear US 9,191,759 B2; Boston University US 2023/0140692 A1.
- **Related guide concepts:** University of Central Florida and Clemson University US 7,668,211 B1; NUS US 12,015,197 B2.

## Relatives and analogs

| relative | how it connects | computed? |
|---|---|---|
| Siegman gain-guided fiber | the mirror path in Figure 9 | yes (Figure 9) |
| surface plasmon (SPP) | same pole formula as Zenneck; sign of Re ε decides | yes (Figure 7 in the post) |
| MIM / gap plasmon | Re n rising as the fat gap closes (Figure 4 in the post, left); tissue never has Re ε < 0 | shape analogy only |
| coal-seam mode | lossier walls guide better (Figure 4 in the post, right) | yes (Figure 4 in the post) |
| lossy core (Nagel's own geometry) | the reverse of lossy-wall loss guiding; §7 above, "The reverse geometry: a lossy core" (both the 7.99 GHz Zenneck-type pair on the faces of a lossy muscle core in lossless fat and the low-index 2.44 GHz fat-in-muscle threshold ε″·d ≈ 28 mm) | yes—see `results/modes.md`'s threshold table (smallest fat ε″ at which lossless muscle \| fat \| lossless muscle carries a proper mode, by thickness and polarization) and its 7.99 GHz continuation: with fat's real loss restored the pair's pole turns improper (3.34 dB/cm, close to fat's own 3.30), its field growing into the fat while power still flows toward the muscle |
| Norton ground wave | not a mode (branch-cut contribution); with the space wave, much of what an antenna on or near the skin launches along it, the Norton part more strongly when the antenna is polarized normal to the skin (Grimm & Manteuffel 2014) | yes in 1D (`results/residues.md` cut integrals); partly in 3D—in case B (dipole 3 mm outside the skin behind the left ear) the air carries the signal around the head: the field is nearly uniform over the first 15 mm of air and 9–20 dB stronger 3 mm outside the skin than 1 mm inside (not fitted to a Norton model) |
| ARROW [Duguay et al., *Appl. Phys. Lett.* 49:13 (1986)] | thin high-index skin over low-index fat could act as an antiresonant reflector | no (not tested) |
| spoof SPP [Pendry et al., *Science* 305:847 (2004)] | needs corrugated conductors: body-worn textiles, Tian et al. *Nat. Electron.* 2:243 (2019), NUS US 12,015,197 B2 | no |

**Optical and ultrasound analogs** (desk research): the stratum corneum on viable epidermis at THz is a candidate for one-sided lossy-wall loss guiding, not yet computed; dentinal tubules are a clean optical antiguide (unmodeled); myelin is ARROW/TIR, not lossy-wall loss guiding; photoreceptors and Müller cells are ordinary index guides. In ultrasound the analog is the Lamb wave in cortical bone, but we found no tissue pair for acoustic lossy-wall loss guiding, plausibly because the fast tissues (bone, tendon) are also the lossiest.

## Project notes

- Scope: these five modes and the leaky twin, the reverse-geometry comparison (§7), and the relatives table. The mode-type comparison (truly bound vs "bound" through wall loss vs leaky) lives in the post's section 3; it is not mirrored here.
- Name: "lossy-wall loss guiding" (or "'bound' through wall loss"), crediting Nagel, Blair & Scarpulla (2011)—who take "loss-guided" from Siegman (2003)—while making clear that high-permittivity wet walls provide the grazing reflection and wall loss moves the pole onto the proper (P/P) sheet.
- Figure 10 (`06_frequency.png`): overdamped stretches stay faded; both skull branches are shown.
- Residue/field-share check: `scripts/residues.py` → `results/residues.md`. ARROW and spoof SPP are not computed.
- 3D head model: SimNIBS Ernie Extended (MRI + CT; skin | fat | muscle | compact/spongy bone), chosen by measurement in `head_data_check/`. Cases (`scripts/head_3d.py`): A / a_mode, the occipital skull mode at the back of the head, with and without a mode-launch sheet; B, a dipole 3 mm outside the skin behind the left ear, the around-head Norton/space-wave check; C_left / C_right, the temporalis mode at each side's thinnest squama; D_skull, the occipital skull mode at 7.99 GHz; and D_csf, a TM CSF mode launched in Ernie's thickest occipital CSF pocket at 7.99 GHz. All seven cases have finished running. Results and numbers are in `results/case_*/`; the narrative is in the post's section 5.
- Author-year citations without details here are listed in the README's References.

## Running the 3D cases

All seven MEEP FDTD runs (conda env `tsm`) need the SimNIBS Ernie Extended head model (MRI + CT labels, 0.5 mm; CC BY-NC 4.0, cite Van Hoornweder et al. 2024, doi:10.1162/imag_a_00379), downloaded first with `head_data_check/head_data_check.py`; `run` imports `meep/common.py` from the repository root. Each case goes through the same four steps; on macOS, MPI needs `FI_PROVIDER=udp FI_UDP_IFACE=lo0` (not needed on Linux):

```
python scripts/head_3d.py prep --case a
python scripts/head_3d.py prep --case a_mode        # after case a's prep
mpirun -np 16 python scripts/head_3d.py run --case a --res 2        # d_skull, d_csf: --res 4
python scripts/head_3d.py analyze --case a --res 2
uv run --with numpy --with matplotlib --with imageio-ffmpeg python scripts/head_3d.py videos --case a --res 2
```

`prep` crops the head around the launch site and builds its source or launch-sheet profile (`c_left`, `c_right`, `d_skull` and `d_csf` each need their own). `run` is the FDTD simulation itself: a CW source switched on smoothly (MEEP's tanh turn-on of width 2 periods, near full amplitude by about 10 periods), 24 periods in all, read out by a DFT over the last two periods and checked against a second DFT over the last four. `d_skull` and `d_csf`, at 7.99 GHz, use a 0.25 mm grid (`--res 4`) in place of the default 0.5 mm (`--res 2`). On 16 MPI ranks, `run`'s wall time was about 51 min (case a), 55 min (a_mode), 54 min (b), 33 min (c_left), 34 min (c_right), 31 min (d_csf) and 2.1 hours (d_skull: the largest 0.25 mm domain, 165 M cells, plus an unexplained slow start).
