# Mode values behind the README

Every mode the README discusses, recomputed by [`scripts/modes.py`](../scripts/modes.py) with the argument-principle solver in [tissue-slab-modes](https://github.com/sinkarusa/tissue-slab-modes) and the Gabriel 4-pole Cole-Cole permittivities [G96]. Regenerate with `python scripts/modes.py` (this run: 3.3 min; `--no-scan` skips the slowest part). Fields ~ exp(j(ωt − βz)); n = β/k₀ is complex; loss is −Im n · k₀ in dB/cm.

Kinds: **truly bound** = proper on both sides, evanescent on both sides (Re(n² − ε) > 0) and wave-like (β/2α > 1). **"bound"** = proper on both sides only because of loss: on at least one side the field is a damped wave, not an evanescent tail. **leaky** = improper on at least one side: the field grows away from the guide.

## Modes

| id | mode | where | pol | n | loss (dB/cm) | sheet | kind | β/2α | largest power shares |
|---|---|---|---|---|---|---|---|---|---|
| `zenneck_temple` | Zenneck surface wave on skin | temple, 2.44 GHz | TM | 0.973 − 0.023j | 0.10 | P/P | "bound": proper only because of loss | 21.11 | air 99 %, fat 1 %, temporalis 0 % |
| `zenneck_skin` | Zenneck surface wave on skin | skin half-space (no layers), 2.44 GHz | TM | 0.988 − 0.003j | 0.01 | P/P | "bound": proper only because of loss | 148.75 | air 100 %, skin 0 % |
| `fat_arm` | Fat mode | upper arm, 10 mm fat, 2.44 GHz | TM | 1.832 − 0.667j | 2.96 | P/P | "bound": proper only because of loss | 1.37 | fat 82 %, muscle 13 %, air 5 % |
| `fat_arm_twin` | The fat mode's leaky twin | upper arm, 10 mm fat, 2.44 GHz | TM | 1.797 − 0.866j | 3.85 | I/P | leaky: grows into the air | 1.04 | undefined (improper side) |
| `fat_arm_8mm` | Fat mode, 8 mm of fat | upper arm, 8 mm fat, 2.44 GHz | TM | 1.700 − 0.816j | 3.63 | P/P | "bound": proper only because of loss | 1.04 | fat 77 %, muscle 16 %, air 7 % |
| `fat_abdomen` | Fat mode, abdomen | abdomen, 20 mm fat, 2.44 GHz | TM | 2.129 − 0.388j | 1.72 | P/P | "bound": proper only because of loss | 2.74 | fat 92 %, muscle 7 %, air 1 % |
| `fat_wrist` | Fat mode, volar wrist | volar wrist, 3 mm fat, 2.44 GHz | TM | 1.305 − 2.577j | 11.45 | P/P | "bound": proper only because of loss, overdamped (β/2α < 1) | 0.25 | fat 43 %, muscle 43 %, tendon 7 % |
| `fat_sym_8mm` | Fat mode, symmetric model | 8 mm fat between muscle half-spaces, 2.44 GHz | TM | 2.420 − 0.848j | 3.77 | P/P | "bound": proper only because of loss | 1.43 | fat 72 %, muscle 28 % |
| `skull_occipital` | Skull mode | back of the head, 8.4 mm bone, 2.44 GHz | TM | 2.949 − 1.246j | 5.53 | P/P | "bound": proper only because of loss | 1.18 | bone 46 %, gray matter 33 %, fat 17 % |
| `temporalis_6mm` | Temporalis mode, 6 mm of bone | temple, 6 mm squama, 2.44 GHz | TM | 4.487 − 0.971j | 4.31 | P/P | "bound": proper only because of loss | 2.31 | temporalis 40 %, bone 36 %, gray matter 17 % |
| `temporalis_1p5mm` | Temporalis mode, 1.5 mm of bone | temple, 1.5 mm squama, 2.44 GHz | TM | 5.151 − 1.331j | 5.91 | P/I | leaky: grows into the gray matter | 1.94 | undefined (improper side) |
| `limb_muscle_2p44` | Limb-muscle mode | limb (assumed), 2.44 GHz | TE | 7.041 − 0.893j | 3.97 | P/P | truly bound | 3.94 | muscle 98 %, bone 1 %, fat 1 % |
| `limb_muscle_0p9` | Limb-muscle mode | limb (assumed), 0.9 GHz | TE | 6.347 − 1.293j | 2.12 | P/P | truly bound | 2.45 | muscle 86 %, fat 6 %, bone 5 % |
| `limb_muscle_0p4` | Limb-muscle mode | limb (assumed), 0.4 GHz | TE | 5.024 − 2.098j | 1.53 | P/P | truly bound | 1.20 | muscle 56 %, yellow marrow 15 %, fat 12 % |
| `csf_uwb` | CSF mode | back of the head, 2 mm CSF, 7.99 GHz | TE | 6.658 − 1.577j | 22.94 | P/P | truly bound | 2.11 | CSF 47 %, gray matter 46 %, dura 4 % |
| `lossy_core_uwb` | Lossy-core mode | 25 mm muscle between lossless fat, 7.99 GHz | TM | 2.089 − 0.032j | 0.47 | P/P | "bound": proper only because of loss | 32.57 | fat (lossless) 99 %, muscle 1 % |
| `gold_spp` | Surface plasmon on gold | air | Drude gold, 10 µm free-space wavelength | TM | 1.000094 − 0.000020j | 1.11 | P/P | truly bound | 24636.47 | air 100 %, gold 0 % |

## Each half-space: evanescent tail or a wave?

For a wave, √(ε − n²) is its transverse index, the angle is measured from the normal to the layers, and the decay is along the depth (negative: the field grows). The 1/e depth is in free-space wavelengths at the mode's own frequency (λ₀ = 12.3 cm at 2.44 GHz, 10 µm for gold).

| id | side | medium | Re(n² − ε) | field there | √(ε − n²) | angle | direction | decay (dB/cm) | 1/e depth (λ₀) |
|---|---|---|---|---|---|---|---|---|---|
| `zenneck_temple` | top | air | -0.05 | wave | -0.25 − 0.09j | 75.7° | toward the guide | 0.40 | 1.76 |
| `zenneck_temple` | bottom | gray matter | -47.98 | wave | 6.99 − 0.95j | 7.9° | away from the guide | 4.20 | 0.168 |
| `zenneck_skin` | top | air | -0.02 | wave | -0.16 − 0.02j | 81.0° | toward the guide | 0.09 | 7.56 |
| `zenneck_skin` | bottom | skin | -37.04 | wave | 6.15 − 0.87j | 9.1° | away from the guide | 3.88 | 0.182 |
| `fat_arm` | top | air | 1.91 | evanescent |  |  |  | 7.03 | 0.101 |
| `fat_arm` | bottom | muscle | -49.83 | wave | 7.10 − 0.73j | 14.5° | away from the guide | 3.23 | 0.219 |
| `fat_arm_twin` | top | air | 1.48 | evanescent |  |  |  | -6.97 |  |
| `fat_arm_twin` | bottom | muscle | -50.26 | wave | 7.12 − 0.68j | 14.2° | away from the guide | 3.01 | 0.235 |
| `fat_arm_8mm` | top | air | 1.22 | evanescent |  |  |  | 6.48 | 0.109 |
| `fat_arm_8mm` | bottom | muscle | -50.52 | wave | 7.14 − 0.70j | 13.4° | away from the guide | 3.10 | 0.228 |
| `fat_abdomen` | top | air | 3.38 | evanescent |  |  |  | 8.40 | 0.0842 |
| `fat_abdomen` | bottom | muscle | -48.36 | wave | 7.00 − 0.79j | 16.9° | away from the guide | 3.52 | 0.201 |
| `fat_wrist` | top | air | -5.94 | wave | -2.73 − 1.23j | 25.5° | toward the guide | 5.47 | 0.129 |
| `fat_wrist` | bottom | muscle | -57.68 | wave | 7.61 − 0.40j | 9.7° | away from the guide | 1.76 | 0.401 |
| `fat_sym_8mm` | top | muscle | -47.61 | wave | 6.93 − 0.62j | 19.3° | away from the guide | 2.77 | 0.255 |
| `fat_sym_8mm` | bottom | muscle | -47.61 | wave | 6.93 − 0.62j | 19.3° | away from the guide | 2.77 | 0.255 |
| `skull_occipital` | top | air | 6.14 | evanescent |  |  |  | 12.45 | 0.0568 |
| `skull_occipital` | bottom | gray matter | -41.79 | wave | 6.48 − 0.46j | 24.5° | away from the guide | 2.03 | 0.348 |
| `temporalis_6mm` | top | air | 18.19 | evanescent |  |  |  | 19.45 | 0.0363 |
| `temporalis_6mm` | bottom | gray matter | -29.74 | wave | 5.47 − 0.42j | 39.4° | away from the guide | 1.85 | 0.383 |
| `temporalis_1p5mm` | top | air | 23.76 | evanescent |  |  |  | 22.47 | 0.0315 |
| `temporalis_1p5mm` | bottom | gray matter | -24.16 | wave | 4.92 + 0.05j | 46.3° | away from the guide | -0.20 |  |
| `limb_muscle_2p44` | top | air | 47.78 | evanescent |  |  |  | 30.96 | 0.0228 |
| `limb_muscle_2p44` | bottom | yellow marrow | 43.48 | evanescent |  |  |  | 29.56 | 0.0239 |
| `limb_muscle_0p9` | top | air | 37.61 | evanescent |  |  |  | 10.27 | 0.0254 |
| `limb_muscle_0p9` | bottom | yellow marrow | 33.11 | evanescent |  |  |  | 9.67 | 0.027 |
| `limb_muscle_0p4` | top | air | 19.84 | evanescent |  |  |  | 3.60 | 0.0322 |
| `limb_muscle_0p4` | bottom | yellow marrow | 15.17 | evanescent |  |  |  | 3.26 | 0.0356 |
| `csf_uwb` | top | air | 40.84 | evanescent |  |  |  | 95.80 | 0.0242 |
| `csf_uwb` | bottom | gray matter | 0.95 | evanescent |  |  |  | 22.57 | 0.103 |
| `lossy_core_uwb` | top | fat (lossless) | -0.40 | wave | -0.64 − 0.10j | 72.9° | toward the guide | 1.52 | 1.52 |
| `lossy_core_uwb` | bottom | fat (lossless) | -0.40 | wave | -0.64 − 0.10j | 72.9° | toward the guide | 1.52 | 1.52 |
| `gold_spp` | top | air | 0.00 | evanescent |  |  |  | 754.07 | 11.5 |
| `gold_spp` | bottom | gold | 5065.68 | evanescent |  |  |  | 3906429.66 | 0.00222 |

## Against the numbers the README and CATALOG quote

The computed value is rounded the way the quote is written.

| id | quantity | quoted | source | computed | |
|---|---|---|---|---|---|
| `zenneck_temple` | n | 0.973 − 0.023j | README | 0.973 − 0.023j | ✓ |
| `zenneck_temple` | loss | 0.10 | README | 0.10 | ✓ |
| `zenneck_temple` | sheet | P/P | README | P/P | ✓ |
| `zenneck_temple` | share:air | 99 | README | 99 | ✓ |
| `zenneck_temple` | depth:top | 1.8 | decision Q40 | 1.8 | ✓ |
| `zenneck_skin` | depth:top | 7.6 | decision Q40 | 7.6 | ✓ |
| `fat_arm` | n | 1.832 − 0.667j | README | 1.832 − 0.667j | ✓ |
| `fat_arm` | loss | 2.96 | README | 2.96 | ✓ |
| `fat_arm` | sheet | P/P | README | P/P | ✓ |
| `fat_arm` | share:fat | 82 | README | 82 | ✓ |
| `fat_arm_twin` | n | 1.797 − 0.866j | README | 1.797 − 0.866j | ✓ |
| `fat_arm_twin` | loss | 3.85 | README | 3.85 | ✓ |
| `fat_arm_twin` | sheet | I/P | README | I/P | ✓ |
| `fat_arm_8mm` | n | 1.70 − 0.82j | README | 1.70 − 0.82j | ✓ |
| `fat_arm_8mm` | kappa:bottom | 7.14 − 0.70j | README | 7.14 − 0.70j | ✓ |
| `fat_arm_8mm` | angle:bottom | 13 | README | 13 | ✓ |
| `fat_arm_8mm` | decay:bottom | 3.1 | README | 3.1 | ✓ |
| `fat_sym_8mm` | loss | 3.77 | README | 3.77 | ✓ |
| `skull_occipital` | n | 2.949 − 1.246j | README | 2.949 − 1.246j | ✓ |
| `skull_occipital` | loss | 5.53 | README | 5.53 | ✓ |
| `skull_occipital` | sheet | P/P | README | P/P | ✓ |
| `skull_occipital` | share:bone | 46 | README | 46 | ✓ |
| `skull_occipital` | b2a | 1.2 | README | 1.2 | ✓ |
| `skull_occipital` | kappa:bottom | 6.48 − 0.46j | README | 6.48 − 0.46j | ✓ |
| `skull_occipital` | angle:bottom | 24 | README | 24 | ✓ |
| `skull_occipital` | decay:bottom | 2.0 | README | 2.0 | ✓ |
| `temporalis_6mm` | n | 4.49 − 0.97j | README | 4.49 − 0.97j | ✓ |
| `temporalis_6mm` | loss | 4.3 | README | 4.3 | ✓ |
| `temporalis_6mm` | sheet | P/P | README | P/P | ✓ |
| `temporalis_1p5mm` | n | 5.151 − 1.331j | README | 5.151 − 1.331j | ✓ |
| `temporalis_1p5mm` | loss | 5.91 | README | 5.91 | ✓ |
| `temporalis_1p5mm` | sheet | P/I | README | P/I | ✓ |
| `limb_muscle_2p44` | n | 7.04 − 0.89j | README | 7.04 − 0.89j | ✓ |
| `limb_muscle_2p44` | loss | 4.0 | README | 4.0 | ✓ |
| `limb_muscle_2p44` | share:muscle | 98 | README | 98 | ✓ |
| `limb_muscle_0p9` | n | 6.35 − 1.29j | README | 6.35 − 1.29j | ✓ |
| `limb_muscle_0p9` | loss | 2.1 | README | 2.1 | ✓ |
| `limb_muscle_0p9` | share:muscle | 86 | README | 86 | ✓ |
| `limb_muscle_0p4` | n | 5.02 − 2.10j | README | 5.02 − 2.10j | ✓ |
| `limb_muscle_0p4` | loss | 1.5 | README | 1.5 | ✓ |
| `limb_muscle_0p4` | share:muscle | 56 | README | 56 | ✓ |
| `limb_muscle_0p4` | b2a | 1.2 | README | 1.2 | ✓ |
| `csf_uwb` | n | 6.66 − 1.58j | README | 6.66 − 1.58j | ✓ |
| `csf_uwb` | loss | 22.9 | README | 22.9 | ✓ |
| `csf_uwb` | share:CSF | 47 | README | 47 | ✓ |
| `lossy_core_uwb` | n | 2.089 − 0.032j | README | 2.089 − 0.032j | ✓ |
| `lossy_core_uwb` | loss | 0.47 | README | 0.47 | ✓ |

## Selection rules and notes

- `zenneck_temple` (temple, 2.44 GHz): selection: lowest-loss P/P root with more than half its power in air (pick 'surface'), as for figure 01
- `zenneck_skin` (air | skin, 2.44 GHz): selection: n = sqrt(eps/(1+eps)) on a skin half-space, as in figure 05
- `fat_arm` (upper arm, 10 mm fat, 2.44 GHz): selection: lowest-loss P/P root with Re n >= 1 and under half its power in air (pick 'tissue')
- `fat_arm_twin` (upper arm, 10 mm fat, 2.44 GHz): selection: the I/P root nearest the fat mode, as in figure 02
- `fat_arm_8mm` (upper arm, 8 mm fat): selection: as for the 10 mm upper arm
- `fat_abdomen` (abdomen, 20 mm fat, 2.44 GHz): selection: lowest-loss P/P root with Re n >= 1 and under half its power in air (pick 'tissue')
- `fat_wrist` (volar wrist, 3 mm fat, 2.44 GHz): selection: lowest-loss P/P root with Re n >= 1 and under half its power in air (pick 'tissue')
- `fat_sym_8mm` (muscle | fat 8 mm | muscle): selection: TM0 of the symmetric slab: lowest-loss bound root with Re n > 1 (make_figures.tm0_sym), the model of figure 04
- `skull_occipital` (back of the head, 8.4 mm bone, 2.44 GHz): selection: pick 'tissue', as for figure 01
- `temporalis_6mm` (temple, 6 mm squama, 2.44 GHz): selection: lowest-loss P/P root at 6 mm of bone with 2 < Re n < 7, continued to 6.0 mm (make_figures.temporalis_mode), as for figure 07
- `temporalis_1p5mm` (temple, 1.5 mm squama, 2.44 GHz): selection: lowest-loss P/P root at 6 mm of bone with 2 < Re n < 7, continued to 1.5 mm (make_figures.temporalis_mode), as for figure 07
- `limb_muscle_2p44` (limb (assumed)): selection: truly bound TE root with the largest power share in muscle (3 truly bound TE roots at this frequency)
- `limb_muscle_0p9` (limb (assumed)): selection: truly bound TE root with the largest power share in muscle (1 truly bound TE roots at this frequency)
- `limb_muscle_0p4` (limb (assumed)): selection: truly bound TE root with the largest power share in muscle (1 truly bound TE roots at this frequency)
- `csf_uwb` (back of the head, 2 mm CSF, 7.99 GHz): selection: truly bound TE root with the largest power share in CSF (1 truly bound TE roots)
- `lossy_core_uwb` (lossless fat | muscle 25 mm | lossless fat): selection: lowest-loss bound TM root with 1.5 < Re n < 3: one of the two coupled Zenneck-type waves on the fat | muscle faces; partner: 2.088 − 0.035j; single interface (lossless fat | muscle): 2.088 − 0.034j
- `gold_spp` (air | gold, 10 µm): selection: n = sqrt(eps/(1+eps)) at a free-space wavelength of 10 µm, Drude gold as in figure 05

## Checks

### Truly bound modes in the five standard stacks

Every root in the APM window Re z ∈ (-3.5, 3.5) tested; a cell lists the truly bound roots (– = none).

| stack | pol | 0.4 GHz | 0.9 GHz | 1.5 GHz | 2.44 GHz | 3.5 GHz | 4.5 GHz | 5.5 GHz | 6 GHz | 6.5 GHz | 6.75 GHz | 7 GHz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abdomen | TM | – | – | – | – | – | – | – | – | – |  |  |
| abdomen | TE | – | – | – | – | – | – | – | – | – |  |  |
| upper arm | TM | – | – | – | – | – | – | – | – | – |  |  |
| upper arm | TE | – | – | – | – | – | – | – | – | – |  |  |
| volar wrist | TM | – | – | – | – | – | – | – | – | – |  |  |
| volar wrist | TE | – | – | – | – | – | – | – | – | – |  |  |
| back of the head | TM | – | – | – | – | – | – | – | – | – | – | – |
| back of the head | TE | – | – | – | – | – | – | – | – | – | – | 6.67 − 1.45j |
| temple | TM | – | – | – | – | – | – | – | – | – | – | – |
| temple | TE | – | – | – | – | – | – | – | – | – | 6.68 − 1.39j | 6.64 − 1.24j, 6.68 − 1.43j |

First truly bound root: 6.75 GHz (temple, TE); none at 0.4, 0.9, 1.5, 2.44, 3.5, 4.5, 5.5, 6, 6.5 GHz.

### Index, wavelength and plane-wave loss in each tissue, 2.44 GHz

| tissue | ε | index Re √ε | wavelength (cm) | plane-wave loss (dB/cm) |
|---|---|---|---|---|
| air | 1.0 + 0.0j | 1.00 | 12.3 | -0.00 |
| fat | 5.3 − 0.8j | 2.30 | 5.3 | 0.74 |
| cortical bone | 11.4 − 2.9j | 3.40 | 3.6 | 1.89 |
| cancellous bone | 18.6 − 5.9j | 4.36 | 2.8 | 3.01 |
| skin | 38.0 − 10.7j | 6.23 | 2.0 | 3.83 |
| dura | 42.0 − 12.2j | 6.55 | 1.9 | 4.15 |
| muscle | 52.7 − 12.8j | 7.31 | 1.7 | 3.87 |
| gray matter | 48.9 − 13.3j | 7.06 | 1.7 | 4.17 |
| CSF | 66.3 − 25.4j | 8.28 | 1.5 | 6.81 |

### Tube cutoffs (lowest mode, TE11, of a tube with conducting walls)

A tube carries nothing below this frequency. The walls are treated as perfect conductors and the core as its thickness-weighted mean ε′, so this is a first-order screen, not a solved mode. Geometries: `BODY_SITES` in tissue-slab-modes `materials.py`.

| site | core width thin / typical / thick (mm) | cutoff thin / typical / thick (GHz) | geometry source |
|---|---|---|---|
| femur shaft (muscle|cortex|yellow marrow|cortex|muscle) | 20 / 25 / 30 | 3.0 / 2.4 / 2.0 | ASSUMED (cortex 5-8 mm, canal 10-14 mm) |
| tibia shaft (muscle|cortex|yellow marrow|cortex|muscle) | 16 / 20 / 24 | 3.8 / 3.0 / 2.5 | ASSUMED (cortex 4-6 mm, canal 8-12 mm) |
| trachea (wall|air|wall) | 13 / 18 / 25 | 13.5 / 9.8 / 7.0 | Lawrence14 (ID 10-25 mm) |
| ear canal (skin|air|skin) | 6 / 7 / 8 | 29.3 / 25.1 / 22.0 | ASSUMED (~7 mm diameter) |

### Lossy-core threshold (Nagel, Blair & Scarpulla's geometry)

Smallest fat ε″ at which lossless muscle | fat | lossless muscle carries a proper mode at 2.44 GHz (real fat has ε″ = 0.77). This is a new loss-guided root, not the leaky fundamental turning proper.

| fat (mm) | pol | ε″ threshold |
|---|---|---|
| 5 | TM | 5.39 |
| 5 | TE | 58.63 |
| 8 | TM | 3.50 |
| 8 | TE | 30.60 |
| 10 | TM | 2.82 |
| 10 | TE | 20.95 |

### Lossy-core pair with the fat's own loss restored (7.99 GHz, TM, 25 mm muscle)

Followed from lossless fat (the `lossy_core_uwb` row) to real fat (ε = 4.76 − 1.00j), whose plane-wave loss is 3.30 dB/cm. γ_fat is normalized to k0.

| fat loss | n | loss (dB/cm) | fat side | Re γ_fat | Zenneck formula, fat \| muscle (dB/cm) |
|---|---|---|---|---|---|
| ×0 | 2.089 − 0.032j | 0.47 | proper | +0.105 | 0.49 |
| ×0.25 | 2.088 − 0.082j | 1.19 | proper | +0.072 | 1.21 |
| ×0.5 | 2.088 − 0.132j | 1.91 | proper | +0.040 | 1.94 |
| ×1 | 2.093 − 0.230j | 3.34 | improper | -0.025 | 3.37 |

### Other quoted values

- Temporalis mode turns leaky below 1.64 mm of bone (decision Q37: "bound" down to 1.64 mm).
- 8 mm of lossless fat in real muscle (figure 03, s = 1): n = 2.458 − 0.672j, 2.99 dB/cm, proper (CATALOG: "bound" at 3.0 dB/cm).
- Figure 04: Re n = 2.29 at 25 mm of fat and 4.42 at 1 mm (README: 2.29 and 4.42).
- Lossy core: CATALOG quotes ε″ ≈ 3.5 for 8 mm of fat and ε″·d ≈ 28 mm; computed (TM) 5.39 at 5 mm (ε″·d = 26.9 mm), 3.50 at 8 mm (ε″·d = 28.0 mm), 2.82 at 10 mm (ε″·d = 28.2 mm).

## Mismatches

None.

## Figures

- `figures/modes/<id>.png`: the transverse profile of each mode above.
- `figures/modes/<id>_launch.png`, `figures/modes/<id>_launch.gif` and `figures/modes/launch_maps.png` / `.gif`: the mode alone, launched by its own profile at z = 0. This is idealized on purpose: a real antenna, a dipole say, also excites other modes and branch-cut waves.
