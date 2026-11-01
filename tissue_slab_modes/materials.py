"""Tissue dielectric properties and head anatomy, with sources.

Every number has a source tag. Tags:
  [G96]      S. Gabriel, R. W. Lau, C. Gabriel, "The dielectric properties of biological
             tissues: III. Parametric models for the dielectric spectrum of tissues,"
             Phys. Med. Biol. 41, 2271-2293 (1996), Table 1 (4-pole Cole-Cole parameters).
             The IT'IS Foundation tissue database v4.1 uses the same model
             (https://itis.swiss/assets/Downloads/TissueDb/V4-1/Documentation-v4-1.pdf).
  [OSSDBS]   CSF Cole-Cole parameters as implemented in OSS-DBS v2 (ossdbs/dielectric_model/
             colecole4.py), Gabriel-based.
  [RDH19]    E. A. Rashed, Y. Diao, A. Hirata, arXiv:1911.01220, Table 1 (Gabriel-based values,
             used here to verify the dura parameters).
  [Pirri21]  C. Pirri et al., Front. Rehabil. Sci. 2:743553 (2021): temporalis 8.1 +/- 1.6 mm
             (relaxed), deep temporal fascia 0.60 +/- 0.20 mm, ultrasound, n = 16.
  [Bhushal25] Bhushal et al., J. Inst. Med. Nepal (2025): temporal squama 1.52 +/- 0.50 mm,
             cadaver, n = 72 (https://www.nepjol.info/index.php/JIOM/article/view/92990).
  [DelBrutto16] O. H. Del Brutto et al., J. Clin. Ultrasound (2016): temporal bone at the
             transcranial-Doppler window 2.7 +/- 0.9 mm (good window), 4.7 +/- 1.2 mm (poor).
  [Saban13]  Sabanciogullari et al., Int. J. Morphol. 31, 38 (2013): skull tables at
             opisthocranion, F/M outer 1.42/1.57, diploe 5.85/5.32, inner 1.29/1.31 mm,
             total 8.57 +/- 1.86 / 8.20 +/- 1.81 mm, MRI, n = 220.
  [NIOSH04]  NIOSH/Anthrotech head-and-face anthropometry (2004), Table 8: head breadth
             M 153.0 +/- 6.0, F 146.8 +/- 5.6 mm; circumference M 575.7 +/- 17.1,
             F 554.9 +/- 17.8 mm; bitragion frontal arc M 304.1 +/- 13.0, F 287.4 +/- 11.9 mm;
             bitragion coronal arc M 350.7, F 339.3 mm; stature M 1753.9, F 1625.4 mm.
  [ANSUR2]   ANSUR II summary (NCSU tables): standing eye height M 1640, F 1517 mm (median).
  [NKJ23]    Nielsen, Kvist, Jakobsen, EuCAP 2023: ear-to-ear geodesic lengths back 2.12 lambda,
             front 2.77 lambda, over the top 3.46 lambda (2.45 GHz).
  [ASSUMED]  no fetched primary source; typical textbook range, flagged so it can be replaced.
  [COMPUTED] derived from the sourced numbers as stated.
"""
from __future__ import annotations

import numpy as np

from .stack import Layer, Stack, EPS0

VARIANTS = ("thin", "typical", "thick")

# ----------------------------------------------------------------------------- dielectrics
# 4-pole Cole-Cole [G96]:  eps(w) = eps_inf + sum_n d_n / (1 + (j w tau_n)^(1 - a_n)) + s_i/(j w eps0)
# rows: eps_inf, [(d1, tau1[s], a1), ... (d4, tau4, a4)], sigma_i [S/m], source
COLE_COLE = {
    "skin":            (4.0, [(32, 7.23e-12, 0.00), (1100, 32.48e-9, 0.20), (0, 1, 0), (0, 1, 0)], 0.0002, "G96 (dry skin)"),
    "fat":             (2.5, [(3, 7.96e-12, 0.20), (15, 15.92e-9, 0.10), (3.3e4, 159.15e-6, 0.05), (1e7, 7.958e-3, 0.01)], 0.01, "G96 (not infiltrated)"),
    "muscle":          (4.0, [(50, 7.23e-12, 0.10), (7000, 353.68e-9, 0.10), (1.2e6, 318.31e-6, 0.10), (2.5e7, 2.274e-3, 0.0)], 0.2, "G96"),
    "bone cortical":   (2.5, [(10, 13.26e-12, 0.20), (180, 79.58e-9, 0.20), (5e3, 159.15e-6, 0.20), (1e5, 15.915e-3, 0.0)], 0.02, "G96"),
    "bone cancellous": (2.5, [(18, 13.26e-12, 0.22), (300, 79.58e-9, 0.25), (2e4, 159.15e-6, 0.20), (2e7, 15.915e-3, 0.0)], 0.07, "G96"),
    "grey matter":     (4.0, [(45, 7.96e-12, 0.10), (400, 15.92e-9, 0.15), (2e5, 106.10e-6, 0.22), (4.5e7, 5.305e-3, 0.0)], 0.02, "G96"),
    "white matter":    (4.0, [(32, 7.96e-12, 0.10), (100, 7.96e-9, 0.10), (4e4, 53.05e-6, 0.30), (3.5e7, 7.958e-3, 0.02)], 0.02, "G96"),
    "blood":           (4.0, [(56, 8.38e-12, 0.10), (5200, 132.63e-9, 0.10), (0, 1, 0), (0, 1, 0)], 0.7, "G96"),
    "csf":             (4.0, [(65, 7.958e-12, 0.10), (40, 1.592e-9, 0.0), (0, 1, 0), (0, 1, 0)], 2.0, "OSSDBS (Gabriel CSF)"),
    "dura":            (4.0, [(40, 7.958e-12, 0.15), (200, 7.958e-9, 0.10), (1e4, 159.155e-6, 0.20), (1e6, 15.915e-3, 0.0)], 0.5, "G96, verified against RDH19"),
    # extra tissues for the body survey: full 4-pole Gabriel parameters as tabulated in
    # IFAC-CNR Appendix C (https://niremf.ifac.cnr.it/docs/DIELECTRIC/AppendixC.html) [G96]
    "bone marrow yellow": (2.5, [(3, 7.958e-12, 0.2), (25, 15.915e-9, 0.1), (5e3, 1591.549e-6, 0.1), (2e6, 15.915e-3, 0.1)], 0.001, "G96 (not infiltrated)"),
    "breast fat":      (2.5, [(3, 17.680e-12, 0.1), (15, 63.660e-9, 0.1), (5e4, 454.700e-6, 0.1), (2e7, 13.260e-3, 0.0)], 0.01, "G96"),
    "cartilage":       (4.0, [(38, 13.263e-12, 0.15), (2500, 144.686e-9, 0.15), (1e5, 318.310e-6, 0.1), (4e7, 15.915e-3, 0.0)], 0.15, "G96"),
    "trachea":         (2.5, [(38, 7.958e-12, 0.1), (400, 63.662e-9, 0.1), (5e4, 15.915e-6, 0.2), (1e6, 15.915e-3, 0.0)], 0.3, "G96"),
    "lung inflated":   (2.5, [(18, 7.958e-12, 0.1), (500, 63.662e-9, 0.1), (2.5e5, 159.155e-6, 0.2), (4e7, 7.958e-3, 0.0)], 0.03, "G96"),
    # G96 tendon (IFAC-CNR Appendix C: tau1 12.243 ps, tau4 1.326 ms) gives eps' 43.1 at 2.44 GHz; NICT
    # animal-tendon measurements give ~51 / ~1.9 S/m at 2.4-2.5 GHz (www2.nict.go.jp dielectric database).
    # Either way tendon is a high-eps lossy wall like muscle, which is all the wrist results depend on.
    "tendon":          (4.0, [(42, 12.243e-12, 0.10), (60, 6.366e-9, 0.10), (6e4, 318.310e-6, 0.22), (2e7, 1.326e-3, 0.0)], 0.25, "G96"),
    "air":             (1.0, [], 0.0, "vacuum"),
}

# reference values at 2.45 GHz used by the tests (IT'IS / Gabriel); eps_r, sigma [S/m]
REFERENCE_2G45 = {"skin": (38.01, 1.464), "fat": (5.28, 0.105), "muscle": (52.73, 1.738),
                  "bone cortical": (11.38, 0.394), "bone cancellous": (18.55, 0.805),
                  "grey matter": (48.91, 1.808), "white matter": (36.17, 1.215), "blood": (58.26, 2.545),
                  "csf": (66.24, 3.458), "dura": (42.04, 1.669),
                  "bone marrow yellow": (5.30, 0.096), "breast fat": (5.15, 0.137), "cartilage": (38.77, 1.756),
                  "trachea": (39.73, 1.449), "lung inflated": (20.48, 0.804)}


def eps(tissue: str, f: float = 2.45e9) -> complex:
    """Complex relative permittivity (Im < 0 = loss, exp(+j w t)) from the 4-pole
    Cole-Cole model [G96]."""
    einf, poles, sig_i, _ = COLE_COLE[tissue]
    if tissue == "air":
        return 1.0 + 0j
    w = 2 * np.pi * f
    e = einf + 0j
    for d, tau, a in poles:
        if d:
            e += d / (1 + (1j * w * tau) ** (1 - a))
    return e + sig_i / (1j * w * EPS0)


def sigma(tissue: str, f: float = 2.45e9) -> float:
    return float(-np.imag(eps(tissue, f)) * 2 * np.pi * f * EPS0)


# ----------------------------------------------------------------------------- anatomy (mm)
# (thin, typical, thick); thin/thick = typical -/+ ~1 SD where an SD is published.
THICKNESS_MM = {
    "temporal": {
        "skin":       ((1.2, 1.5, 2.0), "ASSUMED"),
        "fat":        ((1.0, 2.0, 4.0), "ASSUMED"),
        "temporalis": ((6.5, 8.1, 9.7), "Pirri21 (8.1 +/- 1.6 mm relaxed; incl. 0.6 mm fascia)"),
        "bone":       ((1.0, 1.5, 2.7), "Bhushal25 (1.52 +/- 0.50); thick = DelBrutto16 TCD window 2.7"),
        "dura":       ((0.5, 0.5, 1.0), "ASSUMED"),
        "csf":        ((1.0, 2.0, 3.0), "ASSUMED"),
    },
    "occipital": {
        "skin":       ((1.5, 2.0, 2.5), "ASSUMED (scalp skin)"),
        "fat":        ((2.0, 3.0, 4.5), "ASSUMED (subcutaneous scalp; scalp total ~4-7 mm)"),
        "outer":      ((1.5, 1.5, 1.5), "Saban13 outer table (F 1.42 / M 1.57)"),
        "diploe":     ((3.8, 5.6, 7.4), "Saban13 diploe (F 5.85 / M 5.32), +/- total SD ~1.8"),
        "inner":      ((1.3, 1.3, 1.3), "Saban13 inner table (F 1.29 / M 1.31)"),
        "dura":       ((0.5, 0.5, 1.0), "ASSUMED"),
        "csf":        ((1.0, 2.0, 3.0), "ASSUMED"),
    },
}

GEOMETRY = {
    # head breadth (euryon-euryon): F mean - 1 SD, sex-averaged mean, M mean + 1 SD [NIOSH04]
    "head_breadth": {"thin": 0.1412, "typical": 0.1499, "thick": 0.1590},
    # ear-ear separation used for the floor-reflection geometry (bitragion breadth not found;
    # head breadth used; the floor path is insensitive to it) [COMPUTED]
    "biauricular": {"thin": 0.1412, "typical": 0.1499, "thick": 0.1590},
    # around-the-back path = head circumference - bitragion frontal arc [NIOSH04, COMPUTED]:
    # F 554.9 - 287.4 = 267.5 mm, M 575.7 - 304.1 = 271.6 mm; +/- ~20 mm for thin/thick.
    # Cross-check: back geodesic 2.12 lambda = 260 mm at 2.45 GHz [NKJ23].
    "path_back": {"thin": 0.250, "typical": 0.270, "thick": 0.290},
    "temporal_segment": 0.04,   # length of temporal-type stack at each end of the path [ASSUMED]
    # ear (tragion) height: standing eye height [ANSUR2] minus ~25 mm, sex-averaged (1.55 m);
    # seated: ~1.20 m [ASSUMED, typical office chair]
    "ear_height": {"standing": 1.55, "seated": 1.20},
}

# Measured / modelled ear-to-ear values at 2.45 GHz (loss = -|S21| in dB)
LITERATURE_E2E = [
    dict(loss_dB=66.1, what="measured average, in-the-ear antennas, SAM phantom",
         ref="Ruaro, Thaysen, Jakobsen, IEEE TAP (2016)", url="https://backend.orbit.dtu.dk/ws/files/123683538/07436778.pdf"),
    dict(loss_dB=70.25, what="analytic model vs simulation, heterogeneous 'Billie' (11 y) phantom, incl. pinna",
         ref="Chandra & Johansson, IEEE TAP 61, 6180 (2013)", url="https://arxiv.org/pdf/1304.1739"),
    dict(loss_dB=81.7, what="analytic model vs simulation, heterogeneous 'Duke' (34 y) phantom, incl. pinna",
         ref="Chandra & Johansson, IEEE TAP 61, 6180 (2013)", url="https://arxiv.org/pdf/1304.1739"),
    dict(loss_dB=80.0, what="'around -80 dB' for realistic in-the-ear hearing instruments",
         ref="Kammersgaard, PhD thesis DTU (2018); Nielsen et al., EuCAP (2023)",
         url="https://backend.orbit.dtu.dk/ws/portalfiles/portal/325050721/Ear_to_Ear_Propagation_ModelAntenna_Radiation_Pattern_Optimization.pdf"),
]
LITERATURE_E2E_SAM_OUTER = dict(range_dB=(35, 48), what="SAM phantom, normal monopoles outside the ear, time-gated",
                                ref="Kvist et al., LAPC (2010)", url="https://orbit.dtu.dk/files/5588418/74C11B08d01.pdf")

# Bluetooth LE: nRF5340-class LE-Audio SoC, LE 2M PHY, 0 dBm TX, -95 dBm sensitivity
# (nRF5340 product brief); Core-spec minimum sensitivity -70 dBm (uncoded PHYs).
BLE = {"tx_dBm": 0.0, "sens_dBm": -95.0, "spec_min_sens_dBm": -70.0}


def _i(v):
    return VARIANTS.index(v)


def _t(region, key, v):
    return THICKNESS_MM[region][key][0][_i(v)] * 1e-3


def head_stacks(f=2.44e9, variant="typical"):
    """Planar stacks (air on top, grey matter half-space below) for the temporal region
    next to the ear and the occipital region on the path around the back of the head."""
    T = lambda k: _t("temporal", k, variant)
    O = lambda k: _t("occipital", k, variant)
    temporal = Stack(f, "air", 1.0, [
        Layer("skin", eps("skin", f), T("skin")),
        Layer("fat", eps("fat", f), T("fat")),
        Layer("temporalis muscle", eps("muscle", f), T("temporalis")),
        Layer("cortical bone", eps("bone cortical", f), T("bone")),
        Layer("dura", eps("dura", f), T("dura")),
        Layer("CSF", eps("csf", f), T("csf")),
    ], "grey matter", eps("grey matter", f), f"temporal ({variant})")
    occipital = Stack(f, "air", 1.0, [
        Layer("skin", eps("skin", f), O("skin")),
        Layer("fat", eps("fat", f), O("fat")),
        Layer("cortical bone", eps("bone cortical", f), O("outer")),
        Layer("cancellous bone", eps("bone cancellous", f), O("diploe")),
        Layer("cortical bone", eps("bone cortical", f), O("inner")),
        Layer("dura", eps("dura", f), O("dura")),
        Layer("CSF", eps("csf", f), O("csf")),
    ], "grey matter", eps("grey matter", f), f"occipital ({variant})")
    return {"temporal": temporal, "occipital": occipital}


def through_head_line(f=2.44e9, variant="typical"):
    """Layers along a straight temporal-to-temporal line (just above the ear canals):
    temporal stack, brain, mirrored temporal stack. Brain = 50/50 grey/white matter by
    path length [ASSUMED]."""
    st = head_stacks(f, variant)["temporal"]
    side = [(L.name, L.eps, L.thickness) for L in st.layers]
    brain = GEOMETRY["head_breadth"][variant] - 2 * sum(d for _, _, d in side)
    return side + [("grey matter", eps("grey matter", f), 0.25 * brain),
                   ("white matter", eps("white matter", f), 0.5 * brain),
                   ("grey matter", eps("grey matter", f), 0.25 * brain)] + side[::-1]


# ----------------------------------------------------------------------------- body survey
# Candidate loss-guided channels: a low-eps "core" between high-eps lossy tissue.
# kind 'surface': air | above | core | below | clad half-space ; kind 'deep': clad | core | clad.
# Thickness triples (thin, typical, thick) in mm. Tags: [Stoerchle18] Stoerchle et al., Sci. Rep. 8,
# 16268 (2018) (mean SAT over 8 standard sites 2.9-10.2 mm in lean men; buttocks ~12 mm);
# [Lawrence14] Lawrence et al., Applied Radiology 43(11) 8 (2014) (tracheal ID men 13-25 mm,
# women 10-21 mm; wall 1-3 mm); [ASSUMED] = typical textbook range, not from a fetched source.
def _v(core_fn, triple):
    return {k: core_fn(x) for k, x in zip(VARIANTS, triple)}


BODY_SITES = {
    "temple fat (skin|fat|temporalis)": dict(kind="surface", above=[("skin", 1.5)], below=[], clad="muscle",
        variants=_v(lambda d: [("fat", d)], (1.0, 2.0, 4.0)), source="ASSUMED"),
    "abdomen fat (skin|fat|muscle)": dict(kind="surface", above=[("skin", 2.0)], below=[], clad="muscle",
        variants=_v(lambda d: [("fat", d)], (10.0, 20.0, 40.0)), source="ASSUMED; cf. Stoerchle18"),
    "buttock fat": dict(kind="surface", above=[("skin", 2.0)], below=[], clad="muscle",
        variants=_v(lambda d: [("fat", d)], (8.0, 12.0, 25.0)), source="Stoerchle18 (mean ~12 mm, lean men); range ASSUMED"),
    "thigh fat (anterior)": dict(kind="surface", above=[("skin", 1.5)], below=[], clad="muscle",
        variants=_v(lambda d: [("fat", d)], (5.0, 10.0, 20.0)), source="ASSUMED; cf. Stoerchle18"),
    "upper-arm fat (triceps)": dict(kind="surface", above=[("skin", 1.5)], below=[], clad="muscle",
        variants=_v(lambda d: [("fat", d)], (5.0, 10.0, 15.0)), source="ASSUMED"),
    # Wrist at the watch position (~2-3 cm proximal to the crease). Published ultrasound at the
    # distal forearm reports fascia (0.36-0.44 mm, Pirri22) but not fat, so fat is ASSUMED.
    "wrist volar (skin|fat|flexor tendons|muscle)": dict(kind="surface", above=[("skin", 1.3)],
        below=[("tendon", 3.0)], clad="muscle",
        variants=_v(lambda d: [("fat", d)], (2.0, 3.0, 5.0)), source="ASSUMED; fascia Pirri22"),
    "wrist dorsal (skin|fat|extensor tendons|radius)": dict(kind="surface", above=[("skin", 1.3)],
        below=[("tendon", 1.5), ("bone cortical", 2.0)], clad="bone cancellous",
        variants=_v(lambda d: [("fat", d)], (1.0, 2.0, 4.0)), source="ASSUMED"),
    "breast fat over pectoralis": dict(kind="surface", above=[("skin", 1.5)], below=[], clad="muscle",
        variants=_v(lambda d: [("breast fat", d)], (10.0, 20.0, 40.0)), source="ASSUMED"),
    "temporal squama (temporalis|bone|dura+CSF|brain)": dict(kind="surface",
        above=[("skin", 1.5), ("fat", 2.0), ("muscle", 8.1)], below=[("dura", 0.5), ("csf", 2.0)], clad="grey matter",
        variants=_v(lambda d: [("bone cortical", d)], (1.0, 1.5, 2.7)), source="Bhushal25, DelBrutto16"),
    "occipital skull (scalp|tables+diploe|CSF|brain)": dict(kind="surface",
        above=[("skin", 2.0), ("fat", 3.0)], below=[("dura", 0.5), ("csf", 2.0)], clad="grey matter",
        variants=_v(lambda d: [("bone cortical", 1.5), ("bone cancellous", d), ("bone cortical", 1.3)], (3.8, 5.6, 7.4)),
        source="Saban13"),
    "femur shaft (muscle|cortex|yellow marrow|cortex|muscle)": dict(geometry="tube", kind="deep", clad="muscle",
        variants={"thin": [("bone cortical", 5.0), ("bone marrow yellow", 10.0), ("bone cortical", 5.0)],
                  "typical": [("bone cortical", 6.5), ("bone marrow yellow", 12.0), ("bone cortical", 6.5)],
                  "thick": [("bone cortical", 8.0), ("bone marrow yellow", 14.0), ("bone cortical", 8.0)]},
        source="ASSUMED (cortex 5-8 mm, canal 10-14 mm)"),
    "tibia shaft (muscle|cortex|yellow marrow|cortex|muscle)": dict(geometry="tube", kind="deep", clad="muscle",
        variants={"thin": [("bone cortical", 4.0), ("bone marrow yellow", 8.0), ("bone cortical", 4.0)],
                  "typical": [("bone cortical", 5.0), ("bone marrow yellow", 10.0), ("bone cortical", 5.0)],
                  "thick": [("bone cortical", 6.0), ("bone marrow yellow", 12.0), ("bone cortical", 6.0)]},
        source="ASSUMED (cortex 4-6 mm, canal 8-12 mm)"),
    "trachea (wall|air|wall)": dict(geometry="tube", kind="deep", clad="trachea",
        variants=_v(lambda d: [("air", d)], (13.0, 18.0, 25.0)), source="Lawrence14 (ID 10-25 mm)"),
    "ear canal (skin|air|skin)": dict(geometry="tube", kind="deep", clad="skin",
        variants=_v(lambda d: [("air", d)], (6.0, 7.0, 8.0)), source="ASSUMED (~7 mm diameter)"),
}
