"""3D cases on a real head (MEEP FDTD): 2.44 GHz, and case D at 7.99 GHz (UWB).

Case a: a dipole in the occipital diploe, normal to the bone. Does the lossy-wall skull mode of the
  1D occipital stack show up in a real skull, and where does its power go?
Case b: a dipole 3 mm outside the skin behind the left ear, normal to the skin (where a hearing aid
  sits). What runs along the surface around the back of the head?
Case a_mode: case A's spot, but instead of a dipole a Huygens sheet that launches the 1D skull mode
  alone (its own E and H profile, laid onto the real layers), toward the right ear. With nothing but
  the mode launched, everything else in the field is what the real head turns the mode into.
Cases c_left, c_right: the temporalis mode alone, launched forward at the thinnest spot of each
  temporal squama (left 2.05 mm along the bone normal, right 3.0 mm). The 1D mode of each side's own
  layers turns leaky into the brain below 1.70-1.75 mm (left) and 1.95-2.00 mm (right) of bone.
Case d_skull: case a_mode at 7.99 GHz, on a 0.25 mm grid (--res 4).
Case d_csf: a TM CSF mode alone at 7.99 GHz (--res 4), launched toward the right ear in Ernie's thickest
  occipital CSF pocket (2 x the distance transform: 3.6-5.4 mm over about 1.5 x 1 cm, 7.8 mm along the
  launch point's bone normal; just right of the superior sagittal sinus; elsewhere under the occipital
  bone the CSF is mostly 1-2 mm).

Head: SimNIBS Ernie Extended (MRI + CT labels, 0.5 mm; CC BY-NC 4.0, cite Van Hoornweder et al.
2024, doi:10.1162/imag_a_00379). Download it first with head_data_check/head_data_check.py.
Tissue properties: the same Gabriel 4-pole Cole-Cole values [G96] the 1D stacks use
(tissue_slab_modes.materials), evaluated at the case's frequency and loaded into MEEP as eps' +
D-conductivity.

Steps (conda env tsm; MPI needs FI_PROVIDER=udp FI_UDP_IFACE=lo0 on macOS):
  python scripts/head_3d.py prep --case a                   # labels -> cropped cache, source, normal
  python scripts/head_3d.py prep --case a_mode              # (after case a's prep) the launch sheet
  python scripts/head_3d.py prep --case c_left              # crop and sheet (c_right, d_skull, d_csf alike)
  mpirun -np 16 python scripts/head_3d.py run --case a --res 2     # d_skull, d_csf: --res 4 (0.25 mm)
  python scripts/head_3d.py analyze --case a --res 2        # figures + numbers + ribbon fields
  uv run --with numpy --with matplotlib --with imageio-ffmpeg python scripts/head_3d.py videos --case a --res 2

The run uses a CW source switched on smoothly (MEEP's tanh turn-on, width 2 periods: half amplitude at
6 periods, 98 % at 10), 24 periods in all (waves through brain travel at
~c/7). The phasor is the DFT over the last two periods; a second DFT over the last four periods
checks that the transient has died out. Time-domain frames of E . r_hat (every 1/16 period, both
planes) feed the videos.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NII = ROOT / "head_data_check/data/ernie_extended/tissue_labeling_upsampled.nii.gz"
CASES = {
    "a": dict(ray=(0.0, -1.0, 0.2), source="diploe", ref="mid-bone",
              title="Case A: dipole in the occipital diploe, normal to the bone"),
    "a_mode": dict(ray=(0.0, -1.0, 0.2), source="diploe", launch="mode", mode="skull", crop="a", ref="mid-bone",
                   tag="Case A, mode launch",
                   title="Case A, mode launch: the skull mode alone, launched toward the right ear"),
    "b": dict(ray=(-0.8, -0.8, -0.1), source="air", offset_mm=3.0, ref="air, 3 mm outside the skin",
              title="Case B: dipole 3 mm outside the skin behind the left ear, normal to the skin"),
    # thinnest squama per side: largest inscribed sphere (2 x EDT) over a fan of rays, as in
    # head_data_check; crop boxes are (lo, hi) mm per axis relative to the source
    "c_left": dict(ray=(-1.0, -0.05, 0.4), source="bone_mid", launch="mode", mode="temporalis", axis="y",
                   planes=("axial", "coronal"), box=((-25, 95), (-40, 110), (-55, 55)),
                   ref="1 mm outside the outer table", tag="Case C, left temple",
                   title="Case C, left temple: the temporalis mode alone at the thinnest left squama, launched forward"),
    "c_right": dict(ray=(1.0, 0.05, -0.05), source="bone_mid", launch="mode", mode="temporalis", axis="y",
                    planes=("axial", "coronal"), box=((-95, 25), (-40, 110), (-55, 55)),
                    ref="1 mm outside the outer table", tag="Case C, right temple",
                    title="Case C, right temple: the temporalis mode alone at the thinnest right squama, launched forward"),
    "d_skull": dict(ray=(0.0, -1.0, 0.2), source="diploe", launch="mode", mode="skull", f=7.9872e9,
                    box=((-30, 70), (-25, 45), (-45, 45)), dpml=25.0, w_mode=10.0, mode_below=15.0,
                    ref="mid-bone", tag="Case D, skull mode",
                    title="Case D: case A's skull mode alone at 7.99 GHz (UWB), launched toward the right ear"),
    "d_csf": dict(ray=(0.098, -0.935, 0.342), source="bone_mid", launch="mode", mode="csf", f=7.9872e9,
                  box=((-20, 50), (-25, 35), (-30, 30)), dpml=25.0, w_mode=6.0, mode_below=15.0,
                  ref="1 mm inside the inner table", tag="Case D, CSF mode",
                  title="Case D: the CSF mode alone in the thickest occipital CSF pocket, launched toward the right ear"),
}
CASE, CFG, CACHE, SHEET, OUT = None, None, None, None, None

C0 = 299792458.0
VOX = 0.5                                     # label grid, mm
MARGIN = 15.0                                 # air between head and PML, mm
# per-case settings (configure): frequency, PML thickness (mm); for a mode launch the launched beam's
# 1/e half-width along the bone, how far the sheet reaches above the skin (the crop keeps 15 mm of
# air) and below the inner table, and how far downstream of the sheet the second recording plane sits
F, DPML, W_MODE, MODE_ABOVE, MODE_BELOW = 2.44e9, 40.0, 20.0, 12.0, 50.0
PLANE_OFFSET = 30.0
# recording planes: normal axis, in-plane axes, axis labels, view
PLANES = {"axial": ("z", ("x", "y"), ("x (mm, R+)", "y (mm, A+)"), "seen from above; anterior up"),
          "sagittal": ("x", ("y", "z"), ("y (mm, A+)", "z (mm, S+)"), "seen from the right; anterior right"),
          "coronal": ("y", ("x", "z"), ("x (mm, R+)", "z (mm, S+)"), "seen from behind; right side right")}


def configure(case: str):
    global CASE, CFG, CACHE, SHEET, OUT, F, DPML, W_MODE, MODE_ABOVE, MODE_BELOW
    CASE, CFG = case, CASES[case]
    data = ROOT / "head_data_check/data/ernie_extended"
    CACHE = data / f"case_{CFG.get('crop', case)}_crop.npz"
    SHEET = data / f"case_{case}_sheet.npz"
    OUT = ROOT / f"results/case_{case}"
    F, DPML = CFG.get("f", 2.44e9), CFG.get("dpml", 40.0)
    W_MODE, MODE_ABOVE, MODE_BELOW = CFG.get("w_mode", 20.0), CFG.get("mode_above", 12.0), CFG.get("mode_below", 50.0)


def is_bone_case():
    return CFG["source"] in ("diploe", "bone_mid")


def planes_of_case():
    """Recording planes: the first holds the bone normal and the propagation direction; for a mode
    launch the second is parallel to the sheet, PLANE_OFFSET downstream."""
    return CFG.get("planes", ("axial", "sagittal"))


def ghz():
    return f"{F / 1e9:.2f} GHz"


# Ernie Extended charm labels -> tissue_slab_modes.materials names. Eyes (vitreous, eps ~ 69) are
# outside the crop; CSF stands in for them.
TISSUES = ["air", "white matter", "grey matter", "csf", "bone cortical", "skin", "csf",
           "bone cortical", "bone cancellous", "blood", "muscle", "cartilage", "fat"]
BONE = (4, 7, 8)                              # generic bone, compact, spongy
SHORT = {0: "air", 1: "WM", 2: "GM", 3: "CSF", 4: "bone", 5: "skin", 6: "eye", 7: "bone",
         8: "diploe", 9: "blood", 10: "muscle", 11: "cartilage", 12: "fat"}


def read_nifti_labels(path: Path):
    """Label volume resampled (exactly; the affine is a signed axis permutation) to world axes:
    lab[ix, iy, iz] at world = origin + VOX * index (RAS mm)."""
    b = gzip.open(path).read()
    dim = np.frombuffer(b, "<i2", 8, 40)
    assert np.frombuffer(b, "<i2", 1, 70)[0] == 4, "expected int16 labels"
    off = int(np.frombuffer(b, "<f4", 1, 108)[0])
    srow = np.frombuffer(b, "<f4", 12, 280).reshape(3, 4).astype(float)
    shape = tuple(int(n) for n in dim[1:4])
    arr = np.frombuffer(b, "<i2", int(np.prod(shape)), off).reshape(shape, order="F")
    A, t = srow[:, :3], srow[:, 3]
    perm = [int(np.argmax(abs(A[w]))) for w in range(3)]
    assert np.allclose(abs(A[range(3), perm]), VOX) and np.count_nonzero(A) == 3
    lab = arr.transpose(perm)
    origin = t.copy()
    for w in range(3):
        s = A[w, perm[w]]
        if s < 0:
            lab = np.flip(lab, axis=w)
            origin[w] = t[w] + s * (lab.shape[w] - 1)
    return np.ascontiguousarray(lab.astype(np.int8)), origin


def prep():
    if CFG.get("launch") != "mode" or "crop" not in CFG:
        prep_crop()
    elif not CACHE.exists():
        raise SystemExit(f"{CACHE.name} missing: run `prep --case {CFG['crop']}` first")
    if CFG.get("launch") == "mode":
        prep_sheet()


def prep_crop():
    lab, origin = read_nifti_labels(NII)
    idx = np.argwhere(np.isin(lab, (1, 2, 3)))
    center = origin + VOX * idx.mean(0)
    d = np.array(CFG["ray"]) / np.linalg.norm(CFG["ray"])
    a = np.arange(150.0, 0.0, -0.1)
    pts = center + a[:, None] * d
    ii = np.rint((pts - origin) / VOX).astype(int)
    seq = lab[ii[:, 0], ii[:, 1], ii[:, 2]]
    if CFG["source"] == "diploe":                       # middle of the first spongy-bone run from outside
        sp = np.flatnonzero(seq == 8)
        run = [sp[0]]
        for k in sp[1:]:
            if k != run[-1] + 1:
                break
            run.append(k)
        src = pts[run].mean(0)
        print("diploe run at source", round(len(run) * 0.1, 1), "mm")
    elif CFG["source"] == "bone_mid":                   # middle of the first bone run from outside
        b = np.isin(seq, BONE)
        first = int(np.argmax(b))
        end = first + int(np.argmax(~b[first:]))
        src = pts[first:end].mean(0)
        print("bone run at source", round((end - first) * 0.1, 1), "mm along the ray")
    else:                                               # offset_mm outside the skin entry point
        k_skin = np.flatnonzero(seq > 0)[0]
        src = pts[k_skin] + CFG["offset_mm"] * d
    # local normal: smallest-variance axis of the bone (a) / head-surface (b) voxels within 8 mm
    lo = np.floor((src - 8 - origin) / VOX).astype(int)
    hi = np.ceil((src + 8 - origin) / VOX).astype(int) + 1
    block = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    near = np.isin(block, BONE) if is_bone_case() else (block > 0) & ~ndi_binary_erosion(block > 0)
    sub = np.argwhere(near) + lo
    p = origin + VOX * sub
    p = p[np.linalg.norm(p - src, axis=1) < 8]
    w, v = np.linalg.eigh(np.cov((p - p.mean(0)).T))
    normal = v[:, 0] * np.sign(v[:, 0] @ d)
    head = origin + VOX * np.argwhere(lab > 0)
    if "box" in CFG:                                    # box around the source, at most MARGIN of air
        box = np.array(CFG["box"], float)
        box_lo = np.maximum(src + box[:, 0], head.min(0) - MARGIN)
        box_hi = np.minimum(src + box[:, 1], head.max(0) + MARGIN)
    else:                                               # posterior head; cut at the front (y) and bottom (z)
        zlo, yhi = src[2] - 70.0, center[1] + 40.0
        hw = head[(head[:, 2] > zlo) & (head[:, 1] < yhi)]
        box_lo = np.array([hw[:, 0].min() - MARGIN, hw[:, 1].min() - MARGIN, zlo])
        box_hi = np.array([hw[:, 0].max() + MARGIN, yhi, hw[:, 2].max() + MARGIN])
    i0 = np.floor((box_lo - origin) / VOX).astype(int)
    i1 = np.ceil((box_hi - origin) / VOX).astype(int) + 1
    i0, i1 = np.maximum(i0, 0), np.minimum(i1, lab.shape)
    crop = lab[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]].copy()
    corig = origin + VOX * i0
    np.savez_compressed(CACHE, lab=crop, origin=corig, center=center, src=src, normal=normal)
    print("cranial center", center.round(1), "source", src.round(1), "normal", normal.round(3),
          f"(angle to ray {np.degrees(np.arccos(normal @ d)):.1f} deg)")
    print("crop", crop.shape, "voxels =", (np.array(crop.shape) * VOX).round(0), "mm; labels",
          np.unique(crop).tolist())


def ndi_binary_erosion(mask):
    """6-neighbor binary erosion (numpy only, so prep needs no scipy)."""
    e = mask.copy()
    for ax in range(3):
        e[tuple(slice(1, None) if a == ax else slice(None) for a in range(3))] &= \
            mask[tuple(slice(None, -1) if a == ax else slice(None) for a in range(3))]
        e[tuple(slice(None, -1) if a == ax else slice(None) for a in range(3))] &= \
            mask[tuple(slice(1, None) if a == ax else slice(None) for a in range(3))]
    return e


def load_cache():
    z = np.load(CACHE)
    return {k: z[k] for k in z.files}


def run(res: float, periods_on: float):
    import meep as mp
    sys.path.insert(0, str(ROOT.parent / "meep"))
    from common import fmeep, medium
    from tissue_slab_modes.materials import eps

    c = load_cache()
    lab = c["lab"]
    nx, ny, nz = lab.shape
    flat = lab.tobytes()
    size = np.array(lab.shape) * VOX
    mid = c["origin"] + size / 2 - VOX / 2           # world coordinate of the MEEP origin
    media = [mp.air] + [medium(eps(t, F), F) for t in TISSUES[1:]]
    ox, oy, oz = (mid - c["origin"]) / VOX           # index of the MEEP origin
    inv = 1 / VOX

    def material(p):
        i = min(max(int(p.x * inv + ox + 0.5), 0), nx - 1)
        j = min(max(int(p.y * inv + oy + 0.5), 0), ny - 1)
        k = min(max(int(p.z * inv + oz + 0.5), 0), nz - 1)
        return media[flat[(i * ny + j) * nz + k]]

    fm = fmeep(F)
    T = 1 / fm
    s = c["src"] - mid
    n = c["normal"]
    cw = mp.ContinuousSource(frequency=fm, width=2 * T)
    mode = CFG.get("launch") == "mode"
    if mode:
        sources = sheet_sources(mp, mid, cw)
    else:
        sources = [mp.Source(cw, component=comp, center=mp.Vector3(*s), amplitude=float(a))
                   for comp, a in zip((mp.Ex, mp.Ey, mp.Ez), n) if abs(a) > 1e-3]
    cell = mp.Vector3(*(size + 2 * DPML))
    # extra_materials: a material function is opaque to MEEP, which then allocates no conductivity
    # and silently makes every tissue lossless (as the first runs of cases a and b did)
    sim = mp.Simulation(cell_size=cell, resolution=res, boundary_layers=[mp.PML(DPML)],
                        default_material=material, extra_materials=media[1:], eps_averaging=False,
                        sources=sources)
    # a plane through the source per planes_of_case(); the one parallel to a launch sheet sits
    # PLANE_OFFSET downstream of it instead
    launch_ax = "xyz".index(CFG.get("axis", "x"))
    planes = {}
    for name in planes_of_case():
        ax = "xyz".index(PLANES[name][0])
        cen, sz = np.zeros(3), size.copy()
        cen[ax], sz[ax] = s[ax] + (PLANE_OFFSET if mode and ax == launch_ax else 0.0), 0.0
        planes[name] = dict(center=mp.Vector3(*cen), size=mp.Vector3(*sz))
    comps = [mp.Ex, mp.Ey, mp.Ez]
    # time-domain frames for the videos: E . r_hat (r_hat from the cranial center, ~ normal to the
    # skull, the TM mode's main component) on both planes, every 1/16 period
    sim.init_sim()
    rhat, coords, frames, times = {}, {}, {k: [] for k in planes}, []
    for name, v in planes.items():
        x, y, z, _ = sim.get_array_metadata(center=v["center"], size=v["size"])
        X, Y, Z = np.meshgrid(np.asarray(x) + mid[0], np.asarray(y) + mid[1], np.asarray(z) + mid[2], indexing="ij")
        R = np.stack([X, Y, Z], -1).squeeze() - c["center"]
        rhat[name] = R / np.linalg.norm(R, axis=-1, keepdims=True)
        coords[name] = (np.asarray(x) + mid[0], np.asarray(y) + mid[1], np.asarray(z) + mid[2])

    def record(sim_):
        times.append(sim_.meep_time() / T)
        for name, v in planes.items():
            e = [np.asarray(sim_.get_array(component=cp, center=v["center"], size=v["size"])).squeeze() for cp in comps]
            if mp.am_master():
                frames[name].append(sum(ei * rhat[name][..., k] for k, ei in enumerate(e)).astype(np.float32))

    rec = mp.at_every(T / 16, record)
    sim.run(rec, until=(periods_on - 4) * T)
    early = {k: sim.add_dft_fields(comps, fm, 0, 1, **v) for k, v in planes.items()}
    sim.run(rec, until=2 * T)
    late = {k: sim.add_dft_fields(comps, fm, 0, 1, **v) for k, v in planes.items()}
    sim.run(rec, until=2 * T)
    out = {}
    for name in planes:
        x, y, z, _ = sim.get_array_metadata(dft_cell=late[name])
        out[f"{name}_x"], out[f"{name}_y"], out[f"{name}_z"] = (np.asarray(v) + m for v, m in zip((x, y, z), mid))
        for comp, cn in zip(comps, "xyz"):
            out[f"{name}_E{cn}"] = np.asarray(sim.get_dft_array(late[name], comp, 0)) / (2 * T)
            out[f"{name}_E{cn}_early"] = np.asarray(sim.get_dft_array(early[name], comp, 0)) / (4 * T)
    if mp.am_master():
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"fields_res{res:g}.npz"
        np.savez_compressed(path, **out, src=c["src"], normal=n, center=c["center"], res=res,
                            periods=periods_on)
        print("wrote", path)
        fr = {f"{k}_frames": np.array(v) for k, v in frames.items()}
        fr.update({f"{k}_{a}": coords[k][i] for k in planes for i, a in enumerate("xyz")})
        np.savez_compressed(OUT / f"frames_res{res:g}.npz", **fr, t_periods=np.array(times))
        print("wrote", OUT / f"frames_res{res:g}.npz", {k: np.shape(v) for k, v in fr.items() if "frames" in k})


def plane_labels(c, xs, ys, zs):
    """Nearest-neighbor labels at the DFT plane points (one of xs, ys, zs is a single value)."""
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X, Y, Z], -1).squeeze()
    ii = np.rint((P - c["origin"]) / VOX).astype(int)
    for a in range(3):
        ii[..., a] = np.clip(ii[..., a], 0, c["lab"].shape[a] - 1)
    return c["lab"][ii[..., 0], ii[..., 1], ii[..., 2]]


def bone_path(L, u, v, cu, cv, su, sv):
    """Bone midline in a plane. Rays from the cranial center (cu, cv) at 0.5 deg steps; on each,
    the outermost bone segment (bone labels, gaps < 1 mm bridged) and the skin surface. Rows:
    angle from the source direction (deg), r_inner, r_mid, r_outer, thickness along the ray,
    ray angle (rad), r_skin."""
    du, dv = u[1] - u[0], v[1] - v[0]
    th0 = np.arctan2(sv - cv, su - cu)
    rows = []
    for dth in np.radians(np.arange(-180, 180, 0.5)):
        th = th0 + dth
        r = np.arange(140.0, 5.0, -0.1)
        pu, pv = cu + r * np.cos(th), cv + r * np.sin(th)
        iu, iv = np.rint((pu - u[0]) / du).astype(int), np.rint((pv - v[0]) / dv).astype(int)
        ok = (iu >= 0) & (iu < len(u)) & (iv >= 0) & (iv < len(v))
        lab = np.zeros(len(r), np.int8)
        lab[ok] = L[iu[ok], iv[ok]]
        k = np.flatnonzero(np.isin(lab, BONE))
        first = np.flatnonzero(ok)
        if len(k) == 0 or len(first) == 0 or lab[first[0]] != 0:   # ray must start in air, outside the head
            continue
        start, end = k[0], k[0]
        for kk in k[1:]:
            if kk - end > 10:
                break
            end = kk
        r_out, r_in = r[start], r[end]
        r_skin = r[np.flatnonzero(lab > 0)[0]]
        rows.append((np.degrees(dth), r_in, (r_in + r_out) / 2, r_out, r_out - r_in, th, r_skin))
    return np.array(rows)


def ernie_occiput_stack():
    """This head's occiput as measured along case A's source ray (head_data_check): skin 3.1 |
    fat 2.5 | cortical 4.6 | diploe 2.6 | cortical 2.0 mm over the venous sinus (blood half-space)."""
    from tissue_slab_modes import Layer, Stack
    from tissue_slab_modes.materials import eps
    L = lambda name, tissue, mm: Layer(name, eps(tissue, F), mm * 1e-3)
    return Stack(F, "air", 1.0, [L("skin", "skin", 3.1), L("fat", "fat", 2.5), L("cortical bone", "bone cortical", 4.6),
                                 L("cancellous bone", "bone cancellous", 2.6), L("cortical bone", "bone cortical", 2.0)],
                 "blood", eps("blood", F), "Ernie occiput")


def guided_mode(st, tissue):
    """The mode guided by one tissue: of the "bound" TM roots, the one carrying the largest share of its
    power in layers named after that tissue ("bone": the skull mode, which at 2.44 GHz is also the
    lowest-loss tissue mode while at 7.99 GHz that one rides in the skin; "csf": the CSF mode)."""
    from tissue_slab_modes import find_modes
    from tissue_slab_modes.fields import mode_profile, power_fractions
    roots = [r for r in find_modes(st, "TM", zr=(-2.6, 2.6)) if r.is_bound and r.n_eff.real > 1 and r.loss_dB_per_cm > 0]
    share = lambda r: sum(v for k, v in power_fractions(st, mode_profile(st, r.n_eff, r.gamma_top, r.gamma_bot, "TM")).items()
                          if tissue in k.lower())
    return max(roots, key=share)


STACK_TISSUE = {5: "skin", 12: "fat", 10: "muscle", 4: "bone cortical", 7: "bone cortical", 8: "bone cancellous",
                3: "csf", 6: "csf", 9: "blood", 11: "cartilage", 2: "grey matter", 1: "white matter"}


def local_stack(c):
    """Planar stack along the bone normal through the source, read from the labels: tissue runs from
    the skin inward (runs under 0.45 mm, partial-volume slivers, merged into the run above) down to
    the first brain run, which becomes the bottom half-space."""
    from tissue_slab_modes import Layer, Stack
    from tissue_slab_modes.materials import eps
    t = np.arange(40.0, -40.0, -0.05)
    ii = np.rint((c["src"] + t[:, None] * c["normal"] - c["origin"]) / VOX).astype(int)
    ii = np.clip(ii, 0, np.array(c["lab"].shape) - 1)
    runs = []
    for k in c["lab"][tuple(ii.T)]:
        nm = STACK_TISSUE.get(int(k), "air")
        if runs and runs[-1][0] == nm:
            runs[-1][1] += 0.05
        else:
            runs.append([nm, 0.05])
    while runs[0][0] == "air":
        runs.pop(0)
    merged = []
    for nm, w in runs:
        if merged and (w < 0.45 or merged[-1][0] == nm):
            merged[-1][1] += w
        else:
            merged.append([nm, w])
    k = next(i for i, (nm, _) in enumerate(merged) if nm in ("grey matter", "white matter"))
    layers = [Layer(nm, eps(nm, F), round(w, 2) * 1e-3) for nm, w in merged[:k]]
    bot = merged[k][0]
    return Stack(F, "air", 1.0, layers, bot, eps(bot, F), label=f"Ernie, case {CASE}, along the bone normal")


def with_bone(st, d_mm):
    """The stack with its bone layers scaled together to a total of d_mm."""
    from tissue_slab_modes import Layer, Stack
    tot = sum(L.thickness for L in st.layers if "bone" in L.name)
    return Stack(st.freq, st.top_name, st.eps_top,
                 [Layer(L.name, L.eps, L.thickness * d_mm * 1e-3 / tot) if "bone" in L.name else L for L in st.layers],
                 st.bot_name, st.eps_bot, label=st.label)


def temporalis_mode(st):
    """The temporalis-guided TM mode of a temple stack: at 6 mm of bone the lowest-loss "bound" root
    with 2 < Re n < 7, then the nearest root at every 0.05 mm step down to this stack's bone (a global
    root search per step, so it can cross sheets; its improper-in-air twin sits ~0.02 away). Below
    that it keeps going to find where the mode turns leaky into the brain. Returns the mode and that
    threshold (mm; None if it stays "bound" down to 1 mm)."""
    from tissue_slab_modes import find_modes
    d_loc = round(sum(L.thickness for L in st.layers if "bone" in L.name) * 1e3, 2)
    m, mode, thr = None, None, None
    for d in np.round(np.r_[np.arange(6.0, d_loc, -0.05), np.arange(d_loc, 0.99, -0.05)], 3):
        roots = [r for r in find_modes(with_bone(st, d), "TM", zr=(-2.6, 2.6))
                 if r.n_eff.real > 1 and r.loss_dB_per_cm > 0]
        prev = m
        m = (min([r for r in roots if r.is_bound and 2.0 < r.n_eff.real < 7.0], key=lambda r: r.loss_dB_per_cm)
             if prev is None else min(roots, key=lambda r: abs(r.n_eff - prev.n_eff)))
        if d == d_loc:
            mode = m
        if mode is not None and prev is not None and prev.is_bound and not m.is_bound:
            thr = d + 0.05
            break
    return mode, thr


def case_modes_1d(c):
    """The 1D mode(s) a case compares with: [(label, stack, mode)], the launched mode first, and the
    "bound"/leaky bone threshold of a temple stack (None otherwise). Skull cases: this head's occiput
    (ernie_occiput_stack) and the catalog's generic occipital stack."""
    if CFG.get("mode") == "temporalis":
        st = local_stack(c)
        m, thr = temporalis_mode(st)
        return [("this head's temple stack (along the bone normal)", st, m)], thr
    if CFG.get("mode") == "csf":
        st = local_stack(c)
        return [("this head's stack at the CSF pocket (along the bone normal)", st, guided_mode(st, "csf"))], None
    from tissue_slab_modes.materials import head_stacks
    ern, cat = ernie_occiput_stack(), head_stacks(F)["occipital"]
    return [("this head's occiput stack", ern, guided_mode(ern, "bone")),
            ("catalog's generic occipital stack", cat, guided_mode(cat, "bone"))], None


def _ramp(t):
    """0 -> 1 raised-cosine ramp for t in [0, 1]."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))


def mode_phasors(st, m):
    """A TM mode on a depth grid, as MEEP phasors. MEEP's time convention is exp(-i w t) and the
    library's exp(+j w t), so every field is conjugated; a mode travelling along +s is then
    exp(+i k0 conj(n) s). Returns depth x (mm, 0 at the skin, negative inward; from MODE_BELOW under
    the inner table to MODE_ABOVE over the skin, with raised-cosine ends), H (along the layers, across
    the propagation direction), E (along the outward normal), and the depths of the bone's outer and
    inner surfaces."""
    from tissue_slab_modes.fields import mode_profile
    p = mode_profile(st, m.n_eff, m.gamma_top, m.gamma_bot, "TM", above=MODE_ABOVE * 1e-3,
                     below=MODE_BELOW * 1e-3, npts=20001)
    x = p["x"] * 1e3
    w = _ramp((x - x[0]) / min(15.0, MODE_BELOW / 2)) * _ramp((x[-1] - x) / 5.0)
    H = np.conj(p["Hy"]) * w
    E = np.conj(m.n_eff * p["Hy"] / p["eps"]) * w
    edges = p["edges"] * 1e3
    ib = [i for i, L in enumerate(st.layers) if "bone" in L.name]
    return x, H, E, edges[ib[0]], edges[ib[-1] + 1]


def prep_sheet():
    """Huygens sheet for a mode launch on the plane through the source normal to the launch axis
    (CFG "axis", default x). That plane holds the bone normal to within a few degrees, so the mode
    runs along +axis. On each ray from the cranial center in the plane the 1D profile is laid onto
    the real layers: the soft tissue over the bone stretched to its local thickness, the bone
    stretched to the local bone thickness, and below it depth under the inner table. A Gaussian in
    angle (1/e half-width W_MODE at the bone midline) limits the sheet along the bone.
    Equivalence-principle currents for a wave leaving the sheet along +s (s = the axis, outward
    normal d, w = s x d): J = s x H = -H d, M = -s x E = -E w. For a planar stack these excite the
    mode alone, and only along +s."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = load_cache()
    modes, thr = case_modes_1d(c)
    label, st, m = modes[0]
    x, H, E, xo, xi = mode_phasors(st, m)
    lab, o = c["lab"], c["origin"]
    ax = "xyz".index(CFG.get("axis", "x"))
    iu, iv = [k for k in range(3) if k != ax]
    an, bn = "xyz"[iu], "xyz"[iv]
    L2 = np.take(lab, int(round((c["src"][ax] - o[ax]) / VOX)), axis=ax)
    us, vs = o[iu] + VOX * np.arange(lab.shape[iu]), o[iv] + VOX * np.arange(lab.shape[iv])
    cu, cv = c["center"][iu], c["center"][iv]
    path = bone_path(L2, us, vs, cu, cv, c["src"][iu], c["src"][iv])
    a = np.radians(path[:, 0])
    k = int(np.argmin(abs(a)))
    assert path[k, 0] == 0, "no bone on the source ray"
    th0, w_ang = path[k, 5], W_MODE / path[k, 2]
    # contiguous rays around the source: 0.5 deg apart, no jump > 2 mm in the bone or skin radii
    smooth = (np.diff(path[:, 0]) < 0.75) & np.all(abs(np.diff(path[:, [1, 3, 6]], axis=0)) < 2.0, axis=1)
    lo = hi = k
    while lo > 0 and smooth[lo - 1] and a[lo - 1] >= -2.5 * w_ang:
        lo -= 1
    while hi < len(a) - 1 and smooth[hi] and a[hi + 1] <= 2.5 * w_ang:
        hi += 1
    rays, ra = path[lo:hi + 1], a[lo:hi + 1]
    # sheet grid: bounding box of the sector the profile covers, inside the crop
    A, Rr = np.meshgrid(np.linspace(ra[0], ra[-1], 200), np.linspace(rays[:, 1].min() + x[0] - xi,
                                                                         rays[:, 6].max() + x[-1], 200))
    pu, pv = cu + Rr * np.cos(th0 + A), cv + Rr * np.sin(th0 + A)
    gu = np.arange(max(pu.min() - 1, us[0] + 1), min(pu.max() + 1, us[-1] - 1), VOX / 2)
    gv = np.arange(max(pv.min() - 1, vs[0] + 1), min(pv.max() + 1, vs[-1] - 1), VOX / 2)
    U, V = np.meshgrid(gu, gv, indexing="ij")
    r, th = np.hypot(U - cu, V - cv), np.arctan2(V - cv, U - cu)
    aa = (th - th0 + np.pi) % (2 * np.pi) - np.pi
    r_in, r_out, r_sk = (np.interp(aa, ra, rays[:, j]) for j in (1, 3, 6))
    xl = np.where(r >= r_sk, r - r_sk,
                  np.where(r >= r_out, xo * (r_sk - r) / (r_sk - r_out),
                           np.where(r >= r_in, xi + (r - r_in) * (xo - xi) / (r_out - r_in), xi - (r_in - r))))
    cint = lambda f: np.interp(xl, x, f.real, left=0, right=0) + 1j * np.interp(xl, x, f.imag, left=0, right=0)
    wgt = ((aa >= ra[0]) & (aa <= ra[-1])) * np.exp(-(aa / w_ang) ** 2) \
        * _ramp((aa - ra[0]) / (0.3 * w_ang)) * _ramp((ra[-1] - aa) / (0.3 * w_ang))
    Hl, El = cint(H) * wgt, cint(E) * wgt
    eu, ev = np.eye(3)[iu], np.eye(3)[iv]
    w3 = np.cross(np.eye(3)[ax], np.cos(th)[..., None] * eu + np.sin(th)[..., None] * ev)
    J = np.stack([-Hl * np.cos(th), -Hl * np.sin(th)])            # -H d, (u, v) components
    M = np.stack([-El * w3[..., iu], -El * w3[..., iv]])          # -E w
    scale = np.abs(Hl).max()
    J, M = J / scale, M / scale
    trunc_dB = 20 * np.log10(np.exp(-(min(-ra[0], ra[-1]) / w_ang) ** 2))
    print(f"{label}: " + " | ".join(f"{L.name} {L.thickness * 1e3:.2f}" for L in st.layers) + f" mm | {st.bot_name}")
    print(f"{CFG['mode']} mode n = {m.n_eff:.4f}, {m.loss_dB_per_cm:.2f} dB/cm, {m.top}/{m.bot}, |H| peak at "
          f"{x[np.argmax(abs(H))]:.1f} mm (bone {xo:.2f} to {xi:.2f}); bone normal "
          f"{np.degrees(np.arcsin(abs(c['normal'][ax]))):.1f} deg out of the sheet plane"
          + (f"; turns leaky below {thr - 0.05:.2f}-{thr:.2f} mm of bone" if thr else ""))
    print(f"sheet {len(gu)} x {len(gv)} points, rays {np.degrees(ra[0]):.1f} to {np.degrees(ra[-1]):.1f} deg "
          f"(1/e at +-{np.degrees(w_ang):.1f} deg; Gaussian cut at {trunc_dB:.0f} dB)")
    np.savez_compressed(SHEET, axis=ax, gu=gu, gv=gv, plane=c["src"][ax], J=J, M=M, n_eff=m.n_eff,
                        loss=m.loss_dB_per_cm, label=label, threshold_mm=np.nan if thr is None else thr,
                        w_ang=w_ang, rays_deg=np.degrees(ra))
    print("wrote", SHEET)
    # picture: the sheet over the labels of its plane, and the 1D profile it was laid out from
    fig, (pa, pb) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    pa.imshow((L2 > 0).T, origin="lower", extent=(us[0], us[-1], vs[0], vs[-1]), cmap="Greys", vmin=0, vmax=4,
              alpha=0.6)
    Jm = np.hypot(abs(J[0]), abs(J[1]))
    im = pa.imshow(np.ma.masked_less(20 * np.log10(Jm + 1e-12), -40).T, origin="lower", cmap="magma",
                   extent=(gu[0], gu[-1], gv[0], gv[-1]), vmin=-40, vmax=0)
    pa.contour(us, vs, np.isin(L2, BONE).T.astype(float), [0.5], colors="c", linewidths=0.6)
    pa.contour(us, vs, (L2 > 0).T.astype(float), [0.5], colors="k", linewidths=0.6)
    pa.plot(c["src"][iu], c["src"][iv], "c*", ms=10)
    pa.set_xlim(gu[0] - 10, gu[-1] + 10); pa.set_ylim(gv[0] - 10, gv[-1] + 10); pa.set_aspect("equal")
    pa.set_xlabel(f"{an} (mm)"); pa.set_ylabel(f"{bn} (mm)")
    pa.set_title(f"launch sheet in the plane {'xyz'[ax]} = {c['src'][ax]:.1f} mm through the source: |J| (dB)\n"
                 "bone cyan, skin black", fontsize=10)
    fig.colorbar(im, ax=pa, shrink=0.8).set_label("|J| of the launch sheet, dB re its peak")
    for f_, lab_, cc in ((H, "|H| (along the bone)", "C0"), (E, "|E| (normal to the bone)", "C3")):
        pb.plot(x, 20 * np.log10(abs(f_) / abs(f_).max() + 1e-12), color=cc, label=lab_)
    for e in (0.0, xo, xi):
        pb.axvline(e, color="0.5", lw=0.7, ls=":")
    pb.set_ylim(-60, 3)
    pb.set_xlabel(f"depth x (mm; 0 = skin, bone between the dotted lines at {xo:.2f} and {xi:.2f})")
    pb.set_ylabel("dB re max"); pb.legend(fontsize=8)
    pb.set_title(f"1D {CFG['mode']} mode of {label}\nat {ghz()}, n = {m.n_eff:.3f}", fontsize=10)
    fig.suptitle(CFG["title"] + " (head: SimNIBS Ernie Extended, CC BY-NC 4.0)", fontsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"case_{CASE}_launch_sheet.png", dpi=110)
    print("wrote", OUT / f"case_{CASE}_launch_sheet.png")


def sheet_sources(mp, mid, cw):
    """MEEP sources for the prep_sheet currents: J as the two in-plane E components and M as the two
    in-plane (magnetic-current) H components. amp_func positions are relative to the source center."""
    sh = np.load(SHEET)
    ax = int(sh["axis"])
    iu, iv = [k for k in range(3) if k != ax]
    gu, gv, J, M = sh["gu"], sh["gv"], sh["J"], sh["M"]
    dg = gu[1] - gu[0]
    cen, size = np.zeros(3), np.zeros(3)
    cen[ax], cen[iu], cen[iv] = float(sh["plane"]), (gu[0] + gu[-1]) / 2, (gv[0] + gv[-1]) / 2
    size[iu], size[iv] = gu[-1] - gu[0], gv[-1] - gv[0]
    an, bn = "xyz"[iu], "xyz"[iv]

    def lookup(arr):
        def amp(p):
            i = min(max(int(round((getattr(p, an) + cen[iu] - gu[0]) / dg)), 0), len(gu) - 1)
            j = min(max(int(round((getattr(p, bn) + cen[iv] - gv[0]) / dg)), 0), len(gv) - 1)
            return complex(arr[i, j])
        return amp
    Ec, Hc = (mp.Ex, mp.Ey, mp.Ez), (mp.Hx, mp.Hy, mp.Hz)
    return [mp.Source(cw, component=comp, center=mp.Vector3(*(cen - mid)), size=mp.Vector3(*size),
                      amp_func=lookup(arr))
            for comp, arr in ((Ec[iu], J[0]), (Ec[iv], J[1]), (Hc[iu], M[0]), (Hc[iv], M[1]))]


AIR = "air, 3 mm outside the skin"
CLASSES = {0: 0, 5: 1, 12: 2, 10: 2, 4: 3, 7: 3, 8: 3, 3: 4, 9: 4, 1: 5, 2: 5}   # air|skin|fat+muscle|bone|CSF+blood|brain


def analyze(res: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import RegularGridInterpolator

    c = load_cache()
    f = np.load(OUT / f"fields_res{res:g}.npz")
    modes1d, thr = case_modes_1d(c) if is_bone_case() else ([], None)
    refname = CFG["ref"]
    col_s = 2 if is_bone_case() else 6        # arc length along the bone midline (a) or the skin (b)
    s_label = "arc length along the bone midline (cm)" if is_bone_case() else "arc length along the skin (cm)"
    mode = CFG.get("launch") == "mode"
    fig, axes = plt.subplots(3, 2, figsize=(15, 17.5), gridspec_kw=dict(height_ratios=[1.15, 1, 0.75]))
    figr, raxes = plt.subplots(2, 1, figsize=(15, 9.5))
    k0 = 2 * np.pi * F / C0 / 100                       # rad/cm
    report = dict(case=CASE, res_mm=1 / res, periods=float(f["periods"]), reference=f"{refname}, 1 cm from the source")
    if modes1d:
        report["mode_1d"] = {k: dict(n_eff=f"{m.n_eff.real:.4f}{m.n_eff.imag:+.4f}j", loss_dB_per_cm=round(m.loss_dB_per_cm, 2),
                                     stack_mm=" | ".join(f"{L.name} {L.thickness * 1e3:.2f}" for L in st.layers)
                                     + f" | {st.bot_name}")
                             for k, st, m in modes1d}
    if thr is not None:
        report["mode_1d"][modes1d[0][0]]["turns_leaky_below_bone_mm"] = f"{thr - 0.05:.2f}-{thr:.2f}"
    # spreading removed before fitting a loss: a point source spreads cylindrically along the head; the
    # mode launch is a Gaussian beam of waist W_MODE in the bone (Rayleigh range at the mode's index)
    if mode:
        z_r = np.pi * (W_MODE / 10) ** 2 * modes1d[0][2].n_eff.real / (C0 / F * 100)
        spread_dB = lambda s: -5 * np.log10(1 + (s / z_r) ** 2)
        report["spreading"] = f"Gaussian beam, waist {W_MODE:g} mm, Rayleigh range {z_r:.1f} cm"
    else:
        spread_dB = lambda s: -10 * np.log10(s)
        report["spreading"] = "cylindrical"
    ribbon = {}
    launch_ax = "xyz".index(CFG.get("axis", "x"))
    for col, name in enumerate(planes_of_case()):
        _, (au, av), labels, view = PLANES[name]
        xs, ys, zs = (np.atleast_1d(f[f"{name}_{k}"]) for k in "xyz")
        E = np.sqrt(sum(abs(f[f"{name}_E{k}"]) ** 2 for k in "xyz")).squeeze()
        Ee = np.sqrt(sum(abs(f[f"{name}_E{k}_early"]) ** 2 for k in "xyz")).squeeze()
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        R = np.stack([X, Y, Z], -1).squeeze() - c["center"]
        R /= np.linalg.norm(R, axis=-1, keepdims=True)
        er = sum(np.squeeze(f[f"{name}_E{k}"]) * R[..., i] for i, k in enumerate("xyz"))   # E . r_hat phasor
        L = plane_labels(c, xs, ys, zs)
        u, v = dict(x=xs, y=ys, z=zs)[au], dict(x=xs, y=ys, z=zs)[av]
        iu, iv = "xyz".index(au), "xyz".index(av)
        su, sv = c["src"][iu], c["src"][iv]
        cu, cv = c["center"][iu], c["center"][iv]
        interp = RegularGridInterpolator((u, v), E, bounds_error=False)
        interp_e = RegularGridInterpolator((u, v), Ee, bounds_error=False)
        interp_c = lambda q: (RegularGridInterpolator((u, v), er.real, bounds_error=False)(q)
                              + 1j * RegularGridInterpolator((u, v), er.imag, bounds_error=False)(q))
        path = bone_path(L, u, v, cu, cv, su, sv)
        if mode and launch_ax in (iu, iv):              # "+" = the launch direction
            th0 = path[path[:, 0] == 0, 5][0]
            if (-np.sin(th0) if launch_ax == iu else np.cos(th0)) < 0:
                path[:, 0] *= -1
        curves, ref = {}, None
        for sgn, side in ((1, "+"), (-1, "-")):
            p = path[(np.sign(path[:, 0]) == sgn) | (path[:, 0] == 0)]
            p = p[np.argsort(abs(p[:, 0]))]
            at = lambda r: np.c_[cu + r * np.cos(p[:, 5]), cv + r * np.sin(p[:, 5])]
            spts = at(p[:, col_s])
            jump = np.flatnonzero(np.linalg.norm(np.diff(spts, axis=0), axis=1) > 5.0)
            n = jump[0] + 1 if len(jump) else len(p)        # stop where the path breaks
            p, spts = p[:n], spts[:n]
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(spts, axis=0), axis=1))] / 10   # cm
            pts = {"mid-bone": at(p[:, 2]), "1 mm inside the inner table": at(p[:, 1] - 1.0),
                   "1 mm inside the skin": at(p[:, 6] - 1.0), AIR: at(p[:, 6] + 3.0)}
            if refname == "1 mm outside the outer table":
                pts[refname] = at(p[:, 3] + 1.0)
            phase_keys = ("mid-bone", AIR) if refname in ("mid-bone", AIR) else (refname, AIR)
            curves[side] = (s, p, {k: interp(q) for k, q in pts.items()}, interp_e(pts[refname]),
                            {k: interp_c(pts[k]) for k in phase_keys})
            if sgn == 1:
                ref = curves[side][2][refname][np.argmin(abs(s - 1.0))]
        report[name] = dict(ref_E=float(ref))
        if mode:                                        # the sheet launches toward +: what leaks backward?
            sp_, vp_ = curves["+"][0], curves["+"][2][refname]
            sm_, vm_ = curves["-"][0], curves["-"][2][refname]
            ss_ = np.linspace(1.0, 3.0, 21)
            report[name]["backward_minus_forward_dB_1_to_3cm"] = round(float(np.median(
                20 * np.log10(np.interp(ss_, sm_, vm_) / np.interp(ss_, sp_, vp_)))), 1)
        # field map
        ax = axes[0, col]
        im = ax.imshow(20 * np.log10(E / ref).T, origin="lower", extent=(u[0], u[-1], v[0], v[-1]),
                       cmap="magma", vmin=-50, vmax=20)
        ax.contour(u, v, (L > 0).T.astype(float), [0.5], colors="c", linewidths=0.7)
        ax.contour(u, v, np.isin(L, BONE).T.astype(float), [0.5], colors="w", linewidths=0.6)
        ax.contour(u, v, np.isin(L, (1, 2)).T.astype(float), [0.5], colors="0.6", linewidths=0.4)
        ax.plot(su, sv, "c*", ms=12)
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_aspect("equal")
        where = (f"plane {PLANE_OFFSET / 10:g} cm downstream of the launch sheet"
                 if mode and PLANES[name][0] == "xyz"[launch_ax] else "plane through the source")
        ax.set_title(f"{name} {where}\n({view})\nskin cyan, bone white, brain gray", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8).set_label(
            f"|E|, time-averaged field strength (dB re {refname} 1 cm from the source)\n"
            "black = weakest (−50 dB or below), yellow = strongest")
        if CASE == "b" and name == "axial":             # the same spot behind the right ear
            mirror = np.array([[2 * c["center"][0] - su, sv]])
            report[name]["mirror_point_dB"] = round(float(20 * np.log10(interp(mirror)[0] / ref)), 1)
            ax.plot(*mirror[0], "co", mfc="none", ms=10)
        # decay along the path
        bx = axes[1, col]
        tw = bx.twinx()
        smax = 0.0
        for side, ls in (("+", "-"), ("-", "--")):
            s, p, vals, early, _ = curves[side]
            smax = max(smax, s[-1])
            for (k, val), cc in zip(vals.items(), ("k", "C3", "C0", "C1", "C5")):
                bx.plot(s, 20 * np.log10(val / ref), ls, color=cc, lw=1.2, label=f"{k} ({side})")
            tw.plot(s, p[:, 4], ls, color="0.6", lw=0.8)

            def slope(val, lo, hi):
                """dB/cm loss after removing the spreading (spread_dB), fitted over lo < s < hi (cm)."""
                m = (s > lo) & (s < hi) & np.isfinite(val)
                if m.sum() < 5:
                    return None
                return round(float(-np.polyfit(s[m], 20 * np.log10(val[m]) - spread_dB(s[m]), 1)[0]), 2)
            m8 = s < 8.0
            rv = vals[refname]
            report[name][side] = dict(
                path_cm=round(float(s[-1]), 1),
                dB_per_cm_after_spreading={
                    k: {f"{lo:g}-{hi:g} cm": slope(val, lo, hi) for lo, hi in ((1.0, 2.5), (2.5, 6.0), (6.0, 12.0))}
                    for k, val in vals.items()},
                dB_every_cm_from_1={k: [round(float(20 * np.log10(val[np.argmin(abs(s - x))] / ref)), 1)
                                        for x in range(1, min(int(s[-1]), 12) + 1)] for k, val in vals.items()},
                bone_mm_every_cm=[round(float(p[np.argmin(abs(s - x)), 4]), 1) for x in range(0, int(s[-1]) + 1)],
                transient_max_rel_diff_0_8cm=round(float(np.nanmax(abs(early[m8] - rv[m8]) / rv[m8])), 4))
        ss = np.linspace(0.3, 10, 100)
        mode_lines = [bx.plot(ss, -m.loss_dB_per_cm * (ss - 1) + spread_dB(ss) - spread_dB(1.0), ":", color=cc, lw=2,
                              label=f"1D mode, {lab.split(' (')[0]}: {m.loss_dB_per_cm:.2f} dB/cm + "
                                    f"{report['spreading'].split(',')[0]} spreading")[0]
                      for (lab, _, m), cc in zip(modes1d, ("C2", "C4"))]
        xmax = 10 if is_bone_case() else min(20, np.ceil(smax))
        tw.set_ylabel("bone thickness along the ray (mm, gray)")
        tw.set_ylim(0, 30)
        bx.set_xlim(0, xmax); bx.set_ylim(-50, 40)
        bx.set_xlabel(s_label)
        bx.set_ylabel(f"|E| (dB re {refname} at 1 cm)")
        curves_legend = bx.legend(handles=[h for h in bx.get_lines() if h not in mode_lines], fontsize=7,
                                  loc="upper right", ncol=2)
        if mode_lines:
            bx.legend(handles=mode_lines, fontsize=7, loc="lower left")
            bx.add_artist(curves_legend)
        bx.set_title(f"{name}: decay along the head (solid +, dashed − direction)", fontsize=10)
        # local phase index along the path: n_eff = (d phase / ds) / k0, sliding 1 cm linear fit
        cx = axes[2, col]
        for side, ls in (("+", "-"), ("-", "--")):
            s, _, _, _, cplx = curves[side]
            report[name][side]["n_eff_phase"] = {}
            for (k, z), cc in zip(cplx.items(), ("k", "C1")):
                ph = np.unwrap(np.angle(z))
                ne = np.full(len(s), np.nan)
                for i in range(len(s)):
                    m = abs(s - s[i]) < 0.5
                    if m.sum() >= 5 and s[i] > 0.5:
                        ne[i] = np.polyfit(s[m], ph[m], 1)[0] / k0
                cx.plot(s, ne, ls, color=cc, lw=1.2, label=f"{k} ({side})")
                report[name][side]["n_eff_phase"][k] = {
                    f"{lo:g}-{hi:g} cm": (round(float(np.nanmedian(ne[(s > lo) & (s < hi)])), 2)
                                          if ((s > lo) & (s < hi)).sum() else None)
                    for lo, hi in ((1.0, 2.5), (3.0, 6.0), (6.0, 12.0))}
        from tissue_slab_modes.materials import eps
        nrefs = [(1.0, "air (Norton / space wave)")] + [
            (float(np.sqrt(eps(t, F)).real), f"{name_}, Re √ε")
            for t, name_ in (("fat", "fat"), ("bone cortical", "cortical bone"), ("grey matter", "gray matter"))]
        if modes1d:
            nrefs.append((modes1d[0][2].n_eff.real, f"1D {CFG.get('mode', 'skull')} mode ({modes1d[0][0]})"))
        for nref, lab in nrefs:
            cx.axhline(nref, color="0.5", lw=0.7, ls=":")
            cx.text(xmax - 0.1, nref + 0.08, lab, ha="right", va="bottom", fontsize=7, color="0.35")
        cx.set_xlim(0, xmax); cx.set_ylim(-1, 9)
        cx.set_xlabel(s_label)
        cx.set_ylabel("local phase index (dφ/ds)/k0")
        cx.legend(fontsize=7, loc="upper left", ncol=2)
        cx.set_title(f"{name}: which wave? phase velocity along the path (E normal to the skull)", fontsize=10)
        # straightened ribbon: field vs (signed arc length, depth below the skin) along the same rays
        sm, pm = curves["-"][:2]
        sp, pp = curves["+"][:2]
        s_all = np.r_[-sm[:0:-1], sp]
        p_all = np.r_[pm[:0:-1], pp]
        depth = np.arange(-15.0, 45.01, 0.5)
        r = p_all[:, 6][:, None] - depth[None, :]
        qu = cu + r * np.cos(p_all[:, 5])[:, None]
        qv = cv + r * np.sin(p_all[:, 5])[:, None]
        q = np.stack([qu.ravel(), qv.ravel()], -1)
        er_rib = interp_c(q).reshape(r.shape)
        iu_ = np.clip(np.rint((qu - u[0]) / (u[1] - u[0])).astype(int), 0, len(u) - 1)
        iv_ = np.clip(np.rint((qv - v[0]) / (v[1] - v[0])).astype(int), 0, len(v) - 1)
        cls = np.vectorize(lambda k: CLASSES.get(int(k), 2))(L[iu_, iv_])
        ribbon[name] = dict(s=s_all, depth=depth, er=er_rib, cls=cls)
        rx = raxes[col]
        imr = rx.imshow(20 * np.log10(abs(er_rib) / ref).T, origin="upper", aspect="auto", cmap="magma",
                        extent=(s_all[0], s_all[-1], depth[-1], depth[0]), vmin=-50, vmax=20)
        rx.contour(s_all, depth, cls.T, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="w", linewidths=0.6)
        rx.axvline(0, color="c", lw=0.8, ls="--")
        rx.set_xlabel(f"signed {s_label}; source at 0")
        rx.set_ylabel("depth below the skin (mm)")
        rx.set_title(f"{name}: the head straightened. |E·r̂| in dB re {refname} 1 cm away; "
                     "lines: skin, fat/muscle, bone, CSF/blood, brain (top to bottom)", fontsize=10)
        figr.colorbar(imr, ax=rx, shrink=0.9).set_label(f"|E·r̂|, dB re {refname} 1 cm from the source")
    title = f"{CFG['title']}, {ghz()}, {1 / res:g} mm grid (head: SimNIBS Ernie Extended, CC BY-NC 4.0)"
    fig.suptitle(title, fontsize=12)
    figr.suptitle(title, fontsize=12)
    fig.tight_layout()
    figr.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"case_{CASE}_res{res:g}"
    fig.savefig(f"{stem}.png", dpi=110)
    figr.savefig(f"{stem}_ribbon.png", dpi=110)
    np.savez_compressed(f"{stem}_ribbon.npz", **{f"{k}_{a}": v for k, d in ribbon.items() for a, v in d.items()})
    Path(f"{stem}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("wrote", f"{stem}.png", f"{stem}_ribbon.png")


def videos(res: float):
    """Three videos, each as MP4 and GIF: (1) excitation, the recorded time-domain frames from source
    turn-on to steady state; (2) steady state, one period from the DFT phasor, looped; (3) the
    same steady state on the straightened head (ribbon from analyze). All show E . r_hat on a signed
    log scale (-30 to +20 dB re the case's reference point 1 cm from the source)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    c = load_cache()
    fr = np.load(OUT / f"frames_res{res:g}.npz")
    fd = np.load(OUT / f"fields_res{res:g}.npz")
    rep = json.loads((OUT / f"case_{CASE}_res{res:g}.json").read_text())
    planes = planes_of_case()
    ref = 2 * rep[planes[0]]["ref_E"]                   # time-domain peak = 2 x normalized DFT magnitude
    rib = np.load(OUT / f"case_{CASE}_res{res:g}_ribbon.npz")
    tag_case = CFG.get("tag", f"Case {CASE.upper()}")
    lo_dec, hi_dec = -1.5, 1.0                         # -30 dB .. +20 dB

    def slog(e):
        return np.sign(e) * np.clip((np.log10(np.abs(e) / ref + 1e-30) - lo_dec) / (hi_dec - lo_dec), 0, 1)

    tick_dB = (20, 0, -20)
    db = lambda v: f"{v:+d} dB" if v else "0 dB"
    ticks = [-(v / 20 - lo_dec) / (hi_dec - lo_dec) for v in tick_dB] + [0] + \
            [(v / 20 - lo_dec) / (hi_dec - lo_dec) for v in tick_dB[::-1]]
    tick_labels = [db(v) for v in tick_dB] + [f"below {20 * lo_dec:g} dB"] + [db(v) for v in tick_dB[::-1]]

    def sign_bar(cb):
        """Color depth = level in dB, hue = sign of E . r_hat (r_hat outward from the cranial center)."""
        cb.ax.set_yticklabels(tick_labels)
        cb.set_label(f"E normal to the skull: level re {CFG['ref']} 1 cm away (log scale)\n"
                     "red: pointing out of the head;  blue: pointing into the head")

    launch_ax = CFG.get("axis", "x") if CFG.get("launch") == "mode" else None
    views = [(name, PLANES[name][1],
              f"{name} plane{f' {PLANE_OFFSET / 10:g} cm downstream of the launch sheet' if PLANES[name][0] == launch_ax else ''}"
              f" ({PLANES[name][3]})") for name in planes]

    def setup(coord_src):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.0))
        ims = []
        for ax, (name, (au, av), title) in zip(axes, views):
            xs, ys, zs = (np.atleast_1d(coord_src[f"{name}_{k}"]) for k in "xyz")
            u, v = dict(x=xs, y=ys, z=zs)[au], dict(x=xs, y=ys, z=zs)[av]
            L = plane_labels(c, xs, ys, zs)
            im = ax.imshow(np.zeros((len(v), len(u))), origin="lower", extent=(u[0], u[-1], v[0], v[-1]),
                           cmap="RdBu_r", vmin=-1, vmax=1, interpolation="bilinear")
            ax.contour(u, v, (L > 0).T.astype(float), [0.5], colors="k", linewidths=0.6)
            ax.contour(u, v, np.isin(L, BONE).T.astype(float), [0.5], colors="0.35", linewidths=0.6)
            ax.contour(u, v, np.isin(L, (1, 2)).T.astype(float), [0.5], colors="0.6", linewidths=0.4,
                       linestyles=":")
            ax.plot(c["src"]["xyz".index(au)], c["src"]["xyz".index(av)], "k*", ms=9)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(f"{au} (mm)"); ax.set_ylabel(f"{av} (mm)"); ax.set_aspect("equal")
            ims.append(im)
        sign_bar(fig.colorbar(ims[-1], ax=axes, shrink=0.75, ticks=ticks))
        return fig, ims

    credit = (f"{CFG['title'].split(': ', 1)[1]}; skin black, bone gray, brain dotted.  "
              "Head: SimNIBS Ernie Extended (Van Hoornweder et al. 2024), CC BY-NC 4.0")
    T_ns = 1 / F * 1e9
    jobs = []
    # 1. excitation
    t = fr["t_periods"]
    fig, ims = setup(fr)
    sup = fig.suptitle("")
    fig.text(0.5, 0.02, credit, ha="center", fontsize=7.5)

    def draw_exc(i):
        for im, (name, _, _) in zip(ims, views):
            im.set_data(slog(fr[f"{name}_frames"][i]).T)
        sup.set_text(f"{tag_case}, {ghz()} source switched on at t = 0:  t = {t[i]:5.2f} periods = {t[i] * T_ns:5.2f} ns")
        return ims
    jobs.append((fig, draw_exc, len(t), "excitation", 24))
    # 2. steady state from the phasor
    phase = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    er = {}
    for name, _, _ in views:
        xs, ys, zs = (np.atleast_1d(fd[f"{name}_{k}"]) for k in "xyz")
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        R = np.stack([X, Y, Z], -1).squeeze() - c["center"]
        R /= np.linalg.norm(R, axis=-1, keepdims=True)
        er[name] = sum(np.squeeze(fd[f"{name}_E{k}"]) * R[..., i] for i, k in enumerate("xyz"))
    fig2, ims2 = setup(fd)
    sup2 = fig2.suptitle("")
    fig2.text(0.5, 0.02, credit, ha="center", fontsize=7.5)

    def draw_ss(i):
        ph = phase[i % len(phase)]
        for im, (name, _, _) in zip(ims2, views):
            im.set_data(slog(np.real(2 * er[name] * np.exp(-1j * ph))).T)
        sup2.set_text(f"{tag_case}, steady state at {ghz()}: one period, phase {np.degrees(ph):3.0f} deg")
        return ims2
    jobs.append((fig2, draw_ss, 4 * len(phase), "steady_state", 16))
    # 3. steady state on the straightened head: horizontal motion = along the layers, vertical = into them
    fig3, axes3 = plt.subplots(2, 1, figsize=(12.8, 7.6))
    ims3 = []
    for ax, name in zip(axes3, planes):
        s, depth, cls = rib[f"{name}_s"], rib[f"{name}_depth"], rib[f"{name}_cls"]
        ims3.append(ax.imshow(np.zeros((len(depth), len(s))), origin="upper", aspect="auto", cmap="RdBu_r",
                              vmin=-1, vmax=1, interpolation="bilinear",
                              extent=(s[0], s[-1], depth[-1], depth[0])))
        ax.contour(s, depth, cls.T, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors="k", linewidths=0.5)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_ylabel("depth below the skin (mm)")
        ax.set_title(f"{name} plane, straightened: lines are skin, fat/muscle, bone, CSF/blood, brain (top to bottom)",
                     fontsize=9)
    axes3[-1].set_xlabel("signed arc length along the head (cm); source at 0")
    sign_bar(fig3.colorbar(ims3[-1], ax=axes3, shrink=0.75, ticks=ticks))
    sup3 = fig3.suptitle("")
    fig3.text(0.5, 0.01, credit.replace("skin black, bone gray, brain dotted", "air above the top line"),
              ha="center", fontsize=7.5)

    def draw_rib(i):
        ph = phase[i % len(phase)]
        for im, name in zip(ims3, planes):
            im.set_data(slog(np.real(2 * rib[f"{name}_er"] * np.exp(-1j * ph))).T)
        sup3.set_text(f"{tag_case}, steady state on the straightened head: phase {np.degrees(ph):3.0f} deg")
        return ims3
    jobs.append((fig3, draw_rib, 4 * len(phase), "ribbon", 16))
    for fig, draw, n, tag, fps in jobs:
        anim = animation.FuncAnimation(fig, draw, frames=n, blit=False)
        mp4 = OUT / f"case_{CASE}_{tag}_res{res:g}.mp4"
        anim.save(mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2500), dpi=110)
        # GIFs play inline on GitHub but are large: the excitation GIF keeps the first 10 periods
        # (the field is steady by then) at every 2nd frame; the steady-state GIF is one period
        gif = OUT / f"case_{CASE}_{tag}_res{res:g}.gif"
        frames_g = range(0, min(n, 160), 2) if tag == "excitation" else range(len(phase))
        step = 2 if tag == "excitation" else 1
        animation.FuncAnimation(fig, draw, frames=frames_g).save(
            gif, writer=animation.PillowWriter(fps=fps // step), dpi=60 if tag == "steady_state" else 50)
        print("wrote", mp4, f"{mp4.stat().st_size / 1e6:.1f} MB;", gif, f"{gif.stat().st_size / 1e6:.1f} MB")
        if tag == "excitation":                     # filmstrip: four moments of the turn-on
            import io
            from PIL import Image
            tiles = []
            for tp in (1.5, 4.0, 7.5, 22.0):
                draw(int(np.argmin(abs(t - tp))))
                buf = io.BytesIO()
                fig.savefig(buf, dpi=75, bbox_inches="tight", pad_inches=0.1)
                tiles.append(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
            w, h = max(im.size[0] for im in tiles), max(im.size[1] for im in tiles)
            strip = Image.new("RGB", (2 * w, 2 * h), "white")
            for k, im in enumerate(tiles):
                strip.paste(im, ((k % 2) * w, (k // 2) * h))
            png = OUT / f"case_{CASE}_excitation_filmstrip_res{res:g}.png"
            strip.save(png, optimize=True)
            print("wrote", png, f"{png.stat().st_size / 1e6:.1f} MB")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "run", "analyze", "videos"])
    ap.add_argument("--case", choices=sorted(CASES), required=True)
    ap.add_argument("--res", type=float, default=2.0, help="MEEP pixels per mm")
    ap.add_argument("--periods", type=float, default=24.0, help="total run length in source periods")
    a = ap.parse_args()
    configure(a.case)
    if a.step == "prep":
        prep()
    elif a.step == "run":
        run(a.res, a.periods)
    elif a.step == "analyze":
        analyze(a.res)
    else:
        videos(a.res)
