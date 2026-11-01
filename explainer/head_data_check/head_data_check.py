"""Which open head model resolves the layers the tissue modes need? Visual evidence.

For each dataset and probe ray: (1) the label slice in the plane that contains the ray and the
vertical, with the ray drawn; (2) a zoom where the ray crosses scalp and skull, 1 mm ticks;
(3) the tissue runs read along the ray. The image and the runs use the same nearest-neighbor
sampler, so what you see is exactly what was measured. Also: a one-page snapshot with a dimension
arrow per layer, and an Ernie Extended temporal-bone thickness map (largest sphere inside the bone
along each ray, 2 x Euclidean distance transform).

Run (downloads ~770 MB once into ./data, which is git-ignored; see README.md for licenses):
    uv run --with nibabel --with numpy --with scipy --with matplotlib \
        python head_data_check/head_data_check.py
"""
import pathlib
import urllib.request
import zipfile

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Circle
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
NYHEAD_URL = "https://www.parralab.org/nyhead/NYhead_segmentations.zip"
ERNIE_URL = "https://osf.io/6qv2z/download"
ERNIE_MEMBERS = ("ErnieExtended/readme.txt",
                 "ErnieExtended/m2m_ernie_extended/label_prep/tissue_labeling_upsampled.nii.gz")


def fetch():
    """Download each dataset from its original source unless already cached in ./data."""
    DATA.mkdir(exist_ok=True)
    ny = DATA / "nyhead"
    if not (ny / "skin.nii").exists():
        print("downloading New York Head segmentation (4.7 MB)...", flush=True)
        z = DATA / "nyhead.zip"
        urllib.request.urlretrieve(NYHEAD_URL, z)
        zipfile.ZipFile(z).extractall(ny)
        z.unlink()
    ee = DATA / "ernie_extended"
    if not (ee / "tissue_labeling_upsampled.nii.gz").exists():
        print("downloading SimNIBS Ernie Extended (650 MB zip; keeps 2 files)...", flush=True)
        z = DATA / "ernie_extended.zip"
        urllib.request.urlretrieve(ERNIE_URL, z)
        ee.mkdir(exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for m in ERNIE_MEMBERS:
                (ee / pathlib.Path(m).name).write_bytes(zf.read(m))
        z.unlink()


# common label scheme
C = [("background", "#ffffff"), ("skin / scalp", "#f4a582"), ("fat", "#ffe680"),
     ("muscle", "#b2182b"), ("lumped soft tissue", "#e7298a"), ("bone (compact / single label)", "#6e6e6e"),
     ("spongy bone / marrow", "#a6761d"), ("dura", "#1b9e77"), ("CSF", "#4393c3"),
     ("gray matter", "#c2a5cf"), ("white matter", "#efe6f2"), ("connective", "#fdbf6f"),
     ("blood / vessels", "#542788"), ("eye", "#66c2a5"), ("cartilage", "#a6d854"), ("air cavity", "#d0d0d0")]
CMAP = ListedColormap([c for _, c in C])
NAME = [n for n, _ in C]
BRAIN = (8, 9, 10)  # CSF, gray, white -> cranial center


def load_nyhead():
    order = [("skin", 4), ("bone", 5), ("csf", 8), ("gray", 9), ("white", 10), ("air", 15)]
    lab = None
    for f, code in order:
        im = nib.load(DATA / "nyhead" / f"{f}.nii")
        m = np.asarray(im.dataobj) > 0.5
        lab = np.zeros(m.shape, np.int8) if lab is None else lab
        lab[m] = code
    return "New York Head (0.5 mm)", lab, im.affine


def load_ernie():
    im = nib.load(DATA / "ernie_extended" / "tissue_labeling_upsampled.nii.gz")
    raw = np.asarray(im.dataobj).astype(np.int16)
    m = {1: 10, 2: 9, 3: 8, 4: 5, 5: 1, 6: 13, 7: 5, 8: 6, 9: 12, 10: 3, 11: 14, 12: 2}
    lab = np.zeros(raw.shape, np.int8)
    for k, v in m.items():
        lab[raw == k] = v
    return "SimNIBS Ernie Extended (MRI+CT, 0.5 mm)", lab, im.affine


class Head:
    def __init__(self, title, lab, aff):
        self.title, self.lab, self.aff, self.inv = title, lab, aff, np.linalg.inv(aff)
        idx = np.argwhere(np.isin(lab, BRAIN))
        self.cen = (aff @ np.append(idx.mean(0), 1))[:3]

    def sample(self, pts):
        """Nearest-neighbor labels at world points (..., 3)."""
        p = pts.reshape(-1, 3)
        ijk = np.rint(p @ self.inv[:3, :3].T + self.inv[:3, 3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < self.lab.shape), axis=1)
        out = np.zeros(len(p), np.int8)
        out[ok] = self.lab[tuple(ijk[ok].T)]
        return out.reshape(pts.shape[:-1])

    def frame(self, d):
        u = np.asarray(d, float); u /= np.linalg.norm(u)
        v = np.array([0, 0, 1.0]) - u[2] * u; v /= np.linalg.norm(v)
        return u, v

    def plane(self, d, a0, a1, b0, b1, step):
        u, v = self.frame(d)
        A, B = np.meshgrid(np.arange(a0, a1, step), np.arange(b0, b1, step))
        pts = self.cen + A[..., None] * u + B[..., None] * v
        return self.sample(pts), (a0, a1, b0, b1)

    def runs(self, d, L=120.0, ds=0.1):
        """Walk inward from outside along the ray; return [(label, start_a, length)] (a = mm from center)."""
        u, _ = self.frame(d)
        a = np.arange(L, 0, -ds)
        labs = self.sample(self.cen + a[:, None] * u)
        out, start = [], 0
        for i in range(1, len(a) + 1):
            if i == len(a) or labs[i] != labs[start]:
                if labs[start] != 0:
                    out.append((int(labs[start]), a[start], (i - start) * ds))
                start = i
        return out


def ray_figure(head, rays, fname):
    fig, axes = plt.subplots(len(rays), 3, figsize=(17, 5.2 * len(rays)),
                             gridspec_kw=dict(width_ratios=[1.1, 1.0, 0.9]))
    used = set()
    for row, (rname, d) in zip(axes, rays):
        runs = head.runs(d)
        a_in = runs[0][1]  # outer surface, mm from center along the ray
        # 1. overview plane
        img, ext = head.plane(d, -100, 115, -100, 100, 0.5)
        used |= set(np.unique(img).tolist())
        ax = row[0]
        ax.imshow(img, origin="lower", extent=ext, cmap=CMAP, vmin=0, vmax=len(C) - 1, interpolation="nearest")
        ax.plot([0, 115], [0, 0], "k-", lw=1.2)
        ax.plot(0, 0, "k+", ms=10)
        ax.add_patch(plt.Rectangle((a_in - 26, -8), 30, 16, fill=False, ec="k", lw=1, ls="--"))
        ax.set_title(f"{rname}: plane containing the ray (black) and vertical\n"
                     f"ray direction (R, A, S) = {tuple(round(float(v), 2) for v in d)} from cranial center (+)",
                     fontsize=10)
        ax.set_xlabel("mm along ray"); ax.set_ylabel("mm, up in plane")
        # 2. zoom
        img, ext = head.plane(d, a_in - 26, a_in + 4, -8, 8, 0.1)
        used |= set(np.unique(img).tolist())
        ax = row[1]
        ax.imshow(img, origin="lower", extent=ext, cmap=CMAP, vmin=0, vmax=len(C) - 1, interpolation="nearest")
        ax.plot([a_in - 26, a_in + 4], [0, 0], "k-", lw=1)
        for t in np.arange(np.ceil(a_in - 26), a_in + 4, 1.0):
            ax.plot([t, t], [-0.4, 0.4], "k-", lw=0.8)
        for lab, a_start, length in runs[:7]:
            mid = a_start - length / 2
            if a_start < a_in - 26:
                break
            ax.annotate(f"{length:.1f}", (mid, 0), xytext=(mid, 5.5), ha="center", fontsize=8,
                        arrowprops=dict(arrowstyle="-", lw=0.5))
        ax.set_title("zoom (dashed box): ray with 1 mm ticks; numbers = run length (mm)", fontsize=10)
        ax.set_xlabel("mm along ray (outer surface at right)")
        # 3. runs bar
        ax = row[2]
        x = 0.0
        for lab, a_start, length in runs:
            if x >= 30:
                break
            ax.barh(0, length, left=x, color=C[lab][1], ec="k", lw=0.5)
            ax.text(x + length / 2, 0.45, f"{NAME[lab].split(' (')[0]}\n{length:.1f} mm", ha="center",
                    va="bottom", fontsize=7.5, rotation=60)
            x += length
        ax.set_ylim(-0.6, 2.6); ax.set_yticks([]); ax.set_xlim(0, 30)
        ax.set_xlabel("depth from skin along the ray (mm)")
        ax.set_title("what the measurement reads (outside -> in)", fontsize=10)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.legend(handles=[Patch(color=C[k][1], label=C[k][0]) for k in sorted(used) if k],
               loc="lower center", ncol=8, fontsize=9, frameon=False)
    fig.suptitle(head.title, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print("wrote", fname)


def squama_figure(head, fname, vox=0.5):
    bone = np.isin(head.lab, (5, 6))
    edt = ndi.distance_transform_edt(bone) * vox
    els, azs = np.linspace(-0.05, 0.45, 11), np.linspace(-0.25, 0.55, 17)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for col, sx in enumerate((-1.0, 1.0)):
        T = np.full((len(els), len(azs)), np.nan)
        best = None
        for i, el in enumerate(els):
            for j, az in enumerate(azs):
                d = (sx, az, el)
                u, _ = head.frame(d)
                a = np.arange(120, 0, -0.1)
                pts = head.cen + a[:, None] * u
                ijk = np.rint(pts @ head.inv[:3, :3].T + head.inv[:3, 3]).astype(int)
                b = bone[tuple(ijk.T)]
                if not b.any():
                    continue
                first = np.argmax(b)
                end = first + np.argmax(~b[first:]) if (~b[first:]).any() else len(b)
                e = edt[tuple(ijk[first:end].T)]
                T[i, j] = 2 * e.max()
                k = first + int(np.argmax(e))
                if best is None or T[i, j] < best[0]:
                    best = (T[i, j], d, a[k], e.max(), (end - first) * 0.1)
        ax = axes[0, col]
        im = ax.imshow(T, origin="lower", cmap="viridis", vmin=1.5, vmax=7,
                       extent=(azs[0] - 0.025, azs[-1] + 0.025, els[0] - 0.025, els[-1] + 0.025), aspect="auto")
        ax.plot(best[1][1], best[1][2], "r*", ms=14)
        plt.colorbar(im, ax=ax, label="2 x EDT (mm): inscribed-sphere bone thickness")
        ax.set_xlabel("ray direction: anterior component (y)"); ax.set_ylabel("superior component (z)")
        ax.set_title(f"{'left' if sx < 0 else 'right'} temporal fan, 187 rays; red star = thinnest "
                     f"({best[0]:.1f} mm; true value {best[0]-vox:.1f}-{best[0]:.1f} mm, voxel bias)", fontsize=10)
        # zoom at thinnest ray with the inscribed circle
        tb, d, a_c, r, run = best
        img, ext = head.plane(d, a_c - 14, a_c + 14, -10, 10, 0.1)
        ax = axes[1, col]
        ax.imshow(img, origin="lower", extent=ext, cmap=CMAP, vmin=0, vmax=len(C) - 1, interpolation="nearest")
        ax.plot([a_c - 14, a_c + 14], [0, 0], "k-", lw=1)
        ax.add_patch(Circle((a_c, 0), r, fill=False, ec="r", lw=1.5))
        ax.set_title(f"thinnest ray, plane view: red circle = largest sphere inside the bone (r {r:.2f} mm)\n"
                     f"oblique bone run along the ray {run:.1f} mm", fontsize=10)
        ax.set_xlabel("mm along ray")
    fig.suptitle(f"{head.title}: temporal bone thickness", fontsize=14)
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print("wrote", fname)


def snapshot_figure(heads, rays, fname):
    """One page: rows = datasets, columns = sites; each cell is a zoom across scalp and skull with
    a dimension arrow and thickness label for every layer the ray crosses."""
    fig, axes = plt.subplots(len(heads), len(rays), figsize=(6.2 * len(rays), 4.6 * len(heads)))
    used = set()
    for r, h in enumerate(heads):
        for c, (rname, d) in enumerate(rays):
            ax = axes[r, c]
            runs = h.runs(d)
            a_in = runs[0][1]
            lo, hi = a_in - 22, a_in + 3
            img, ext = h.plane(d, lo, hi, -7, 7, 0.1)
            used |= set(np.unique(img).tolist())
            ax.imshow(img, origin="lower", extent=ext, cmap=CMAP, vmin=0, vmax=len(C) - 1,
                      interpolation="nearest")
            ax.plot([lo, hi], [0, 0], "k-", lw=0.8)
            for k, (lab, a_start, length) in enumerate(runs):
                a_end = a_start - length
                if a_start <= lo:
                    break
                a_end = max(a_end, lo)
                y = 2.2 if k % 2 == 0 else -2.2
                ax.annotate("", (a_end, y), (a_start, y),
                            arrowprops=dict(arrowstyle="<->", lw=1.0, color="k", shrinkA=0, shrinkB=0))
                for a in (a_start, a_end):
                    ax.plot([a, a], [0, y], "k:", lw=0.6)
                ax.text((a_start + a_end) / 2, y + (0.5 if y > 0 else -0.5),
                        f"{NAME[lab].split(' (')[0].split(' /')[0]}\n{length:.1f} mm", ha="center",
                        va="bottom" if y > 0 else "top", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
            ax.set_xlim(lo, hi)
            ax.set_yticks([])
            ax.set_xlabel("mm along the ray (skin at right)", fontsize=8)
            if r == 0:
                ax.set_title(rname, fontsize=12)
            if c == 0:
                ax.set_ylabel(h.title.replace(" (", "\n("), fontsize=10)
    fig.legend(handles=[Patch(color=C[k][1], label=C[k][0]) for k in sorted(used) if k],
               loc="lower center", ncol=8, fontsize=9, frameon=False)
    fig.suptitle("Layer thickness along each probe ray (outside -> in). Label slices from each "
                 "dataset; see README.md for sources and licenses.", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print("wrote", fname)


RAYS = [("temple (left)", (-1.0, 0.25, 0.0)), ("occiput", (0.0, -1.0, 0.2)),
        ("behind the left ear", (-0.8, -0.8, -0.1))]

if __name__ == "__main__":
    fetch()
    heads = []
    for loader, stem in ((load_nyhead, "nyhead"), (load_ernie, "ernie_extended")):
        h = Head(*loader())
        heads.append(h)
        print(h.title, "cranial center", h.cen.round(1))
        for rname, d in RAYS:
            print(f"   {rname:20s}", [(NAME[l].split(' (')[0], round(L, 1)) for l, _, L in h.runs(d)[:8]])
        ray_figure(h, RAYS, HERE / f"{stem}_rays.png")
    snapshot_figure(heads, RAYS, HERE / "measurement_snapshot.png")
    squama_figure(heads[-1], HERE / "ernie_extended_squama.png")
