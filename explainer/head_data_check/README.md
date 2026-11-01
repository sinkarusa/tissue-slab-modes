# Head-model check: which open dataset has the layers the modes need?

The 1D modes in this repo live in specific layers: skin | fat | temporalis muscle | cortical and cancellous bone | CSF. A 3D simulation needs a head model that resolves those layers. This folder probes two open datasets along the same rays and shows what each one actually contains.

**Use.** This repository is educational and non-commercial. The datasets below are used under their own terms for that purpose only.

## Files

| file | what it shows |
|---|---|
| `measurement_snapshot.png` | one page: rows = datasets, columns = sites (temple, occiput, behind the ear); a dimension arrow and thickness on every layer the ray crosses |
| `nyhead_rays.png`, `ernie_extended_rays.png` | per dataset: the label slice in the plane of each ray, a zoom with 1 mm ticks, and the tissue runs read along the ray |
| `ernie_extended_squama.png` | Ernie Extended temporal-bone thickness over a 187-ray fan per side, and the thinnest spot with its largest inscribed sphere |
| `head_data_check.py` | makes all of the above; downloads the data from the original sources into `data/` (git-ignored, ~770 MB) |

Run from `explainer/`:

```
uv run --with nibabel --with numpy --with scipy --with matplotlib python head_data_check/head_data_check.py
```

## Method

From the center of the brain + CSF labels, a ray is cast outward at each site. The script walks it from outside in at 0.1 mm steps, reads the nearest voxel's label, and reports each run of the same tissue as one layer. The figures resample the label volume with the same nearest-neighbor sampler, in the plane that contains the ray and the vertical, so the picture shows exactly what was measured. Run lengths are along the ray. Where the ray crosses a layer obliquely, the run is longer than the layer's true thickness.

For the temporal bone, the thickness at a ray is the diameter of the largest sphere inside the bone centered on that ray: 2 × the Euclidean distance transform, maximized along the bone segment. On a 0.5 mm grid this overestimates by up to one voxel, so a value T means a true thickness of T − 0.5 to T.

## Results

| dataset | temple (left), outside in | occiput | verdict |
|---|---|---|---|
| New York Head | lumped soft tissue 10.8 \| bone 4.6 \| CSF | lumped 8.7 \| bone 8.6 \| CSF | skin, fat and temporalis are one label; no diploë |
| **SimNIBS Ernie Extended** | skin 2.6 \| fat 2.1 \| muscle 7.2 \| bone \| CSF 2.0 | skin 3.1 \| fat 2.5 \| bone 4.6 \| spongy 2.6 \| bone 2.0 | **matches the 1D stacks; chosen for the 3D cases** |

Ernie Extended's thinnest temporal bone is 1.5–2.0 mm on the left (above the ear) and 2.5–3.0 mm on the right; the median over the fan is 4.1 mm. The left side sits at the 1.64 mm "bound"/leaky threshold of the 1D temporalis mode. This is one subject, not a population.

## Sources, licenses and citations

The code in this folder is MIT, like the rest of the repo ([`LICENSE`](../../LICENSE)). **The figures are not MIT.** They are derived from the datasets below (re-colored label slices with annotations), so each figure carries the terms of the dataset it shows. No dataset is redistributed here: `data/` is git-ignored, and the script downloads from the original sources.

**SimNIBS Ernie Extended** (`ernie_extended_rays.png`, `ernie_extended_squama.png`, bottom row of `measurement_snapshot.png`)
- Source: SimNIBS datasets page, https://simnibs.github.io/simnibs/build/html/dataset.html (download https://osf.io/6qv2z/download).
- License: **CC BY-NC 4.0**, https://creativecommons.org/licenses/by-nc/4.0/ ("You can share and adapt the dataset if you do not use it for any commercial purposes", dataset readme). The derived figures are shared under the same license. Changes made: labels re-colored and merged into a common scheme (compact bone and the generic bone label shown together), slices resampled along probe rays, annotations added.
- Cite: Van Hoornweder S, Cappozzo V, De Herde L, Puonti O, Siebner HR, Meesen RLJ, Thielscher A (2024), "Head and shoulders—The impact of an extended head model on the simulation and optimization of transcranial electric stimulation," *Imaging Neuroscience*, doi:10.1162/imag_a_00379.

**New York Head (ICBM-NY)** (`nyhead_rays.png`, top row of `measurement_snapshot.png`)
- Source: https://www.parralab.org/nyhead/ (symmetric segmentation, NIfTI).
- License (https://www.parralab.org/nyhead/license): "Copyright (C) 2015 Yu Huang (Andy), Lucas C. Parra and Stefan Haufe. Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby granted, provided that the above copyright notice appear in all copies." The dataset's documentation PDF also carries a GNU GPL v3 notice.
- Cite: Huang Y, Parra LC, Haufe S (2016), "The New York Head — A precise standardized volume conductor model for EEG source localization and tES targeting," *NeuroImage*, doi:10.1016/j.neuroimage.2015.12.019.

**Brain Mesh Library** (NeuroJSON, https://github.com/NeuroJSON/BrainMeshLibrary), CC BY-SA 4.0: inspected but not plotted. Its conversion script (`tojmesh.m`) defines five labels: Scalp, Skull, CSF, WM, GM. That merges the layers this check is about.
