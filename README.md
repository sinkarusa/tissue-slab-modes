# tissue-slab-modes

`tissue_slab_modes` finds and classifies electromagnetic modes in lossy planar multilayer stacks, including layers of human tissue, and reports their fields, loss, power distribution and source coupling.

**Start with the illustrated explainer:** [How can fat guide a radio wave?](explainer/README.md). Fat cannot guide by the optical-fiber rule; the post shows how partial grazing reflection can nevertheless confine a lossy radio mode for a few centimeters.

## Method

The argument-principle method (APM) locates every TM or TE root in a search region and places it on one of the dispersion function's four Riemann sheets, so bound, leaky and improper modes come from one search. Tissue permittivities follow the Cole-Cole model of Gabriel et al. (1996). The package also gives mode profiles, power in each layer and each pole's residue in the TM reflection coefficient.

The dispersion function F depends on the normalized transverse decay constants γ_t = √(n² − ε_t) and γ_b = √(n² − ε_b) of the two half-spaces, so as a function of the effective index n it lives on a four-sheeted Riemann surface (Smith, Houde-Walter and Forbes 1992). Since γ_t² − γ_b² = ε_b − ε_t = C, the substitution γ_t = (√C/2)(t + 1/t), γ_b = (√C/2)(t − 1/t) makes F single-valued in t, and one rectangle in z = log t covers all four sheets.

The APM counts the zeros in a rectangle from the change of arg F along its edges, splits rectangles that hold more than one zero, and locates each zero from contour moments (Delves and Lyness 1967; Anemogiannis and Glytsis 1992). The edges are refined until every phase step is small, and a split's finer count overrides its parent's, which guards the count against aliasing. Roots are then polished with mpmath. Transfer matrices are rescaled by positive real factors to avoid overflow in thick lossy layers, which leaves arg F unchanged. The code began as the author's Mathematica notebook, which builds on Şükrü Ekin Kocabaş's argument-principle notebook for metal-insulator-metal waveguides ([`ModalShapes_Plasmon.nb`](https://kocabas.gitlab.io/publications/code/ModalShapes_Plasmon.nb); Kocabaş, Veronis, Miller and Fan 2009).

## Install

```bash
pip install -e .              # numpy, scipy, mpmath, matplotlib
pip install -e ".[check]"     # also pytest
```

`meep/common.py` also needs MEEP (conda-forge package `pymeep`); the package and the tests do not.

## Quick example

```python
from tissue_slab_modes import Layer, Stack, find_modes
from tissue_slab_modes.materials import eps
from tissue_slab_modes.sommerfeld import residue_R

f = 2.45e9                                          # Hz
st = Stack(f, "air", 1.0, [Layer("skin", eps("skin", f), 2e-3), Layer("fat", eps("fat", f), 25e-3)],
           "muscle", eps("muscle", f))              # air | skin 2 mm | fat 25 mm | muscle
modes = find_modes(st, "TM")                        # 34 roots on the four sheets
fat = min((m for m in modes if m.is_bound and m.loss_dB_per_cm > 0), key=lambda m: m.loss_dB_per_cm)
print(fat)
print(f"{residue_R(fat.n_eff, st):.4f}")
```

```
TM n_eff =   2.1645  -0.3421j   loss =    1.526 dB/cm   sheet(top/bot) = P/P
0.0176+0.0216j
```

`sheet` marks the top and bottom half-spaces as proper (P, the field decays away from the stack) or improper (I); P/P is a bound mode, and one improper side makes a mode leaky. This mode carries 94 % of its power in the fat (`fields.power_fractions`).

## Layout

| path | contents |
|---|---|
| `tissue_slab_modes/stack.py` | `Layer`, `Stack`: top half-space, layers, bottom half-space |
| `tissue_slab_modes/dispersion.py` | dispersion function, t-plane, sheet classification |
| `tissue_slab_modes/apm.py` | argument-principle root finder |
| `tissue_slab_modes/modes.py` | `find_modes`, `Mode`, mode kinds |
| `tissue_slab_modes/fields.py` | mode profiles, power per layer |
| `tissue_slab_modes/loss_guiding.py` | symmetric slabs, loss- and gain-guiding thresholds |
| `tissue_slab_modes/sommerfeld.py` | reflection coefficients, pole residues, dipoles over a stack |
| `tissue_slab_modes/materials.py` | Gabriel permittivities, head and body stacks with sourced thicknesses |
| `tissue_slab_modes/plotting.py` | figure palette and helpers |
| `tests/test_solver.py` | analytic limits and regression tests |
| `meep/common.py` | MEEP material mapping (D-conductivity) and planar layer blocks |
| `docs/REFERENCES.md` | references cited in the code |
| `explainer/` | the post, its scripts, figures and results |

## Tests

```bash
python -m pytest -q
```

The 14 tests compare the solver with closed forms and published values: symmetric-slab TM and TE roots, dipoles over a perfect conductor (Balanis), the Friis link, the pole term against the full Sommerfeld integral, Gabriel permittivities against IT'IS values, loss-guiding thresholds, and the mode of a 25 mm fat layer against the ≈1.5 dB/cm of Asan et al. (2017). Others check the overflow scaling, the root residuals and a zero count that an earlier version aliased. They take about 6 s.

## Tested with

Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1, mpmath 1.4.1, Matplotlib 3.11.2 and pytest 9.1.1 on macOS 26.5 (arm64); `meep/common.py` with MEEP 1.35.0 from conda-forge.

## License

The code is released under the MIT License ([`LICENSE`](LICENSE)). Figures and videos in `explainer/` that are derived from the SimNIBS Ernie Extended head model are licensed CC BY-NC 4.0; [`explainer/README.md`](explainer/README.md) (Data and licenses) lists them.
