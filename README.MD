# Graphene Quantum Dot Defect Analysis

A computational-chemistry toolkit for studying structural defects, substitutional
doping, and edge functionalization in graphene quantum dots (GQDs) and small 2D
carbon nanostructures. Energies are evaluated either with **GPAW** (DFT, PBE) on
Linux or with the **SevenNet** machine-learning interatomic potential on Windows;
the calculator is chosen automatically from the operating system.

## What it does

- **Vacancy / divacancy / Stone-Wales mapping** — systematically scan every site of
  a GQD and produce a formation-energy map (`map.py`).
- **Single-defect application** — apply a chosen vacancy, divacancy, split-vacancy,
  or Stone-Wales defect to one bond and track the core during relaxation
  (`GQD_basic_defects.py`).
- **Pairwise formation-energy comparisons** — compare a pristine structure against
  one or more pre-built defect / doped / functionalized structures
  (`GQD_defects_…`, `GQD_dopants_…`, `GQD_functionalization_mols_energy_comparison.py`).
- **DFT electronic structure** (GPAW only) — HOMO/LUMO and gap, electron density,
  a single-particle absorption spectrum, dipole moment, and vibrational modes
  (`gpaw_analysis.py`).

## Project structure

```
.
├── settings.py                                      # central config: plot style, DPI, FMAX, BFGS_MAXSTEP
├── config.py                                        # molecule database (paths + cell sizes)
├── utils.py                                         # transforms, formation-energy formula, μ helpers, MOL/XYZ I/O
├── GQD_basic_defects.py                             # setup_calculator() + single-defect driver
├── map.py                                           # systematic vacancy/divacancy/Stone-Wales mapping
├── ring_overlay.py                                  # non-hexagonal ring overlay figures (used by map.py + defect comparison)
├── gpaw_analysis.py                                 # GPAWAnalyzer class + analyze_molecules() batch driver
├── GQD_defects_mols_energy_comparison.py            # pristine vs pre-made defect
├── GQD_dopants_mols_energy_comparison.py            # pristine vs doped (legacy) + automated site scan
├── GQD_functionalization_mols_energy_comparison.py  # pristine vs functionalized
└── data/                                            # input structures (.mol with bonds, .xyz)
```

`setup_calculator()` lives in `GQD_basic_defects.py` and is the single calculator
factory imported by the other scripts.

## Installation

Create an environment and install ASE plus one calculator backend.

```bash
conda create --name gqd_defects python=3.10
conda activate gqd_defects
pip install ase matplotlib scipy numpy networkx
```

**Linux — GPAW (DFT):**
```bash
pip install gpaw
gpaw install-data <datasets_dir>
```

**Windows — SevenNet (ML potential):**
```bash
# CPU PyTorch is sufficient
pip install sevenn
```
The SevenNet model used is `7net-omni-i12` with `modal='mpa'`. `networkx` is only
needed for the ring-overlay figures.

## Configuration

### settings.py — single source of truth

All relaxation constants and plot styling live here and are imported everywhere
else (so `from GQD_basic_defects import FMAX, BFGS_MAXSTEP` still resolves to these
values):

| Constant | Value | Meaning |
|---|---|---|
| `FMAX` | `0.05` | force-convergence criterion [eV/Å] |
| `BFGS_MAXSTEP` | `0.1` | max BFGS step [Å] (smaller than the ASE default to keep GPAW stable on defects) |
| `USE_SAVED_RELAXED` | `True` | reuse a cached relaxed structure when present |
| `RELAXED_STRUCTURES_DIR` | `"relaxed_structures"` | cache directory |

Importing `settings` also applies the project matplotlib style (Times New Roman,
publication DPI). Call `settings.print_settings()` to dump the active configuration.

### Caching of expensive results (`cache.py`)

Relaxations and reference-energy calculations dominate runtime, and many repeat
across runs, so two kinds of result are cached under `relaxed_structures/`, keyed
by the active calculator (so GPAW and SevenNet never share an energy scale):

- **Chemical potentials** → `relaxed_structures/chemical_potentials.json`. μ_C
  (graphene), μ_H (½·E(H₂)) and μ_X (½·E(X₂)) depend only on the calculator and
  `FMAX`, so the graphene/H₂/diatomic relaxations run at most once per calculator
  and are reused by every script and every run.
- **Relaxed structures** → `relaxed_structures/relaxed_<key>_<calc>_fmax_<f>_cell_<cell>.xyz`.
  The pristine reference and the pre-made config-key structures in the comparison
  scripts (and the base structure in `map.py` / automated doping) are cached and
  reused. Per-site relaxations (each vacancy/divacancy/Stone-Wales/doping site)
  are **not** cached.

Every cache is verified before reuse — a structure is reused only if its max force
is still below `FMAX` and its composition matches; otherwise it is re-relaxed (from
the cached geometry when available). So a cache hit can never return a stale or
under-converged result. Delete `relaxed_structures/` to force everything to
recompute. The structure-cache filename matches the `USE_SAVED_RELAXED` scheme in
`GQD_basic_defects.py`, so those caches are shared.

### config.py — molecule database

```python
molecules_data = {
    "Your_Molecule": {
        "path": os.path.join("data", "your_molecule.mol"),
        "cell": 30,   # cubic cell edge in Å (rule: ≥ max molecular extent + 20 Å)
    },
}
```
`.mol` (V2000) inputs keep their bond block, which is preserved through relaxation
and written back into the output `.mol` files.

## Usage

Each script is configured by editing the constants near its `__main__` / `main()`
block, then run directly.

### Systematic defect mapping — `map.py`

```python
molecule_name = "QD_6"   # key from config.py
defect_type   = "all"    # 'vacancy', 'divacancy', 'stw', or 'all'
show_atom_idx = True
DEBUG_MODE    = False     # True → just open the structure in the ASE viewer
```
For vacancy/divacancy, any H bonded to a removed carbon is removed with it and
accounted for via μ_H. Output goes to
`vacancy_map_<mol>_cell_size_<n>_fmax_<f>_<calc>/` (and `divacancy_map_…`,
`stw_map_…`), each containing:
- `formation_energy_map.png` / `divacancy_energy_map.png` / `stw_energy_map.png`
- `energy_distribution.png` + `energy_distribution.txt` (histogram + Gaussian fit)
- `summary.txt`, `energy_map.txt`, `formation_energies.npz`
- `relaxed_structures_xyz/`, `relaxed_structures_mol/`, `structure_images/`,
  `ring_overlays/`

### Single-defect application — `GQD_basic_defects.py`

```python
molecule_name = "input_pyrene_C16H10"
axis_atoms    = (0, 1)     # two atom indices defining the defect bond/site
ENABLE_VACANCY       = True
ENABLE_DIVACANCY     = False
ENABLE_SPLIT_VACANCY = False
ENABLE_STONE_WALES   = False
```
The pristine structure is relaxed once (and cached under
`relaxed_structures/` when `USE_SAVED_RELAXED`), each enabled defect is applied and
relaxed, and the core region is tracked into `experiments/<task>_<timestamp>/`.

### Pairwise comparisons — `GQD_{defects,dopants,functionalization}_mols_energy_comparison.py`

Compare a pristine structure to one or more pre-built modified structures (all
registered in `config.py`). The pristine reference is relaxed **once** and shared
across every comparison. Example (functionalization):

```python
pristine_key  = "GQD_HEX_2_2"
modified_keys = ["GQD_HEX_2_2_func_all"]
```
Outputs land in `legacy_func_<pristine>_vs_<modified>_<calc>/` (analogously
`legacy_defect_…`, `legacy_doping_…`) and contain the copied inputs, relaxed
XYZ/MOL structures, and a `*_energy_analysis.txt` with the full per-element
breakdown.

`GQD_dopants_mols_energy_comparison.py` additionally has an **automated** mode
(`mode = "automated"`): `DopingFormationCalculator.analyze_doping_sites()` scans
every substitution site of a GQD for one dopant and writes a doping energy map to
`doping_map_<mol>_<dopant>_<calc>/`.

### GPAW electronic structure — `gpaw_analysis.py` (Linux only)

```python
from gpaw_analysis import analyze_molecules

analyze_molecules(["Coronene"], xc='PBE', mode_cutoff=500,
                  include_optical=True, include_phonons=False)
```
or use the class directly:
```python
from gpaw_analysis import GPAWAnalyzer
analyzer = GPAWAnalyzer("Coronene", xc='PBE', mode_cutoff=300, spinpol=True)
results  = analyzer.run_full_analysis(include_optical=True, include_phonons=False)
print(f"Band gap: {results['homo_lumo']['gap']:.3f} eV")
```
Each molecule writes to its own `gpaw_results_<name>/`: relaxed structure,
`homo_lumo.txt` + `energy_levels.png`, `density.cube` + `density_slice.png`,
`simple_spectrum.txt`/`absorption_spectrum.png`, `dipole_moment.txt`, optional
phonon files, and `analysis_summary.txt`. The absorption spectrum is a fast
single-particle (Kohn-Sham) approximation, not full TDDFT.

## Formation-energy conventions

**Defects (`utils.calculate_formation_energy`, used by `map.py` and the defect
comparison):**

```
n_C_removed == 0 (Stone-Wales):  E_f = E_defect − E_pristine
n_C_removed ≥ 1  (V / VV):        E_f = E_defect − E_pristine + n_C·μ_C + n_H·μ_H
```

**Doping / functionalization (per-element composition difference):**

```
E_f = E_modified − E_pristine − Σ_i Δn_i · μ_i        (Δn_i = n_i(mod) − n_i(pristine))
```
An added atom (Δn > 0) is drawn from a reservoir at cost μ and lowers E_f by μ; a
removed atom raises it. For a substitutional N→C swap this reduces to the familiar
`E_doped − E_pristine + μ_C − μ_N`.

### Chemical potentials

Computed dynamically with the **active calculator** so the energy scale matches:
- **μ_C** = energy/atom of a relaxed 3×3 graphene supercell (`utils.calculate_mu_C`,
  the K-point folds to Γ so `kpts=(1,1,1)` samples the Dirac cone correctly).
- **μ_H** = ½·E(H₂) (`utils.calculate_mu_H`).
- **μ_N, μ_O, μ_F, μ_Cl** = ½·E(X₂) from the relaxed gas-phase diatomic.

Elements with no diatomic/graphene reference (**B, P, S, Li**) fall back to
hard-coded literature values; these are on a different energy scale and are flagged
loudly at runtime — supply a proper bulk reference for publication-grade numbers.
The automated doping mode uses a separate hard-coded `chemical_potentials` table
(with uncertainties) instead of the dynamic values.

## Calculator settings

**GPAW (`setup_calculator`, Linux):** PBE, `mode=PW(400)`, `kpts=(1,1,1)`,
`symmetry='off'`, `spinpol=True`, `FermiDirac(0.05)`, Pulay mixer, `maxiter=500`,
convergence `energy=1e-5`, `density=1e-5`, `eigenstates=1e-7`. `GPAWAnalyzer` uses
its own looser convergence and a configurable `mode_cutoff` (300 default, 500 in the
batch driver).

**SevenNet (Windows / other):** `7net-omni-i12`, `modal='mpa'` — much faster than
DFT, well suited to large-scale carbon screening.

Spin polarization is essential for open-shell defects (vacancies, radicals).

## Tips

- Set `DEBUG_MODE = True` first to inspect atom indices before committing to a run.
- Leave `USE_SAVED_RELAXED = True` to skip re-relaxing the pristine reference.
- `.cube` densities open in VESTA/VMD; `.xyz`/`.traj` open with `ase gui file.xyz`.

## Citation

- **GPAW** — J. J. Mortensen et al., *Phys. Rev. B* **71**, 035109 (2005).
- **SevenNet** — Park et al., arXiv:2409.06676 (2024).
- **ASE** — A. H. Larsen et al., *J. Phys.: Condens. Matter* **29**, 273002 (2017).

For academic research use.
</content>
</invoke>
