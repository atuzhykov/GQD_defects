"""
GQD_functionalization_mols_energy_comparison.py
Legacy-style functionalization formation-energy comparison.

Sibling of GQD_defects_mols_energy_comparison.py / GQD_dopants_mols_energy_comparison.py,
but for edge functionalization (-OH, -NH2, -COOH, -O, ...): you pass a pristine
structure and one or more pre-made functionalized structures (all registered in
config.py molecules_data), the script relaxes them and reports the
functionalization formation energy.

------------------------------------------------------------------------------
FORMATION-ENERGY FORMULA
------------------------------------------------------------------------------
We do NOT try to guess "which functional group was attached". Instead, exactly
like the defect/dopant scripts, we read the change in composition straight from
the two structures and reference every changed element to its own elemental
chemical potential:

    E_form = E_modified - E_pristine - Σ_i  Δn_i · μ_i

    Δn_i = n_i(modified) - n_i(pristine)      (per element i)
    μ_i  = elemental chemical potential of i  (reservoir energy per atom)

Sign convention: an *added* atom (Δn_i > 0) is taken from a reservoir at cost
μ_i, so it lowers E_form by μ_i; a *removed* atom (Δn_i < 0) is returned to the
reservoir, raising E_form by μ_i. This is the same convention as
utils.calculate_formation_energy (defects) and DopingFormationCalculator
(dopants), just generalised to any set of elements.

------------------------------------------------------------------------------
DOES THE FORMULA ACCOUNT FOR THE EXTRA HYDROGEN? — YES.
------------------------------------------------------------------------------
This was the open question. Worked example, -NH2 (verified from the .mol files):

    GQD-H  (C24 H12)  ->  GQD-NH2  (C24 H13 N1)
    edge  C-H  becomes  C-NH2 : one C-H is lost, two N-H are gained.

    Δn_N = +1,  Δn_H = +13 - 12 = +1.

Because the formula sums over EVERY element whose count changes, the net +1 H
shows up as its own term -Δn_H·μ_H = -1·μ_H. Nothing is dropped. For -OH the
net is Δn_O=+1, Δn_H=0 (a C-H is replaced by an O-H, so H is unchanged) and only
the -μ_O term appears. The composition table printed for every comparison makes
this explicit so the hydrogen balance can be checked by eye.

------------------------------------------------------------------------------
Chemical potentials are computed dynamically with the SAME calculator used for
the structures (so the energy scale matches), mirroring how the defect script
calls utils.calculate_element_mu / calculate_mu_H:
    C : energy/atom of relaxed graphene          (utils.calculate_element_mu)
    H : E(H2)/2                                   (utils.calculate_mu_H)
    N : E(N2)/2,  O : E(O2)/2,  F : E(F2)/2       (gas-phase diatomic, /2)
Only the elements that actually change across the requested comparisons are
computed, so no reference molecule is relaxed needlessly.

Outputs to  legacy_func_<pristine>_vs_<modified>_<calc>/ :
    input_<...>                       copied input files
    relaxed_structures_xyz/           relaxed pristine + functionalized (XYZ)
    relaxed_structures_mol/           relaxed pristine + functionalized (MOL, bonds kept)
    functionalization_energy_analysis.txt   full thermodynamic breakdown
    ring_overlay_<modified>.png       ring-topology overlay of relaxed structure
"""

import os
import shutil

import numpy as np
from ase.build import molecule
from ase.io import read
from ase.optimize import BFGS

from config import molecules_data
from GQD_basic_defects import setup_calculator, FMAX, BFGS_MAXSTEP
from map import read_bonds_from_mol, save_structure_file
from utils import calculate_element_mu, calculate_mu_H
from ring_overlay import save_ring_overlay


# Gas-phase diatomic references X2 -> mu(X) = E(X2)/2.  C and H are handled
# separately (graphene / H2 via utils) to stay consistent with the other scripts.
_DIATOMIC_REFERENCE = {
    'H': 'H2', 'N': 'N2', 'O': 'O2', 'F': 'F2', 'Cl': 'Cl2',
}


def _mu_diatomic(calc, element):
    """mu(element) = E(X2)/2 from a relaxed gas-phase diatomic, same calculator."""
    formula = _DIATOMIC_REFERENCE[element]
    ref = molecule(formula)
    ref.set_cell([15.0, 15.0, 15.0])
    ref.center()
    ref.set_pbc(True)
    ref.calc = calc
    BFGS(ref, logfile=None).run(fmax=FMAX)
    return ref.get_potential_energy() / 2.0


def build_chemical_potentials(calc, elements):
    """Build {element: mu} for exactly the elements requested.

    C -> graphene energy/atom, H -> E(H2)/2, diatomic-forming elements -> E(X2)/2.
    Raises for any element without a defined reference so mistakes are loud.
    """
    mu = {}
    for el in sorted(elements):
        if el == 'C':
            mu[el] = calculate_element_mu(calc)          # graphene reference
        elif el == 'H':
            mu[el] = calculate_mu_H(calc)                # E(H2)/2 reference
        elif el in _DIATOMIC_REFERENCE:
            mu[el] = _mu_diatomic(calc, el)              # E(X2)/2 reference
        else:
            raise ValueError(
                f"No elemental chemical-potential reference defined for '{el}'. "
                f"Add one to build_chemical_potentials() / _DIATOMIC_REFERENCE."
            )
        print(f"  mu({el}) = {mu[el]:.4f} eV")
    return mu


def _load_centered(mol_key, calc):
    """Read a config.py structure, center it in its cell, attach the calculator."""
    mol_path = molecules_data[mol_key]["path"]
    cell = molecules_data[mol_key]["cell"]

    atoms = read(mol_path)
    positions = atoms.get_positions()
    com = np.mean(positions, axis=0)
    translation = np.array([cell / 2] * 3) - com
    atoms.set_positions(positions + translation)
    atoms.set_cell([cell] * 3)
    atoms.set_pbc(True)

    if mol_path.endswith('.mol'):
        bonds = read_bonds_from_mol(mol_path)
        if bonds:
            atoms.info['original_bonds'] = bonds

    atoms.calc = calc
    return atoms, mol_path


def _relax(atoms, label):
    print(f"Relaxing {label} ...")
    dyn = BFGS(atoms, maxstep=BFGS_MAXSTEP)
    dyn.run(fmax=FMAX)
    energy = atoms.get_potential_energy()
    max_force = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
    print(f"  {label}: E = {energy:.4f} eV, max_force = {max_force:.4f} eV/A")
    return energy


def _composition(atoms):
    comp = {}
    for sym in atoms.get_chemical_symbols():
        comp[sym] = comp.get(sym, 0) + 1
    return comp


def compare_functionalization(pristine_key, pristine, E_pristine, comp_pristine,
                              modified_key, calc, calc_name, mu):
    """Relax the functionalized structure (pristine is pre-relaxed) and report E_form.

    pristine, E_pristine, comp_pristine are passed in already-computed so a series
    of comparisons against the same pristine reference re-uses one relaxation.
    """
    print(f"\n{'=' * 60}")
    print(f"FUNCTIONALIZATION FORMATION ENERGY: {pristine_key} -> {modified_key}")
    print(f"{'=' * 60}")

    modified, modified_path = _load_centered(modified_key, calc)
    pristine_path = molecules_data[pristine_key]["path"]

    E_modified = _relax(modified, f"functionalized ({modified_key})")

    comp_modified = _composition(modified)

    # ---- composition change, per element ----
    elements = sorted(set(comp_pristine) | set(comp_modified))
    deltas = {el: comp_modified.get(el, 0) - comp_pristine.get(el, 0) for el in elements}
    changed = {el: d for el, d in deltas.items() if d != 0}

    # ---- formation energy ----
    # E_form = E_modified - E_pristine - Σ_i Δn_i·μ_i  (added atom -> -μ, removed -> +μ)
    # Computed by hand from the cached float E_pristine: re-evaluating pristine here
    # would be silently recomputed by the shared calculator (its internal cache was
    # just overwritten by the functionalized relaxation) and erase the speedup.
    mu_terms = {el: deltas[el] * mu[el] for el in changed}
    mu_total = sum(mu_terms.values())
    E_form = E_modified - E_pristine - mu_total

    added = {el: d for el, d in changed.items() if d > 0}
    removed = {el: -d for el, d in changed.items() if d < 0}

    # ---- results directory ----
    results_dir = f"legacy_func_{pristine_key}_vs_{modified_key}_{calc_name}"
    os.makedirs(results_dir, exist_ok=True)
    xyz_dir = os.path.join(results_dir, "relaxed_structures_xyz")
    mol_dir = os.path.join(results_dir, "relaxed_structures_mol")
    os.makedirs(xyz_dir, exist_ok=True)
    os.makedirs(mol_dir, exist_ok=True)

    for key, path in [(pristine_key, pristine_path), (modified_key, modified_path)]:
        if os.path.exists(path):
            shutil.copy(path, os.path.join(results_dir, f"input_{os.path.basename(path)}"))

    # Save calculator-free copies: the shared calculator caches results from the
    # last-evaluated structure, which would otherwise corrupt the writers.
    save_structure_file(
        pristine.copy(),
        os.path.join(xyz_dir, f"{pristine_key}_relaxed.xyz"),
        os.path.join(mol_dir, f"{pristine_key}_relaxed.mol"),
        original_bonds=pristine.info.get('original_bonds'),
    )
    save_structure_file(
        modified.copy(),
        os.path.join(xyz_dir, f"{modified_key}_relaxed.xyz"),
        os.path.join(mol_dir, f"{modified_key}_relaxed.mol"),
        original_bonds=modified.info.get('original_bonds'),
    )

    # ---- ring overlay of the relaxed functionalized structure ----
    overlay_path = os.path.join(results_dir, f"ring_overlay_{modified_key}.png")
    save_ring_overlay(modified, overlay_path, ef=E_form,
                      title=f"functionalization: {pristine_key} -> {modified_key}")

    # ---- human-readable description of what changed ----
    added_str = ", ".join(f"{el}x{n}" for el, n in added.items()) or "none"
    removed_str = ", ".join(f"{el}x{n}" for el, n in removed.items()) or "none"

    # ---- analysis text ----
    lines = [
        "=" * 70,
        "FUNCTIONALIZATION FORMATION ENERGY ANALYSIS",
        "=" * 70,
        "",
        f"Pristine system:      {pristine_key}",
        f"Functionalized system: {modified_key}",
        f"Calculator:           {calc_name}",
        "",
        "Composition (pristine -> functionalized, per element):",
    ]
    for el in elements:
        lines.append(
            f"  {el:2s}: {comp_pristine.get(el, 0):3d} -> {comp_modified.get(el, 0):3d}  "
            f"(delta_n = {deltas[el]:+d})"
        )
    lines += [
        "",
        f"  atoms added:   {added_str}",
        f"  atoms removed: {removed_str}",
        "",
        "Energy components:",
        f"  E(pristine):       {E_pristine:.6f} eV",
        f"  E(functionalized): {E_modified:.6f} eV",
        f"  delta_E:           {E_modified - E_pristine:.6f} eV",
        "",
        "Chemical-potential terms  (-delta_n_i * mu_i, summed below):",
    ]
    for el in sorted(changed):
        lines.append(
            f"  {el:2s}: delta_n = {deltas[el]:+d}  *  mu = {mu[el]:.6f} eV"
            f"   ->  {-mu_terms[el]:+.6f} eV"
        )
    lines += [
        f"  ----  total -Sigma delta_n*mu = {-mu_total:+.6f} eV",
        "",
        "=" * 70,
        "  E_form = E_functionalized - E_pristine - Sum_i delta_n_i * mu_i",
        f"  E_form = {E_form:.6f} eV",
        "=" * 70,
    ]
    text = "\n".join(lines)
    print("\n" + text)

    with open(os.path.join(results_dir, "functionalization_energy_analysis.txt"),
              "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"\nResults saved to: {results_dir}/")
    print(f"  - functionalization_energy_analysis.txt")
    print(f"  - relaxed_structures_xyz/ , relaxed_structures_mol/")
    print(f"  - ring_overlay_{modified_key}.png")
    return E_form


def main():
    calc, calc_name = setup_calculator()

    # ============================================================
    # CONFIGURATION
    # Both keys must exist in config.py molecules_data.
    # pristine_key -> the clean GQD; modified_keys -> pre-made functionalized GQDs.
    # Several functionalized structures can be compared in one run; the pristine
    # reference is relaxed only ONCE and shared across all of them.
    # ============================================================
    pristine_key = "GQD_HEX_2_2"
    modified_keys = ["GQD_HEX_2_2_func_all"]

    # ---- which elements CHANGE across all comparisons (build mu only for these) ----
    # Only elements whose count differs contribute a Δn·μ term, so e.g. an
    # unchanged carbon backbone never triggers a graphene relaxation.
    comp_pristine = _composition(read(molecules_data[pristine_key]["path"]))
    needed_elements = set()
    for mkey in modified_keys:
        comp_m = _composition(read(molecules_data[mkey]["path"]))
        for el in set(comp_pristine) | set(comp_m):
            if comp_m.get(el, 0) != comp_pristine.get(el, 0):
                needed_elements.add(el)

    print(f"\n{'=' * 60}")
    print("CHEMICAL POTENTIALS (computed with the active calculator)")
    print(f"{'=' * 60}")
    mu = build_chemical_potentials(calc, needed_elements)

    # Relax pristine ONCE: every comparison shares the same reference, so
    # re-relaxing it per modified_key is pure waste (1x pristine + N x modified,
    # vs. the previous N x of each).
    print(f"\n{'=' * 60}")
    print(f"PRISTINE REFERENCE: {pristine_key}")
    print(f"{'=' * 60}")
    pristine, _ = _load_centered(pristine_key, calc)
    E_pristine = _relax(pristine, f"pristine ({pristine_key})")
    comp_pristine = _composition(pristine)

    results = {}
    for modified_key in modified_keys:
        results[modified_key] = compare_functionalization(
            pristine_key, pristine, E_pristine, comp_pristine,
            modified_key, calc, calc_name, mu)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for modified_key, e_form in results.items():
        print(f"  {modified_key:28s}  E_form = {e_form:.4f} eV")


if __name__ == "__main__":
    main()
