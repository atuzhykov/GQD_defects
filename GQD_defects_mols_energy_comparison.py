"""
GQD_defects_mols_energy_comparison.py
Legacy-style defect formation-energy comparison.

Mirrors the "legacy" mode of GQD_dopants_mols_energy_comparison.py, but for
structural defects instead of doping: you pass a pristine structure and a
pre-made defect structure (both registered in config.py molecules_data), the
script relaxes both and reports the defect formation energy.

Vacancy (V), divacancy (VV) and Stone-Wales (SW) are all handled — the defect
type is auto-detected from the change in carbon count, exactly as in
utils.calculate_formation_energy:
    n_C_removed == 0  -> Stone-Wales   E_f = E_defect - E_pristine
    n_C_removed >= 1  -> V / VV / ...  E_f = E_defect - E_pristine
                                             + n_C_removed*mu_C + n_H_removed*mu_H

Outputs to  legacy_defect_<pristine>_vs_<defect>_<calc>/ :
  input_<...>                       copied input files
  relaxed_structures_xyz/           relaxed pristine + defect (XYZ)
  relaxed_structures_mol/           relaxed pristine + defect (MOL, bonds kept)
  defect_energy_analysis.txt        full thermodynamic breakdown
  ring_overlay_<defect>.png         non-hexagonal ring overlay of relaxed defect
"""

import os
import shutil

import numpy as np
from ase.io import read
from ase.optimize import BFGS

from config import molecules_data
from GQD_basic_defects import setup_calculator, FMAX, BFGS_MAXSTEP
from map import read_bonds_from_mol, save_structure_file
from utils import calculate_element_mu, calculate_mu_H
from ring_overlay import save_ring_overlay


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


def _count(atoms, symbol):
    return sum(1 for a in atoms if a.symbol == symbol)


def compare_defect(pristine_key, pristine, E_pristine,
                   defect_key, calc, calc_name, mu_C, mu_H):
    """Relax the defect structure (pristine is pre-relaxed) and report formation energy.

    pristine, E_pristine are passed in already-relaxed so a series of defect
    comparisons against the same pristine reference re-uses one relaxation.
    """
    print(f"\n{'=' * 60}")
    print(f"DEFECT FORMATION ENERGY: {pristine_key} -> {defect_key}")
    print(f"{'=' * 60}")

    defect, defect_path = _load_centered(defect_key, calc)
    pristine_path = molecules_data[pristine_key]["path"]

    E_defect = _relax(defect, f"defect ({defect_key})")

    n_C_p, n_C_d = _count(pristine, 'C'), _count(defect, 'C')
    n_H_p, n_H_d = _count(pristine, 'H'), _count(defect, 'H')
    n_C_removed = n_C_p - n_C_d
    n_H_removed = n_H_p - n_H_d

    if n_C_removed == 0:
        defect_type = "Stone-Wales"
    elif n_C_removed == 1:
        defect_type = "vacancy"
    elif n_C_removed == 2:
        defect_type = "divacancy"
    else:
        defect_type = f"multi-vacancy ({n_C_removed} C removed)"

    # mu_H only enters the formula when H atoms were also removed
    _mu_H = mu_H if n_H_removed != 0 else 0.0
    # Formation energy is computed by hand from the cached E_pristine.
    # Calling utils.calculate_formation_energy(pristine, defect, calc, ...) would
    # trigger pristine.get_potential_energy() — and since the shared calculator's
    # internal cache was just overwritten by the defect relaxation, that would
    # silently recompute pristine and erase the speedup.
    if n_C_removed == 0:
        E_form = E_defect - E_pristine
    else:
        E_form = E_defect - E_pristine + n_C_removed * mu_C + n_H_removed * _mu_H

    mu_C_term = n_C_removed * mu_C
    mu_H_term = n_H_removed * _mu_H

    # ---- results directory ----
    results_dir = f"legacy_defect_{pristine_key}_vs_{defect_key}_{calc_name}"
    os.makedirs(results_dir, exist_ok=True)
    xyz_dir = os.path.join(results_dir, "relaxed_structures_xyz")
    mol_dir = os.path.join(results_dir, "relaxed_structures_mol")
    os.makedirs(xyz_dir, exist_ok=True)
    os.makedirs(mol_dir, exist_ok=True)

    for key, path in [(pristine_key, pristine_path), (defect_key, defect_path)]:
        if os.path.exists(path):
            shutil.copy(path, os.path.join(results_dir, f"input_{os.path.basename(path)}"))

    # Save calculator-free copies: the shared calculator caches results from the
    # last-evaluated structure, which would otherwise corrupt the XYZ writer.
    save_structure_file(
        pristine.copy(),
        os.path.join(xyz_dir, f"{pristine_key}_relaxed.xyz"),
        os.path.join(mol_dir, f"{pristine_key}_relaxed.mol"),
        original_bonds=pristine.info.get('original_bonds'),
    )
    save_structure_file(
        defect.copy(),
        os.path.join(xyz_dir, f"{defect_key}_relaxed.xyz"),
        os.path.join(mol_dir, f"{defect_key}_relaxed.mol"),
        original_bonds=defect.info.get('original_bonds'),
    )

    # ---- ring overlay of the relaxed defect ----
    overlay_path = os.path.join(results_dir, f"ring_overlay_{defect_key}.png")
    save_ring_overlay(defect, overlay_path, ef=E_form,
                      title=f"{defect_type}: {pristine_key} -> {defect_key}")

    # ---- analysis text ----
    lines = [
        "=" * 70,
        "DEFECT FORMATION ENERGY ANALYSIS",
        "=" * 70,
        "",
        f"Pristine system: {pristine_key}",
        f"Defect system:   {defect_key}",
        f"Defect type:     {defect_type}",
        f"Calculator:      {calc_name}",
        "",
        "Composition:",
        f"  C atoms: {n_C_p} -> {n_C_d}  (removed: {n_C_removed})",
        f"  H atoms: {n_H_p} -> {n_H_d}  (removed: {n_H_removed})",
        "",
        "Energy components:",
        f"  E(pristine):  {E_pristine:.6f} eV",
        f"  E(defect):    {E_defect:.6f} eV",
        f"  delta_E:      {E_defect - E_pristine:.6f} eV",
        "",
        "Chemical potentials:",
        f"  mu_C = {mu_C:.6f} eV   ->  n_C_removed * mu_C = {mu_C_term:+.6f} eV",
        f"  mu_H = {_mu_H:.6f} eV   ->  n_H_removed * mu_H = {mu_H_term:+.6f} eV",
        "",
        "=" * 70,
        "  E_form = E_defect - E_pristine + n_C_removed*mu_C + n_H_removed*mu_H",
        f"  E_form = {E_form:.6f} eV",
        "=" * 70,
    ]
    text = "\n".join(lines)
    print("\n" + text)

    with open(os.path.join(results_dir, "defect_energy_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"\nResults saved to: {results_dir}/")
    print(f"  - defect_energy_analysis.txt")
    print(f"  - relaxed_structures_xyz/ , relaxed_structures_mol/")
    print(f"  - ring_overlay_{defect_key}.png")
    return E_form


def main():
    calc, calc_name = setup_calculator()

    mu_C = calculate_element_mu(calc)
    mu_H = calculate_mu_H(calc)
    print(f"mu_C = {mu_C:.4f} eV   mu_H = {mu_H:.4f} eV")

    # ============================================================
    # CONFIGURATION
    # Both keys must exist in config.py molecules_data.
    # pristine_key -> the perfect GQD; defect_key -> a pre-made V / VV / SW.
    # ============================================================
    pristine_key = "input_pyrene_C16H10"
    defect_keys = ["pyrene_V_edge", "pyrene_VV_edge"]

    # Relax pristine ONCE: every defect in the series shares the same reference,
    # so re-relaxing it per defect_key is pure waste (1× pristine relax + N×
    # defect relax, vs. the previous N× of each).
    print(f"\n{'=' * 60}")
    print(f"PRISTINE REFERENCE: {pristine_key}")
    print(f"{'=' * 60}")
    pristine, _ = _load_centered(pristine_key, calc)
    E_pristine = _relax(pristine, f"pristine ({pristine_key})")

    results = {}
    for defect_key in defect_keys:
        results[defect_key] = compare_defect(
            pristine_key, pristine, E_pristine,
            defect_key, calc, calc_name, mu_C, mu_H)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for defect_key, e_form in results.items():
        print(f"  {defect_key:24s}  E_form = {e_form:.4f} eV")


if __name__ == "__main__":
    main()
