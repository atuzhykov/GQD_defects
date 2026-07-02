"""
GQD_dopants_mols_energy_comparison.py
Substitutional-doping formation-energy comparison.

Sibling of GQD_defects_mols_energy_comparison.py /
GQD_functionalization_mols_energy_comparison.py, but for substitutional doping
(N, B, P, S, O, F, ...): you pass a pristine structure and one or more pre-made
doped structures (all registered in config.py molecules_data), the script relaxes
them and reports the doping formation energy.

Two modes are available in main():
  * "legacy"    — compare a pristine GQD against one OR MORE pre-made doped GQDs.
                  The pristine reference is relaxed only ONCE and shared across all
                  comparisons (mirrors the functionalization script).
  * "automated" — scan every substitution site of a single GQD for one dopant and
                  build an energy map (DopingFormationCalculator.analyze_doping_sites).
                  Uses the SAME dynamically-computed chemical potentials as the
                  legacy mode, so both modes agree on E_form.

------------------------------------------------------------------------------
FORMATION-ENERGY FORMULA  (legacy mode)
------------------------------------------------------------------------------
We do NOT special-case "which dopant replaced which host". Exactly like the
defect/functionalization scripts, we read the change in composition straight from
the two structures and reference every changed element to its own elemental
chemical potential:

    E_form = E_doped - E_pristine - Σ_i  Δn_i · μ_i

    Δn_i = n_i(doped) - n_i(pristine)        (per element i)
    μ_i  = elemental chemical potential of i (reservoir energy per atom)

Sign convention: an *added* atom (Δn_i > 0) is taken from a reservoir at cost
μ_i, so it lowers E_form by μ_i; a *removed* atom (Δn_i < 0) is returned to the
reservoir, raising E_form by μ_i. For a substitutional N→C swap this reduces to
the familiar  E_doped - E_pristine + μ_C - μ_N  (Δn_C = -1, Δn_N = +1).

------------------------------------------------------------------------------
CHEMICAL POTENTIALS — computed dynamically with the SAME calculator
------------------------------------------------------------------------------
Mirrors the functionalization script so the energy scale matches the active
calculator (SevenNet / GPAW):
    C : energy/atom of relaxed graphene          (utils.calculate_element_mu)
    H : E(H2)/2                                   (utils.calculate_mu_H)
    N : E(N2)/2,  O : E(O2)/2,  F : E(F2)/2, ...  (gas-phase diatomic, /2)

Elements with no gas-phase diatomic / graphene reference (B, P, S, Li) fall back
to hard-coded literature values (bulk α-boron, white P4, orthorhombic S, bcc Li).
Those values are on a DIFFERENT energy scale than the active calculator, so the
fallback is flagged loudly — provide a proper bulk reference for publication-grade
numbers. Only the elements that actually change across the requested comparisons
are computed, so no reference molecule is relaxed needlessly.

Outputs (legacy mode) to  legacy_doping_<pristine>_vs_<doped>_<calc>/ :
    input_<...>                       copied input files
    relaxed_structures_xyz/           relaxed pristine + doped (XYZ)
    relaxed_structures_mol/           relaxed pristine + doped (MOL, bonds kept)
    doping_energy_analysis.txt        full thermodynamic breakdown
"""

import os
import shutil
import logging
from typing import List, Tuple

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

import settings  # applies the project matplotlib style on import; ATOM_IDX_FONTSIZE
import cache
from config import molecules_data
from GQD_basic_defects import setup_calculator, FMAX
from utils import determine_target_element

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# MOL I/O helpers (used by the automated site-map mode)
# ============================================================================
def read_bonds_from_mol(filepath):
    """
    Read bond information from MOL file.
    Returns list of (atom1, atom2, bond_type) tuples (0-indexed).
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        if len(lines) < 4:
            return None

        # Parse counts line (line 4 in V2000 format)
        counts_line = lines[3]
        n_atoms = int(counts_line[0:3])
        n_bonds = int(counts_line[3:6])

        # Bond block starts after header (4 lines) + atom block
        bond_start = 4 + n_atoms

        if bond_start + n_bonds > len(lines):
            return None

        bonds = []
        for i in range(n_bonds):
            bond_line = lines[bond_start + i]
            atom1 = int(bond_line[0:3]) - 1  # Convert to 0-indexed
            atom2 = int(bond_line[3:6]) - 1
            bond_type = int(bond_line[6:9])  # 1=single, 2=double, 3=triple
            bonds.append((atom1, atom2, bond_type))

        return bonds

    except Exception as e:
        logger.warning(f"Could not read bonds from {filepath}: {e}")
        return None


def write_mol_with_bonds(atoms, filepath, original_bonds=None):
    """
    Write MOL file (V2000 format) with preserved bond information.

    Args:
        atoms: ASE Atoms object
        filepath: Output file path
        original_bonds: List of (atom1, atom2, bond_type) tuples (0-indexed)
    """
    with open(filepath, 'w') as f:
        # Header block
        f.write(f"{os.path.basename(filepath)}\n")
        f.write("  Generated by orientational_defects\n")
        f.write("\n")  # Comment line

        # Counts line: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
        # aaa = number of atoms, bbb = number of bonds
        bonds = []
        if original_bonds is not None:
            # Use original bonds, but only include bonds where both atoms still exist
            for atom1, atom2, bond_type in original_bonds:
                if atom1 < len(atoms) and atom2 < len(atoms):
                    bonds.append((atom1 + 1, atom2 + 1, bond_type))  # Convert to 1-indexed

        f.write(f"{len(atoms):3d}{len(bonds):3d}  0  0  0  0  0  0  0  0999 V2000\n")

        # Atom block
        for atom in atoms:
            x, y, z = atom.position
            symbol = atom.symbol
            f.write(f"{x:10.4f}{y:10.4f}{z:10.4f} {symbol:3s} 0  0  0  0  0  0  0  0  0  0  0  0\n")

        # Bond block
        for atom1, atom2, bond_type in bonds:
            f.write(f"{atom1:3d}{atom2:3d}{bond_type:3d}  0  0  0  0\n")

        # End tag
        f.write("M  END\n")


def save_structure_image(atoms, filepath, title="Structure"):
    """Save an image of the atomic structure using ASE's plot_atoms"""
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_atoms(atoms, ax, rotation=('0x,0y,0z'))
        ax.set_title(title)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not generate image: {e}")
        plt.close('all')


def save_structure_file(atoms, xyz_path, mol_path, original_bonds=None):
    """
    Save atomic structure in both XYZ and MOL formats with bond preservation.

    Args:
        atoms: ASE Atoms object
        xyz_path: Path for XYZ file (coordinates only)
        mol_path: Path for MOL file (coordinates + bonds)
        original_bonds: List of (atom1, atom2, bond_type) tuples (0-indexed) to preserve in MOL
    """
    # Save XYZ format (coordinates only)
    write(xyz_path, atoms)

    # Save MOL format with preserved bonds
    write_mol_with_bonds(atoms, mol_path, original_bonds=original_bonds)


# ============================================================================
# Chemical potentials (legacy mode) — dynamic + cached, via cache.py
# ============================================================================
# Hard-coded literature fallback (eV/atom) for elements with no gas-phase diatomic
# or graphene reference: bulk α-boron, white P4, orthorhombic S, bcc Li. These are
# on a DIFFERENT energy scale than the active calculator -> flagged when used.
_FALLBACK_MU = {
    'B': -6.678, 'P': -5.641, 'S': -4.136, 'Li': -1.82,
}


def build_chemical_potentials(calc, calc_name, elements, fmax=FMAX):
    """Build {element: mu} for exactly the elements requested, with caching.

    C/H and the diatomic-forming elements (N/O/F/Cl) are fetched through
    cache.get_chemical_potential (graphene / E(H2)/2 / E(X2)/2), so those
    reference relaxations run at most once per calculator across all runs.
    B/P/S/Li fall back to hard-coded literature values (flagged, since their
    energy scale need not match the active calculator). Raises for any element
    with neither reference so mistakes are loud.

    Used by BOTH the legacy pairwise mode and the automated site-map mode, so
    the two report identical E_form for the same substitution.
    """
    mu = {}
    for el in sorted(elements):
        try:
            mu[el] = cache.get_chemical_potential(el, calc, calc_name, fmax=fmax)
        except ValueError:
            if el not in _FALLBACK_MU:
                raise
            mu[el] = _FALLBACK_MU[el]                     # literature bulk reference
            print(f"  WARNING: mu({el}) = {mu[el]:.4f} eV is a hard-coded literature "
                  f"value; its energy scale may NOT match the active calculator. "
                  f"Provide a bulk reference for publication-grade numbers.")
        print(f"  mu({el}) = {mu[el]:.4f} eV")
    return mu


# ============================================================================
# Structure / relaxation helpers (legacy mode)
# ============================================================================
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


def _composition(atoms):
    comp = {}
    for sym in atoms.get_chemical_symbols():
        comp[sym] = comp.get(sym, 0) + 1
    return comp


def compare_doping(pristine_key, pristine, E_pristine, comp_pristine,
                   doped_key, calc, calc_name, mu):
    """Relax the doped structure (pristine is pre-relaxed) and report E_form.

    pristine, E_pristine, comp_pristine are passed in already-computed so a series
    of comparisons against the same pristine reference re-uses one relaxation.
    """
    print(f"\n{'=' * 60}")
    print(f"DOPING FORMATION ENERGY: {pristine_key} -> {doped_key}")
    print(f"{'=' * 60}")

    doped, doped_path = _load_centered(doped_key, calc)
    pristine_path = molecules_data[pristine_key]["path"]

    doped, E_doped = cache.load_or_relax(
        doped, doped_key, calc, calc_name, fmax=FMAX,
        cell=molecules_data[doped_key]["cell"],
        label=f"doped ({doped_key})")

    comp_doped = _composition(doped)

    # ---- composition change, per element ----
    elements = sorted(set(comp_pristine) | set(comp_doped))
    deltas = {el: comp_doped.get(el, 0) - comp_pristine.get(el, 0) for el in elements}
    changed = {el: d for el, d in deltas.items() if d != 0}

    # ---- formation energy ----
    # E_form = E_doped - E_pristine - Σ_i Δn_i·μ_i  (added atom -> -μ, removed -> +μ)
    # Computed by hand from the cached float E_pristine: re-evaluating pristine here
    # would be silently recomputed by the shared calculator (its internal cache was
    # just overwritten by the doped relaxation) and erase the speedup.
    mu_terms = {el: deltas[el] * mu[el] for el in changed}
    mu_total = sum(mu_terms.values())
    E_form = E_doped - E_pristine - mu_total

    added = {el: d for el, d in changed.items() if d > 0}
    removed = {el: -d for el, d in changed.items() if d < 0}

    # ---- doping classification (substitutional / interstitial / vacancy) ----
    n_added = sum(d for el, d in changed.items() if d > 0 and el != 'C')
    n_C_removed = -deltas.get('C', 0) if deltas.get('C', 0) < 0 else 0
    if n_added > 0 and n_C_removed > 0:
        doping_type = "substitutional"
    elif n_added > 0 and n_C_removed == 0:
        doping_type = "interstitial / addition"
    elif n_added == 0 and n_C_removed > 0:
        doping_type = "vacancy"
    else:
        doping_type = "rearrangement"

    # ---- results directory ----
    results_dir = f"legacy_doping_{pristine_key}_vs_{doped_key}_{calc_name}"
    os.makedirs(results_dir, exist_ok=True)
    xyz_dir = os.path.join(results_dir, "relaxed_structures_xyz")
    mol_dir = os.path.join(results_dir, "relaxed_structures_mol")
    os.makedirs(xyz_dir, exist_ok=True)
    os.makedirs(mol_dir, exist_ok=True)

    for key, path in [(pristine_key, pristine_path), (doped_key, doped_path)]:
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
        doped.copy(),
        os.path.join(xyz_dir, f"{doped_key}_relaxed.xyz"),
        os.path.join(mol_dir, f"{doped_key}_relaxed.mol"),
        original_bonds=doped.info.get('original_bonds'),
    )

    # ---- human-readable description of what changed ----
    added_str = ", ".join(f"{el}x{n}" for el, n in added.items()) or "none"
    removed_str = ", ".join(f"{el}x{n}" for el, n in removed.items()) or "none"

    # ---- analysis text ----
    lines = [
        "=" * 70,
        "DOPING FORMATION ENERGY ANALYSIS",
        "=" * 70,
        "",
        f"Pristine system: {pristine_key}",
        f"Doped system:    {doped_key}",
        f"Calculator:      {calc_name}",
        f"Doping type:     {doping_type}",
        "",
        "Composition (pristine -> doped, per element):",
    ]
    for el in elements:
        lines.append(
            f"  {el:2s}: {comp_pristine.get(el, 0):3d} -> {comp_doped.get(el, 0):3d}  "
            f"(delta_n = {deltas[el]:+d})"
        )
    lines += [
        "",
        f"  atoms added:   {added_str}",
        f"  atoms removed: {removed_str}",
        "",
        "Energy components:",
        f"  E(pristine): {E_pristine:.6f} eV",
        f"  E(doped):    {E_doped:.6f} eV",
        f"  delta_E:     {E_doped - E_pristine:.6f} eV",
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
        "  E_form = E_doped - E_pristine - Sum_i delta_n_i * mu_i",
        f"  E_form = {E_form:.6f} eV",
        "=" * 70,
    ]
    text = "\n".join(lines)
    print("\n" + text)

    with open(os.path.join(results_dir, "doping_energy_analysis.txt"),
              "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"\nResults saved to: {results_dir}/")
    print(f"  - doping_energy_analysis.txt")
    print(f"  - relaxed_structures_xyz/ , relaxed_structures_mol/")
    return E_form


class DopingFormationCalculator:
    """
    Doping formation energy calculator — automated site-map mode.

    Scans every substitution site of a GQD for a chosen dopant element and builds
    an energy map + statistical distribution (analyze_doping_sites). The legacy
    pairwise comparison now lives in the module-level compare_doping() function.

    Chemical potentials come from the same build_chemical_potentials() as the
    legacy mode — computed dynamically with the ACTIVE calculator and disk-cached
    (graphene mu_C, E(H2)/2 mu_H, E(X2)/2 for N/O/F/Cl, flagged literature
    fallback for B/P/S/Li). Previously this mode used its own hard-coded mu
    table, whose absolute energy scale did not match the active calculator's
    total energies, so the site map and the legacy pairwise comparison reported
    DIFFERENT formation energies for the identical substitution.
    """

    def __init__(self, calculator, fmax=FMAX, calc_name="MLIP"):
        """
        Args:
            calculator: ASE calculator (e.g., SevenNet or GPAW)
            fmax: Force convergence criterion [eV/Å]
            calc_name: Name of the calculator for result folder naming
        """
        self.calc = calculator
        self.fmax = fmax
        self.calc_name = calc_name
        self.energy_cache = {}  # Cache for calculated energies

    def analyze_doping_sites(self, mol_path: str, dopant_element: str, molecule_name: str,
                           show_atom_idx: bool = True, excluded_atoms: List[int] = None,
                           replace_H: bool = False) -> Tuple[List[int], List[float]]:
        """
        Analyze all possible substitutional doping sites in a GQD structure.

        Similar to map.py's vacancy analysis, but for dopant substitution.

        Args:
            mol_path: Path to the GQD molecule file
            dopant_element: Element symbol for dopant (e.g., 'Li', 'N', 'B')
            molecule_name: Name for output files
            show_atom_idx: Whether to show atom indices on plots
            excluded_atoms: List of atom indices to exclude from analysis
            replace_H: If True, also consider replacing H atoms (expands analysis map)

        Returns:
            Tuple of (atom_indices, formation_energies)
        """
        if excluded_atoms is None:
            excluded_atoms = []

        suffix = "_with_H" if replace_H else ""
        print(f"\n=== Running Doping Site Analysis for {dopant_element} {'(including H)' if replace_H else ''} ===\n")
        results_dir = f"doping_map_{molecule_name}_{dopant_element}{suffix}_{self.calc_name}"
        os.makedirs(results_dir, exist_ok=True)

        # Create subdirectories for structures (separate folders for XYZ and MOL) and images
        structures_xyz_dir = os.path.join(results_dir, "relaxed_structures_xyz")
        structures_mol_dir = os.path.join(results_dir, "relaxed_structures_mol")
        images_dir = os.path.join(results_dir, "structure_images")
        os.makedirs(structures_xyz_dir, exist_ok=True)
        os.makedirs(structures_mol_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        # Copy input file to results directory
        shutil.copy(mol_path, os.path.join(results_dir, f"input_{os.path.basename(mol_path)}"))

        # Load base structure
        base_atoms = read(mol_path)

        # Read and store original bond information if input is MOL file
        original_bonds = None
        if mol_path.endswith('.mol'):
            original_bonds = read_bonds_from_mol(mol_path)
            if original_bonds:
                print(f"Loaded {len(original_bonds)} bonds from input MOL file")
                base_atoms.info['original_bonds'] = original_bonds
            else:
                print("Warning: Could not read bonds from MOL file")
        else:
            print("Input is not a MOL file - bonds will not be preserved")

        # Set up cell size and centering (like map.py)
        cell = molecules_data[molecule_name]["cell"]
        positions = base_atoms.get_positions()
        center_of_mass = np.mean(positions, axis=0)
        translation = np.array([cell / 2] * 3) - center_of_mass
        positions = [list(np.array(pos) + translation) for pos in positions]
        base_atoms.set_positions(positions)
        base_atoms.set_cell([cell] * 3)
        base_atoms.set_pbc(True)

        target_element, _ = determine_target_element(base_atoms)

        # Relax the pristine base structure (cached on disk: the base is a
        # config-key structure shared with the legacy mode and re-used across
        # runs). The per-site doped relaxations below are NOT cached.
        base_atoms.calc = self.calc
        base_atoms, E_pristine = cache.load_or_relax(
            base_atoms, molecule_name, self.calc, self.calc_name,
            fmax=self.fmax, cell=cell, label=f"base ({molecule_name})")

        # Initialize results
        formation_energies = []
        atom_indices = []

        # Determine which elements to replace
        if replace_H:
            target_elements = [target_element, 'H']
            print(f"Analyzing both {target_element} and H substitution sites")
        else:
            target_elements = [target_element]
            print(f"Analyzing only {target_element} substitution sites")

        # Chemical potentials: same dynamic, calculator-consistent references as
        # the legacy mode (cached on disk — computed at most once per calculator).
        print("Chemical potentials (computed with the active calculator):")
        mu = build_chemical_potentials(
            self.calc, self.calc_name,
            set(target_elements) | {dopant_element}, fmax=self.fmax)

        # Loop through each target element atom
        for atom_idx in range(len(base_atoms)):
            if base_atoms[atom_idx].symbol not in target_elements or atom_idx in excluded_atoms:
                continue

            print(f"Processing atom {atom_idx} of {len(base_atoms)} (symbol: {base_atoms[atom_idx].symbol})")
            atom_indices.append(atom_idx)

            try:
                # Create doped structure by substituting this atom
                doped_atoms = base_atoms.copy()
                doped_atoms[atom_idx].symbol = dopant_element

                # Relax doped structure
                doped_atoms.calc = self.calc
                print(f"  Relaxing doped structure...")
                dyn = BFGS(doped_atoms, logfile=None)
                dyn.run(fmax=self.fmax)

                E_doped = doped_atoms.get_potential_energy()

                # Calculate formation energy using chemical potentials
                # Get the host element being replaced (could be C or H)
                host_element = base_atoms[atom_idx].symbol
                mu_host = mu[host_element]
                mu_dopant = mu[dopant_element]

                # For substitutional doping: ΔE_f = E_doped - E_pristine + μ_host - μ_dopant
                formation_energy = E_doped - E_pristine + mu_host - mu_dopant

                formation_energies.append(formation_energy)
                print(f"  Atom {atom_idx} ({host_element}) formation energy: {formation_energy:.3f} eV")

                # Save relaxed structure in both XYZ and MOL formats, and image
                xyz_file = os.path.join(structures_xyz_dir, f"doped_atom_{atom_idx}_{dopant_element}.xyz")
                mol_file = os.path.join(structures_mol_dir, f"doped_atom_{atom_idx}_{dopant_element}.mol")
                image_file = os.path.join(images_dir, f"doped_atom_{atom_idx}_{dopant_element}.png")

                # For substitutional doping, no atoms are removed, so use original bonds directly
                # The atom at atom_idx just changes element, but all atom indices remain the same
                save_structure_file(doped_atoms, xyz_file, mol_file, original_bonds=original_bonds)
                save_structure_image(doped_atoms, image_file,
                                   title=f"{dopant_element} doping at atom {atom_idx} ({host_element})\nE_form = {formation_energy:.3f} eV")
                print(f"  Saved XYZ structure to {xyz_file}")
                print(f"  Saved MOL structure to {mol_file}")
                print(f"  Saved image to {image_file}")

            except Exception as e:
                print(f"  Error processing atom {atom_idx}: {e}")
                formation_energies.append(None)

        # Create visualizations and analysis
        self._process_doping_results(results_dir, base_atoms, atom_indices, formation_energies,
                                     molecule_name, dopant_element, show_atom_idx, replace_H)

        return atom_indices, formation_energies

    def _process_doping_results(self, results_dir: str, base_atoms, atom_indices: List[int],
                               formation_energies: List[float], molecule_name: str,
                               dopant_element: str, show_atom_idx: bool = True,
                               replace_H: bool = False) -> None:
        """Process and visualize doping site results - similar to map.py's vacancy processing."""

        # Create energy map dictionary
        energy_map = {idx: energy for idx, energy in zip(atom_indices, formation_energies)}

        # Save formation energies (failed sites as NaN — a None would force a
        # pickled object array that np.load refuses by default)
        np.savez(os.path.join(results_dir, "formation_energies.npz"),
                indices=np.array(atom_indices),
                energies=np.array([np.nan if e is None else e for e in formation_energies], dtype=float))

        # Write energy map to text file
        with open(os.path.join(results_dir, "energy_map.txt"), 'w') as f:
            f.write("Atom_Index Formation_Energy(eV)\n")
            for idx, energy in zip(atom_indices, formation_energies):
                if energy is not None:
                    f.write(f"{idx} {energy:.3f}\n")
                else:
                    f.write(f"{idx} None\n")

        # Filter out None values
        valid_data = [(idx, energy) for idx, energy in zip(atom_indices, formation_energies) if energy is not None]
        if not valid_data:
            print("No valid doping calculations were completed.")
            return

        valid_atom_idx, valid_energies = zip(*valid_data)

        # Calculate statistics
        min_energy = min(valid_energies)
        max_energy = max(valid_energies)
        min_idx = valid_atom_idx[valid_energies.index(min_energy)]
        max_idx = valid_atom_idx[valid_energies.index(max_energy)]
        avg_energy = np.mean(valid_energies)
        std_energy = np.std(valid_energies)

        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list("formation_energy",
                                                 ["blue", "green", "yellow", "red"])

        # Plot structure with atoms colored by formation energy
        fig, ax = plt.subplots(figsize=(12, 10))

        # First, draw connections from doping sites to their neighbors
        max_bond_distance = 1.8  # Standard C-C bond distance threshold
        for idx in atom_indices:
            if idx in energy_map and energy_map[idx] is not None:
                # Find neighboring atoms within bonding distance
                pos_doping = base_atoms.positions[idx]

                # Normalize energy for coloring
                norm_energy = (energy_map[idx] - min_energy) / (
                            max_energy - min_energy) if max_energy > min_energy else 0.5
                color = cmap(norm_energy)

                # Draw lines to all neighboring atoms
                for neighbor_idx, neighbor_atom in enumerate(base_atoms):
                    if neighbor_idx != idx:
                        distance = base_atoms.get_distance(idx, neighbor_idx)
                        if distance <= max_bond_distance:
                            pos_neighbor = base_atoms.positions[neighbor_idx]
                            ax.plot([pos_doping[0], pos_neighbor[0]],
                                   [pos_doping[1], pos_neighbor[1]],
                                   color=color, linewidth=2.5, alpha=0.7, zorder=5)

        # Now plot atoms colored by formation energy
        for i, atom in enumerate(base_atoms):
            if i in energy_map and energy_map[i] is not None:
                norm_energy = (energy_map[i] - min_energy) / (max_energy - min_energy) if max_energy > min_energy else 0.5
                color = cmap(norm_energy)
                # Different marker size for H atoms vs heavy atoms
                marker_size = 50 if atom.symbol == 'H' else 100
            else:
                color = 'gray'
                marker_size = 50 if atom.symbol == 'H' else 100

            ax.scatter(atom.position[0], atom.position[1], c=[color], s=marker_size,
                      edgecolors='black', linewidths=1, zorder=10)

            if show_atom_idx:
                ax.text(atom.position[0], atom.position[1], str(i),
                        fontsize=settings.ATOM_IDX_FONTSIZE, zorder=15)

        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_energy, max_energy))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Formation Energy (eV)')

        ax.set_aspect('equal')
        title_suffix = " (C + H sites)" if replace_H else ""
        ax.set_title(f'{dopant_element} Doping Formation Energy Map - {molecule_name}{title_suffix}')
        ax.set_xlabel('X (A)')
        ax.set_ylabel('Y (A)')

        plt.savefig(os.path.join(results_dir, "formation_energy_map.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Create Gaussian distribution plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        hist, bins, _ = ax2.hist(valid_energies, bins=20, density=True, alpha=0.6, color='skyblue')

        x = np.linspace(min_energy - 0.5, max_energy + 0.5, 1000)
        gaussian = stats.norm.pdf(x, avg_energy, std_energy)
        ax2.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian Fit\nμ={avg_energy:.3f} eV\nσ={std_energy:.3f} eV')

        ax2.axvline(avg_energy, color='k', linestyle='--', alpha=0.5, label='Mean')
        ax2.axvline(avg_energy + std_energy, color='k', linestyle=':', alpha=0.5, label='+1σ')
        ax2.axvline(avg_energy - std_energy, color='k', linestyle=':', alpha=0.5, label='-1σ')

        ax2.set_xlabel('Formation Energy (eV)')
        ax2.set_ylabel('Probability Density')
        title_suffix = " (C + H sites)" if replace_H else ""
        ax2.set_title(f'{dopant_element} Doping Formation Energy Distribution - {molecule_name}{title_suffix}')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.savefig(os.path.join(results_dir, "energy_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Save distribution data
        with open(os.path.join(results_dir, "energy_distribution.txt"), 'w') as f:
            f.write(f"# {dopant_element} Doping Formation Energy Distribution\n")
            f.write(f"# Mean: {avg_energy:.6f} eV\n")
            f.write(f"# Std Dev: {std_energy:.6f} eV\n")
            f.write(f"# Min: {min_energy:.6f} eV (atom {min_idx})\n")
            f.write(f"# Max: {max_energy:.6f} eV (atom {max_idx})\n")
            f.write("# Format: energy_value probability_density\n")
            for i in range(len(x)):
                f.write(f"{x[i]:.6f} {gaussian[i]:.6f}\n")

        # Write summary file
        with open(os.path.join(results_dir, "summary.txt"), 'w') as f:
            f.write(f"{dopant_element} Doping Formation Energy Analysis for {molecule_name}\n")
            f.write(f"----------------------------------------\n")
            f.write(f"Total atoms analyzed: {len(atom_indices)}\n")
            f.write(f"Valid calculations: {len(valid_energies)}\n")
            f.write(f"Minimum formation energy: {min_energy:.3f} eV at atom index {min_idx}\n")
            f.write(f"Maximum formation energy: {max_energy:.3f} eV at atom index {max_idx}\n")
            f.write(f"Average formation energy: {avg_energy:.3f} eV\n")
            f.write(f"Standard deviation: {std_energy:.3f} eV\n")

            sorted_data = sorted(valid_data, key=lambda x: x[1])

            f.write(f"\nTop 5 most favorable doping sites (lowest energy):\n")
            for i in range(min(5, len(sorted_data))):
                idx, energy = sorted_data[i]
                f.write(f"  Atom {idx}: {energy:.3f} eV\n")

            f.write(f"\nTop 5 least favorable doping sites (highest energy):\n")
            for i in range(min(5, len(sorted_data))):
                idx, energy = sorted_data[-(i+1)]
                f.write(f"  Atom {idx}: {energy:.3f} eV\n")

            f.write(f"----------------------------------------\n")


def main():
    """
    Doping formation-energy driver.

    Two modes:
      1. "legacy"    — compare a pristine GQD against one OR MORE pre-made doped
                       GQDs (dynamic mu, pristine relaxed once, detailed logs +
                       file output, mirrors the functionalization script).
      2. "automated" — scan every substitution site of a GQD for a dopant element
                       and build an energy map (kept as-is).
    """
    calc, calc_name = setup_calculator()

    # ============================================================
    # CONFIGURATION
    # ============================================================
    mode = "legacy"  # "legacy" or "automated"

    # ============================================================
    # MODE 1: Legacy - compare pristine vs one or more pre-made doped GQDs
    # ============================================================
    if mode == "legacy":
        print("\n1. LEGACY MODE - DOPED GQDs:")

        # Both pristine_key and every doped_key must exist in config.py.
        # Several doped structures can be compared in one run; the pristine
        # reference is relaxed only ONCE and shared across all of them.
        pristine_key = "GQD_HEX_2_2"
        doped_keys = ["GQD_HEX_2_2_NH2"]

        # ---- which elements CHANGE across all comparisons (build mu only for these) ----
        # Only elements whose count differs contribute a Δn·μ term, so e.g. an
        # unchanged carbon backbone never triggers a graphene relaxation.
        comp_pristine = _composition(read(molecules_data[pristine_key]["path"]))
        needed_elements = set()
        for dkey in doped_keys:
            comp_d = _composition(read(molecules_data[dkey]["path"]))
            for el in set(comp_pristine) | set(comp_d):
                if comp_d.get(el, 0) != comp_pristine.get(el, 0):
                    needed_elements.add(el)

        print(f"\n{'=' * 60}")
        print("CHEMICAL POTENTIALS (computed with the active calculator)")
        print(f"{'=' * 60}")
        mu = build_chemical_potentials(calc, calc_name, needed_elements)

        # Relax pristine ONCE: every comparison shares the same reference. The
        # relaxation is cached on disk, so repeated runs reuse it entirely.
        print(f"\n{'=' * 60}")
        print(f"PRISTINE REFERENCE: {pristine_key}")
        print(f"{'=' * 60}")
        pristine, _ = _load_centered(pristine_key, calc)
        pristine, E_pristine = cache.load_or_relax(
            pristine, pristine_key, calc, calc_name, fmax=FMAX,
            cell=molecules_data[pristine_key]["cell"],
            label=f"pristine ({pristine_key})")
        comp_pristine = _composition(pristine)

        results = {}
        for doped_key in doped_keys:
            results[doped_key] = compare_doping(
                pristine_key, pristine, E_pristine, comp_pristine,
                doped_key, calc, calc_name, mu)

        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        for doped_key, e_form in results.items():
            print(f"  {doped_key:28s}  E_form = {e_form:.4f} eV")

    # ============================================================
    # MODE 2: Automated - scan all doping sites for one dopant element
    # ============================================================
    elif mode == "automated":
        print("\n2. AUTOMATED MODE - Doping Site Analysis:")

        calculator = DopingFormationCalculator(calc, fmax=FMAX, calc_name=calc_name)

        # Configuration
        molecule_name = "GQD_HEX_3_3"  # Choose which molecule to analyze (must be an ACTIVE key in config.py)
        dopant_element = "B"   # Choose dopant: 'Li', 'N', 'B', 'P', 'S', 'O', 'F', 'Cl'
        show_atom_idx = True    # Show atom indices on plots
        excluded_atoms = []     # Optionally exclude specific atoms
        replace_H = False       # Set True to also analyze H replacement (wider energy map)

        mol_path = molecules_data[molecule_name]["path"]

        atom_indices, formation_energies = calculator.analyze_doping_sites(
            mol_path=mol_path,
            dopant_element=dopant_element,
            molecule_name=molecule_name,
            show_atom_idx=show_atom_idx,
            excluded_atoms=excluded_atoms,
            replace_H=replace_H
        )

        print(f"\n{'=' * 60}")
        print(f"Analysis complete!")
        print(f"Results saved to: doping_map_{molecule_name}_{dopant_element}_{calc_name}/")
        print(f"  - formation_energy_map.png: Visual energy map")
        print(f"  - energy_distribution.png: Statistical distribution")
        print(f"  - summary.txt: Detailed statistics")
        print(f"  - energy_map.txt: Raw data")
        print(f"{'=' * 60}")

    else:
        print(f"ERROR: Unknown mode '{mode}'. Use 'legacy' or 'automated'.")


if __name__ == "__main__":
    main()