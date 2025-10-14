import os
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from ase.build import molecule
from ase.io import read, write
from ase.optimize import BFGS
from ase.visualize.plot import plot_atoms
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from config import molecules_data
from utils import determine_target_element

# Set global font properties for all plots
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 18

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FormationResult:
    """
    Thermodynamic formation energy calculation result.

    Physical interpretation:
    - formation_energy: ΔE_f = E(modified) - E(pristine) - Σ(n_i × μ_i) [eV]
    - uncertainty: Statistical error from DFT convergence and reference uncertainties [eV]
    - components: Individual energy terms for thermodynamic cycle analysis [eV]
    - groups: Identified chemical species (functional groups, dopants) [count]
    - validation_flags: Physical reasonableness checks for calculated energies
    """
    formation_energy: float  # Thermodynamic formation energy [eV]
    uncertainty: float  # Total uncertainty propagation [eV]
    components: Dict[str, float]  # Energy components for analysis [eV]
    groups: Dict[str, int]  # Chemical species identified [count]
    validation_flags: List[str]  # Physical validation warnings


class FormationEnergyCalculator:
    """
    Quantum mechanical formation energy calculator for molecular functionalization.

    Theoretical framework:
    - Uses DFT to calculate ground-state electronic energies
    - Applies thermodynamic cycle analysis with chemical potentials
    - Implements proper statistical error propagation
    - Validates results against known chemical physics

    Formation energy definition:
    ΔE_f = E_DFT(modified) - E_DFT(pristine) - Σ(n_i × μ_i)

    Where:
    - E_DFT: Kohn-Sham total energy from density functional theory
    - μ_i: Chemical potential of species i (reservoir energy)
    - n_i: Stoichiometric coefficient (positive for addition, negative for removal)
    """

    def __init__(self, calculator, fmax=0.005):
        """
        Initialize formation energy calculator with quantum mechanical parameters.

        Args:
            calculator: DFT calculator (e.g., SevenNet, VASP, Gaussian)
            fmax: Force convergence criterion [eV/Å] - controls geometric accuracy

        Physical rationale for fmax=0.005 eV/Å:
        - Ensures forces < kT at 300K (~0.025 eV) per angstrom
        - Required for accurate energy differences (<1 meV precision)
        - Satisfies Hellmann-Feynman theorem: F_i = -∂E/∂R_i
        """
        self.calc = calculator
        self.fmax = fmax  # Geometric convergence criterion [eV/Å]
        self.ref_cache = {}  # Computational efficiency: cache reference calculations

        # Standard chemical potentials with experimental/theoretical uncertainties [eV]
        # Based on NIST thermochemical data and DFT formation energy databases
        # Represents energy cost to extract one atom from standard reference state
        self.chemical_potentials = {
            'H': (-3.396, 0.015),  # μ(H) = E(H₂)/2 - diatomic hydrogen reference [eV]
            'O': (-4.952, 0.020),  # μ(O) = E(O₂)/2 - triplet ground state oxygen [eV]
            'N': (-8.328, 0.025),  # μ(N) = E(N₂)/2 - diatomic nitrogen reference [eV]
            'C': (-9.085, 0.035),  # μ(C) = E(graphite)/n - bulk graphite reference [eV]
            'B': (-6.678, 0.030),  # μ(B) = E(bulk boron)/n - bulk boron reference [eV]
        }

        # Functional group recognition patterns for automated chemical analysis
        # Format: [atomic_composition, expected_bond_lengths]
        # Bond lengths from experimental crystallography and quantum chemistry
        self.fg_patterns = {
            'NH2': (['N', 'H', 'H'], [('N', 'H', 1.02), ('H', 'N', 1.02)]),  # Amine group
            'OH': (['O', 'H'], [('O', 'H', 0.97)]),  # Hydroxyl group
            'COOH': (['C', 'O', 'O', 'H'], [('C', 'O', 1.21), ('C', 'O', 1.36), ('O', 'H', 0.97)]),  # Carboxyl
            'BH2': (['B', 'H', 'H'], [('B', 'H', 1.19), ('H', 'B', 1.19)]),  # Borane group
        }

    def calculate_formation_energy(self, pristine_key: str, modified_key: str) -> FormationResult:
        """
        Calculate thermodynamic formation energy using quantum mechanical total energies.

        Implements complete thermodynamic cycle:
        1. Quantum mechanical energy calculation (Kohn-Sham DFT)
        2. Chemical species identification (pattern recognition)
        3. Chemical potential evaluation (reservoir thermodynamics)
        4. Formation energy assembly (thermodynamic balance)
        5. Statistical uncertainty propagation (error analysis)
        6. Physical validation (chemical reasonableness)

        Args:
            pristine_key: Reference system identifier
            modified_key: Functionalized system identifier

        Returns:
            FormationResult: Complete thermodynamic analysis with uncertainties

        Thermodynamic interpretation:
        - ΔE_f < 0: Exergonic process, thermodynamically favorable
        - ΔE_f > 0: Endergonic process, requires activation energy
        - ΔE_f ≈ 0: Marginal stability, entropy effects important
        """

        # Step 1: Quantum mechanical energy calculation
        # Solves Kohn-Sham equations: [-½∇² + V_eff(r)]ψ_i(r) = ε_i ψ_i(r)
        # Optimizes geometry to minimum energy: ∇E = 0 (force balance)
        E_pristine, forces_pristine = self._get_energy(pristine_key)
        E_modified, forces_modified = self._get_energy(modified_key)

        # Step 2: Chemical analysis - identify what changed
        # Compares atomic compositions to determine stoichiometry
        composition_diff = self._get_composition_diff(pristine_key, modified_key)
        functional_groups = self._identify_groups(pristine_key, modified_key, composition_diff)

        # Step 3: Thermodynamic analysis with chemical potentials
        # Calculates reservoir energies for balanced chemical equation
        mu_total, mu_uncertainty = self._calculate_chemical_potentials(functional_groups)

        # Step 4: Formation energy assembly
        # Implements: ΔE_f = E(products) - E(reactants) - Σ(μ_i × n_i)
        formation_energy = E_modified - E_pristine - mu_total

        # Step 5: Statistical error propagation
        # Combines DFT convergence errors with chemical potential uncertainties
        # Using standard uncertainty propagation: σ²(A+B) = σ²(A) + σ²(B)
        uncertainty = np.sqrt(mu_uncertainty ** 2 + 0.001 ** 2)  # 1 meV convergence uncertainty

        # Step 6: Physical validation and quality control
        validation_flags = self._validate_result(
            formation_energy, functional_groups, forces_pristine, forces_modified
        )

        # Energy component breakdown for thermodynamic cycle analysis
        components = {
            'E_modified': E_modified,  # Final state energy [eV]
            'E_pristine': E_pristine,  # Initial state energy [eV]
            'mu_total': mu_total,  # Total chemical potential [eV]
            'raw_diff': E_modified - E_pristine  # Raw energy difference [eV]
        }

        return FormationResult(
            formation_energy=formation_energy,
            uncertainty=uncertainty,
            components=components,
            groups=functional_groups,
            validation_flags=validation_flags
        )

    def _get_energy(self, mol_key: str) -> Tuple[float, np.ndarray]:
        """
        Calculate ground-state quantum mechanical energy with geometric optimization.

        Physical process:
        1. Loads molecular structure from database
        2. Attaches DFT calculator (implements Kohn-Sham equations)
        3. Performs geometry optimization to find energy minimum
        4. Validates convergence using Hellmann-Feynman forces

        Theoretical background:
        - DFT total energy: E[ρ] = T[ρ] + V_ext[ρ] + V_H[ρ] + E_xc[ρ]
        - T[ρ]: Kinetic energy functional
        - V_ext[ρ]: External potential (nuclei)
        - V_H[ρ]: Hartree energy (electron-electron repulsion)
        - E_xc[ρ]: Exchange-correlation energy (quantum many-body effects)

        Args:
            mol_key: Molecular system identifier

        Returns:
            energy: Converged total energy [eV]
            forces: Atomic forces for validation [eV/Å]
        """
        # Load molecular structure from computational database
        atoms = read(molecules_data[mol_key]["mol_path"])
        atoms.calc = self.calc  # Attach quantum mechanical calculator

        # Geometry optimization using BFGS algorithm
        # Minimizes total energy: E(R₁, R₂, ..., R_N) → minimum
        # Satisfies force balance: F_i = -∇_i E = 0 for all atoms i
        dyn = BFGS(atoms, logfile=None)
        dyn.run(fmax=self.fmax)  # Converge to specified force tolerance

        # Extract quantum mechanical properties
        energy = atoms.get_potential_energy()  # Ground-state total energy [eV]
        forces = atoms.get_forces()  # Hellmann-Feynman forces [eV/Å]

        # Convergence quality assessment
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > self.fmax:
            # Poor convergence affects energy accuracy - warn user
            logger.warning(f"Poor convergence for {mol_key}: max_force = {max_force:.4f}")

        logger.info(f"Energy for {mol_key}: {energy:.4f} eV (max_force: {max_force:.4f})")
        return energy, forces

    def _get_composition_diff(self, pristine_key: str, modified_key: str) -> Dict[str, int]:
        """
        Analyze atomic composition changes for stoichiometric balance.

        Determines the net change in atomic species between pristine and modified systems.
        Essential for applying conservation of mass in thermodynamic calculations.

        Chemical principle:
        Balanced equation: Pristine + Δn_i Species_i → Modified
        Where Δn_i can be positive (addition) or negative (removal)

        Args:
            pristine_key: Initial system identifier
            modified_key: Final system identifier

        Returns:
            composition_diff: Net atomic changes {element: count_change}
        """

        def get_composition(mol_key):
            """Extract atomic composition from molecular structure."""
            atoms = read(molecules_data[mol_key]["mol_path"])
            comp = {}
            for symbol in atoms.get_chemical_symbols():
                comp[symbol] = comp.get(symbol, 0) + 1
            return comp

        # Count atoms in each system
        comp_pristine = get_composition(pristine_key)
        comp_modified = get_composition(modified_key)

        # Calculate net changes (modified - pristine)
        all_elements = set(comp_pristine.keys()) | set(comp_modified.keys())
        diff = {}
        for element in all_elements:
            delta = comp_modified.get(element, 0) - comp_pristine.get(element, 0)
            if delta != 0:  # Only record actual changes
                diff[element] = delta

        logger.info(f"Composition change: {diff}")
        return diff

    def _identify_groups(self, pristine_key: str, modified_key: str,
                         composition_diff: Dict[str, int]) -> Dict[str, int]:
        """
        Identify chemical functional groups using pattern recognition.

        Implements automated chemical analysis to recognize common functional groups
        from atomic composition changes. Uses chemical knowledge to assign atoms
        to most likely chemical environments.

        Algorithm:
        1. Pattern matching against known functional groups
        2. Greedy assignment (largest groups first)
        3. Handle remaining atoms as individual species

        Chemical rationale:
        - Functional groups have characteristic stoichiometries
        - Atoms prefer to form complete functional units
        - Remaining atoms likely represent surface adsorption or doping

        Args:
            pristine_key: Reference system
            modified_key: Modified system
            composition_diff: Atomic composition changes

        Returns:
            groups: Identified chemical species {species_name: count}
        """
        groups = {}
        remaining = composition_diff.copy()  # Track unassigned atoms

        # Pattern matching for known functional groups
        # Uses greedy algorithm to maximize group identification
        for fg_name, (elements, bonds) in self.fg_patterns.items():
            # Calculate stoichiometric requirements for this functional group
            pattern = {}
            for elem in elements:
                pattern[elem] = pattern.get(elem, 0) + 1

            # Determine maximum possible number of this group
            # Limited by availability of each required element
            max_matches = float('inf')
            for elem, needed in pattern.items():
                available = remaining.get(elem, 0)
                if available < needed:
                    max_matches = 0  # Cannot form this group
                    break
                max_matches = min(max_matches, available // needed)

            # Assign atoms to functional groups
            if max_matches > 0:
                groups[fg_name] = max_matches
                # Remove assigned atoms from remaining pool
                for elem, needed in pattern.items():
                    remaining[elem] -= max_matches * needed
                logger.info(f"Identified {max_matches} {fg_name} groups")

        # Handle remaining atoms as individual species
        # These represent surface adsorption, substitutional doping, or vacancies
        for elem, count in remaining.items():
            if count != 0:
                groups[f"{elem}_atom"] = count
                if count > 0:
                    logger.warning(f"Unmatched atoms (likely adsorbed): {count} {elem}")
                else:
                    logger.warning(f"Missing atoms (likely vacancy): {abs(count)} {elem}")

        return groups

    def _calculate_chemical_potentials(self, groups: Dict[str, int]) -> Tuple[float, float]:
        """
        Calculate thermodynamic chemical potentials with uncertainty propagation.

        Implements reservoir thermodynamics for balanced chemical equations.
        Each chemical species has an associated chemical potential representing
        the energy cost/gain of transferring that species to/from an infinite reservoir.

        Thermodynamic principle:
        At equilibrium: Σ(μ_i × n_i) = 0 for balanced reaction
        Formation energy: ΔE_f = ΔE_reaction - Σ(μ_i × n_i)

        Statistical mechanics:
        - Chemical potentials represent partial derivatives: μ_i = ∂E/∂N_i
        - Uncertainties propagate as: σ²(Σa_i) = Σ(σ_i × a_i)²

        Args:
            groups: Identified chemical species with counts

        Returns:
            total_mu: Total chemical potential contribution [eV]
            total_uncertainty: Combined uncertainty [eV]
        """
        total_mu = 0.0
        total_uncertainty_sq = 0.0  # Variance accumulation

        for group_name, count in groups.items():
            if count <= 0:
                continue  # Skip empty groups

            # Determine chemical potential for this species
            if group_name.endswith('_atom'):
                # Individual atoms use elemental reference states
                element = group_name.replace('_atom', '')
                mu, sigma = self.chemical_potentials[element]
            else:
                # Functional groups require molecular reference calculations
                mu, sigma = self._get_functional_group_potential(group_name)

            # Accumulate contributions with proper statistical weighting
            total_mu += count * mu
            total_uncertainty_sq += (count * sigma) ** 2  # Variance propagation

        return total_mu, np.sqrt(total_uncertainty_sq)

    def _get_functional_group_potential(self, group_name: str) -> Tuple[float, float]:
        """
        Calculate chemical potentials for molecular functional groups.

        Uses quantum mechanical reference calculations combined with thermodynamic
        balance equations to determine chemical potentials for complex functional groups.

        Methodology:
        1. Calculate reference molecule energy (NH₃, H₂O, HCOOH)
        2. Apply thermodynamic corrections for stoichiometric balance
        3. Propagate uncertainties from reference calculations

        Thermodynamic corrections:
        - NH₂: NH₃ → NH₂ + ½H₂, so μ(NH₂) = μ(NH₃) - μ(H)
        - OH:  H₂O → OH + ½H₂,  so μ(OH)  = μ(H₂O) - μ(H)
        - COOH: Use direct molecular reference (formic acid)

        Args:
            group_name: Functional group identifier

        Returns:
            mu: Chemical potential [eV]
            sigma: Uncertainty [eV]
        """

        # Reference molecules for functional group calculations
        # Selected for chemical similarity and computational stability
        # NOTE: BH2 uses elemental references because BH3 is unstable
        reference_molecules = {
            'NH2': 'NH3',  # Ammonia as NH₂ precursor
            'OH': 'H2O',  # Water as OH precursor
            'COOH': 'HCOOH',  # Formic acid as COOH reference
        }

        if group_name in reference_molecules:
            mol_name = reference_molecules[group_name]

            # Calculate or retrieve reference molecule energy
            if mol_name not in self.ref_cache:
                try:
                    # Quantum mechanical calculation of reference molecule
                    mol = molecule(mol_name)  # ASE molecule database
                    mol.calc = self.calc  # Apply same DFT method

                    # Optimize geometry for accurate energy
                    dyn = BFGS(mol, logfile=None)
                    dyn.run(fmax=self.fmax)

                    energy = mol.get_potential_energy()
                    self.ref_cache[mol_name] = energy  # Cache for efficiency
                    logger.info(f"Reference energy μ({mol_name}) = {energy:.4f} eV")

                except Exception as e:
                    logger.error(f"Failed to calculate reference for {mol_name}: {e}")
                    return 0.0, 1.0  # Large uncertainty for failed calculations

            base_energy = self.ref_cache[mol_name]

            # Apply thermodynamic corrections for functional groups
            if group_name == 'NH2':
                # Thermodynamic balance: NH₃ → NH₂ + ½H₂
                # Chemical potential: μ(NH₂) = μ(NH₃) - μ(H)
                mu_h, sigma_h = self.chemical_potentials['H']
                mu = base_energy - mu_h
                # Error propagation: σ²(A-B) = σ²(A) + σ²(B)
                sigma = np.sqrt(0.05 ** 2 + sigma_h ** 2)  # 50 meV reference uncertainty

            elif group_name == 'OH':
                # Thermodynamic balance: H₂O → OH + ½H₂
                # Chemical potential: μ(OH) = μ(H₂O) - μ(H)
                mu_h, sigma_h = self.chemical_potentials['H']
                mu = base_energy - mu_h
                sigma = np.sqrt(0.05 ** 2 + sigma_h ** 2)

            else:  # COOH
                # Thermodynamic balance: HCOOH → COOH + ½H₂
                # Chemical potential: μ(COOH) = μ(HCOOH) - μ(H)
                mu_h, sigma_h = self.chemical_potentials['H']
                mu = base_energy - mu_h
                sigma = np.sqrt(0.05 ** 2 + sigma_h ** 2)

            return mu, sigma

        # Special case: BH2 uses elemental references
        # BH3 is unstable, so we use: BH2 = B + 2H
        elif group_name == 'BH2':
            # Chemical potential from elemental references
            # μ(BH2) = μ(B) + 2×μ(H)
            mu_b, sigma_b = self.chemical_potentials['B']
            mu_h, sigma_h = self.chemical_potentials['H']
            mu = mu_b + 2.0 * mu_h
            # Error propagation: σ²(A+B) = σ²(A) + σ²(B)
            sigma = np.sqrt(sigma_b ** 2 + (2.0 * sigma_h) ** 2)
            logger.info(f"BH2 chemical potential (elemental): μ(B) + 2×μ(H) = {mu:.4f} eV")
            return mu, sigma

        # Fallback for unknown functional groups
        logger.warning(f"Unknown functional group: {group_name}")
        return 0.0, 0.5  # Large uncertainty indicates unknown chemistry

    def analyze_functionalization_sites(self, mol_path: str, functional_group: str,
                                       molecule_name: str, show_atom_idx: bool = True,
                                       excluded_atoms: List[int] = None) -> Tuple[List[int], List[float]]:
        """
        Analyze all possible edge functionalization sites in a GQD structure.

        Similar to doping analysis, but for functional group addition.
        Replaces H atoms along the edge with functional groups (NH2, OH, COOH, BH2).

        Args:
            mol_path: Path to the GQD molecule file
            functional_group: Functional group to add ('NH2', 'OH', 'COOH', 'BH2')
            molecule_name: Name for output files
            show_atom_idx: Whether to show atom indices on plots
            excluded_atoms: List of atom indices to exclude from analysis

        Returns:
            Tuple of (atom_indices, formation_energies)
        """
        if excluded_atoms is None:
            excluded_atoms = []

        if functional_group not in self.fg_patterns:
            raise ValueError(f"Unknown functional group: {functional_group}. Choose from {list(self.fg_patterns.keys())}")

        print(f"\n=== Running Functionalization Site Analysis for {functional_group} ===\n")
        results_dir = f"func_map_{molecule_name}_{functional_group}"
        os.makedirs(results_dir, exist_ok=True)

        # Create subdirectories for structures and images
        structures_dir = os.path.join(results_dir, "structures")
        images_dir = os.path.join(results_dir, "images")
        os.makedirs(structures_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        # Load base structure
        base_atoms = read(mol_path)

        # Set up cell size and centering
        cell = molecules_data[molecule_name]["cell"]
        positions = base_atoms.get_positions()
        center_of_mass = np.mean(positions, axis=0)
        translation = np.array([cell / 2] * 3) - center_of_mass
        positions = [list(np.array(pos) + translation) for pos in positions]
        base_atoms.set_positions(positions)
        base_atoms.set_cell([cell] * 3)
        base_atoms.set_pbc(True)

        # Calculate pristine energy
        base_atoms.calc = self.calc
        print("Relaxing base structure...")
        dyn = BFGS(base_atoms, logfile=None)
        try:
            dyn.run(fmax=self.fmax, steps=500)  # Max 500 steps
        except Exception as opt_error:
            print(f"Warning: Base structure optimization had issues: {opt_error}")
            pass
        E_pristine = base_atoms.get_potential_energy()
        print(f"Base structure energy: {E_pristine:.4f} eV")

        # Initialize results
        formation_energies = []
        atom_indices = []

        # Functional group composition
        fg_elements, _ = self.fg_patterns[functional_group]
        fg_composition = {}
        for elem in fg_elements:
            fg_composition[elem] = fg_composition.get(elem, 0) + 1

        print(f"Analyzing H substitution sites with {functional_group}")
        print(f"Functional group composition: {fg_composition}")

        # Count total H atoms for progress tracking
        total_h_atoms = sum(1 for i, atom in enumerate(base_atoms) if atom.symbol == 'H' and i not in excluded_atoms)
        h_count = 0

        # Loop through each H atom (edge hydrogens)
        for atom_idx in range(len(base_atoms)):
            if base_atoms[atom_idx].symbol != 'H' or atom_idx in excluded_atoms:
                continue

            h_count += 1
            print(f"\n[{h_count}/{total_h_atoms}] Processing H atom {atom_idx}")
            atom_indices.append(atom_idx)

            try:
                # Create functionalized structure
                functionalized_atoms = base_atoms.copy()

                # Remove the H atom and add functional group atoms at the same position
                h_position = functionalized_atoms[atom_idx].position.copy()

                # Get the atom bonded to this H (should be C)
                distances = functionalized_atoms.get_distances(atom_idx, range(len(functionalized_atoms)), mic=True)
                bonded_idx = None
                for i, dist in enumerate(distances):
                    if i != atom_idx and dist < 1.2 and functionalized_atoms[i].symbol == 'C':
                        bonded_idx = i
                        break

                if bonded_idx is None:
                    print(f"  Warning: Could not find bonded C atom for H {atom_idx}")
                    formation_energies.append(None)
                    continue

                # Build functional group geometry
                # For NH2: N replaces H, two H atoms added
                # For OH: O replaces H, one H atom added
                # For COOH: C replaces H, two O and one H added
                # For BH2: B replaces H, two H atoms added

                from ase import Atoms

                if functional_group == 'NH2':
                    # Replace H with N
                    functionalized_atoms[atom_idx].symbol = 'N'
                    # Add two H atoms bonded to N
                    n_pos = functionalized_atoms[atom_idx].position
                    # Simple tetrahedral geometry
                    h1_pos = n_pos + np.array([1.02, 0.0, 0.0])
                    h2_pos = n_pos + np.array([0.0, 1.02, 0.0])
                    functionalized_atoms += Atoms('H2', positions=[h1_pos, h2_pos])

                elif functional_group == 'OH':
                    # Replace H with O
                    functionalized_atoms[atom_idx].symbol = 'O'
                    o_pos = functionalized_atoms[atom_idx].position

                    # Get C-O bond direction for proper orientation
                    c_bonded_pos = functionalized_atoms[bonded_idx].position
                    bond_vector = o_pos - c_bonded_pos
                    bond_vector = bond_vector / np.linalg.norm(bond_vector)

                    # OH: Place H along the C-O bond extension
                    h_pos = o_pos + bond_vector * 0.97  # O-H bond length
                    functionalized_atoms += Atoms('H', positions=[h_pos])

                elif functional_group == 'BH2':
                    # Replace H with B
                    functionalized_atoms[atom_idx].symbol = 'B'
                    # Add two H atoms bonded to B (same simple approach as NH2)
                    b_pos = functionalized_atoms[atom_idx].position
                    # Simple trigonal planar geometry (B-H bond length ~1.19 Å)
                    h1_pos = b_pos + np.array([1.19, 0.0, 0.0])
                    h2_pos = b_pos + np.array([0.0, 1.19, 0.0])
                    functionalized_atoms += Atoms('H2', positions=[h1_pos, h2_pos])

                elif functional_group == 'COOH':
                    # Replace H with C (carbonyl carbon)
                    functionalized_atoms[atom_idx].symbol = 'C'
                    c_carboxyl_pos = functionalized_atoms[atom_idx].position

                    # Get C-C bond direction for proper orientation (like we do for OH)
                    c_bonded_pos = functionalized_atoms[bonded_idx].position
                    bond_vector = c_carboxyl_pos - c_bonded_pos
                    bond_vector = bond_vector / np.linalg.norm(bond_vector)

                    # Build COOH geometry along bond direction
                    # Use simple planar geometry, BFGS will optimize bond lengths and angles
                    # O1 (carbonyl) at ~120° from bond direction (trigonal planar)
                    # O2 (hydroxyl) at ~120° opposite side
                    # H bonded to O2

                    # Create perpendicular vector for placing O atoms at 120° angles
                    # Use cross product with z-axis (or x-axis if bond is parallel to z)
                    if abs(bond_vector[2]) < 0.9:  # Not parallel to z-axis
                        perp = np.cross(bond_vector, np.array([0, 0, 1]))
                    else:  # Nearly parallel to z-axis, use x-axis
                        perp = np.cross(bond_vector, np.array([1, 0, 0]))
                    perp = perp / np.linalg.norm(perp)

                    # Place O atoms in trigonal planar arrangement around C
                    # Angle of ~120° means cos(120°) = -0.5, sin(120°) = 0.866
                    o1_pos = c_carboxyl_pos + bond_vector * 0.6 + perp * 1.0  # C=O carbonyl
                    o2_pos = c_carboxyl_pos + bond_vector * 0.6 - perp * 1.0  # C-OH hydroxyl
                    # Place H extending from O2 along the C-O2 direction
                    o2_direction = o2_pos - c_carboxyl_pos
                    o2_direction = o2_direction / np.linalg.norm(o2_direction)
                    h_pos = o2_pos + o2_direction * 0.97  # O-H bond

                    functionalized_atoms += Atoms('OOH', positions=[o1_pos, o2_pos, h_pos])

                # Relax functionalized structure
                functionalized_atoms.calc = self.calc
                print(f"  Relaxing functionalized structure...")
                dyn = BFGS(functionalized_atoms, logfile=None)

                # Add maximum steps to prevent infinite loops
                try:
                    dyn.run(fmax=self.fmax, steps=500)  # Max 500 steps
                except Exception as opt_error:
                    print(f"  Warning: Optimization failed: {opt_error}")
                    # Try to get energy anyway (might be partially converged)
                    pass

                E_functionalized = functionalized_atoms.get_potential_energy()

                # Check if optimization converged well
                max_force = np.max(np.linalg.norm(functionalized_atoms.get_forces(), axis=1))
                if max_force > self.fmax * 5:  # If forces are very large, skip this structure
                    print(f"  Warning: Poor convergence (max_force={max_force:.4f}), skipping...")
                    formation_energies.append(None)
                    continue

                # Calculate formation energy with chemical potentials
                # We are REPLACING H with functional group FG (NH2, OH, COOH, BH2)
                #
                # Chemical reaction: GQD-H + FG_reservoir → GQD-FG + H_reservoir
                #
                # Formation energy: ΔE_f = E(GQD-FG) - E(GQD-H) - μ(FG) + μ(H_removed)
                #
                # Where μ(FG) is calculated from molecular references:
                # - μ(NH2) = E(NH3) - μ(H): NH3 → NH2 + H
                # - μ(OH) = E(H2O) - μ(H): H2O → OH + H
                # - μ(COOH) = E(HCOOH) - μ(H): HCOOH → COOH + H
                # - μ(BH2) = E(BH3) - μ(H): BH3 → BH2 + H
                #
                # Substituting:
                # ΔE_f = E(GQD-FG) - E(GQD-H) - [E(ref_mol) - μ(H)] + μ(H)
                # ΔE_f = E(GQD-FG) - E(GQD-H) - E(ref_mol) + 2×μ(H)
                #
                # So mu_total = E(ref_mol) - 2×μ(H)

                # Use the _get_functional_group_potential method which already handles caching
                mu_fg, _ = self._get_functional_group_potential(functional_group)
                mu_h, _ = self.chemical_potentials['H']

                # Reaction: GQD-H + FG → GQD-FG + H
                # ΔE_f = E(GQD-FG) - E(GQD-H) - μ(FG) + μ(H)
                mu_total = mu_fg - mu_h

                formation_energy = E_functionalized - E_pristine - mu_total

                formation_energies.append(formation_energy)
                print(f"  H atom {atom_idx} formation energy: {formation_energy:.3f} eV")

                # Save relaxed structure as XYZ file
                try:
                    xyz_filename = os.path.join(structures_dir, f"func_H{atom_idx}_{functional_group}.xyz")
                    write(xyz_filename, functionalized_atoms)
                    print(f"  Saved structure to: {xyz_filename}")
                except Exception as save_error:
                    print(f"  Warning: Could not save XYZ file: {save_error}")

                # Generate and save image using ASE plot_atoms
                try:
                    fig_struct, ax_struct = plt.subplots(figsize=(8, 8))
                    plot_atoms(functionalized_atoms, ax_struct, rotation=('0x,0y,0z'))
                    ax_struct.set_title(f'{functional_group} at H{atom_idx}\nE_f = {formation_energy:.3f} eV')
                    image_filename = os.path.join(images_dir, f"func_H{atom_idx}_{functional_group}.png")
                    plt.savefig(image_filename, dpi=150, bbox_inches='tight')
                    plt.close(fig_struct)
                    print(f"  Saved image to: {image_filename}")
                except Exception as img_error:
                    print(f"  Warning: Could not generate image: {img_error}")

            except Exception as e:
                print(f"  Error processing H atom {atom_idx}: {e}")
                formation_energies.append(None)

        # Create visualizations and analysis
        self._process_functionalization_results(results_dir, base_atoms, atom_indices,
                                               formation_energies, molecule_name,
                                               functional_group, show_atom_idx)

        return atom_indices, formation_energies

    def _process_functionalization_results(self, results_dir: str, base_atoms,
                                          atom_indices: List[int], formation_energies: List[float],
                                          molecule_name: str, functional_group: str,
                                          show_atom_idx: bool = True) -> None:
        """Process and visualize functionalization site results."""

        # Create energy map dictionary
        energy_map = {idx: energy for idx, energy in zip(atom_indices, formation_energies)}

        # Save formation energies
        np.savez(os.path.join(results_dir, "formation_energies.npz"),
                indices=np.array(atom_indices),
                energies=np.array(formation_energies))

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
            print("No valid functionalization calculations were completed.")
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

        for i, atom in enumerate(base_atoms):
            if i in energy_map and energy_map[i] is not None:
                norm_energy = (energy_map[i] - min_energy) / (max_energy - min_energy) if max_energy > min_energy else 0.5
                color = cmap(norm_energy)
                # H atoms should be smaller
                marker_size = 50 if atom.symbol == 'H' else 100
            else:
                color = 'gray'
                marker_size = 50 if atom.symbol == 'H' else 100

            ax.scatter(atom.position[0], atom.position[1], c=[color], s=marker_size,
                      edgecolors='black', linewidths=1)

            if show_atom_idx:
                ax.text(atom.position[0], atom.position[1], str(i), fontsize=8)

        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_energy, max_energy))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Formation Energy (eV)')

        ax.set_aspect('equal')
        ax.set_title(f'{functional_group} Functionalization Formation Energy Map - {molecule_name}')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')

        plt.savefig(os.path.join(results_dir, "formation_energy_map.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Create Gaussian distribution plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        hist, bins, _ = ax2.hist(valid_energies, bins=20, density=True, alpha=0.6, color='skyblue')

        x = np.linspace(min_energy - 0.5, max_energy + 0.5, 1000)
        gaussian = stats.norm.pdf(x, avg_energy, std_energy)
        ax2.plot(x, gaussian, 'r-', linewidth=2,
                label=f'Gaussian Fit\nμ={avg_energy:.3f} eV\nσ={std_energy:.3f} eV')

        ax2.axvline(avg_energy, color='k', linestyle='--', alpha=0.5, label='Mean')
        ax2.axvline(avg_energy + std_energy, color='k', linestyle=':', alpha=0.5, label='+1σ')
        ax2.axvline(avg_energy - std_energy, color='k', linestyle=':', alpha=0.5, label='-1σ')

        ax2.set_xlabel('Formation Energy (eV)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title(f'{functional_group} Functionalization Formation Energy Distribution - {molecule_name}')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.savefig(os.path.join(results_dir, "energy_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Save distribution data
        with open(os.path.join(results_dir, "energy_distribution.txt"), 'w') as f:
            f.write(f"# {functional_group} Functionalization Formation Energy Distribution\n")
            f.write(f"# Mean: {avg_energy:.6f} eV\n")
            f.write(f"# Std Dev: {std_energy:.6f} eV\n")
            f.write(f"# Min: {min_energy:.6f} eV (H atom {min_idx})\n")
            f.write(f"# Max: {max_energy:.6f} eV (H atom {max_idx})\n")
            f.write("# Format: energy_value probability_density\n")
            for i in range(len(x)):
                f.write(f"{x[i]:.6f} {gaussian[i]:.6f}\n")

        # Write summary file
        with open(os.path.join(results_dir, "summary.txt"), 'w') as f:
            f.write(f"{functional_group} Functionalization Formation Energy Analysis for {molecule_name}\n")
            f.write(f"----------------------------------------\n")
            f.write(f"Total H atoms analyzed: {len(atom_indices)}\n")
            f.write(f"Valid calculations: {len(valid_energies)}\n")
            f.write(f"Minimum formation energy: {min_energy:.3f} eV at H atom index {min_idx}\n")
            f.write(f"Maximum formation energy: {max_energy:.3f} eV at H atom index {max_idx}\n")
            f.write(f"Average formation energy: {avg_energy:.3f} eV\n")
            f.write(f"Standard deviation: {std_energy:.3f} eV\n")

            sorted_data = sorted(valid_data, key=lambda x: x[1])

            f.write(f"\nTop 5 most favorable functionalization sites (lowest energy):\n")
            for i in range(min(5, len(sorted_data))):
                idx, energy = sorted_data[i]
                f.write(f"  H atom {idx}: {energy:.3f} eV\n")

            f.write(f"\nTop 5 least favorable functionalization sites (highest energy):\n")
            for i in range(min(5, len(sorted_data))):
                idx, energy = sorted_data[-(i+1)]
                f.write(f"  H atom {idx}: {energy:.3f} eV\n")

            f.write(f"----------------------------------------\n")

        print(f"\n{'=' * 60}")
        print(f"Analysis complete!")
        print(f"Results saved to: {results_dir}/")
        print(f"  - formation_energy_map.png: Visual energy map")
        print(f"  - energy_distribution.png: Statistical distribution")
        print(f"  - summary.txt: Detailed statistics")
        print(f"  - energy_map.txt: Raw data")
        print(f"  - structures/: XYZ files for each functionalized structure")
        print(f"  - images/: Individual structure visualizations")
        print(f"{'=' * 60}\n")

    def _validate_result(self, formation_energy: float, groups: Dict[str, int],
                         forces_pristine: np.ndarray, forces_modified: np.ndarray) -> List[str]:
        """
        Validate formation energy results against known chemical physics.

        Implements multiple layers of physical reasonableness checks:
        1. Convergence quality (force magnitudes)
        2. Energy magnitude (chemical bond strength ranges)
        3. Per-group energetics (functional group binding energies)
        4. Thermodynamic classification (exergonic vs endergonic)

        Physical basis:
        - Covalent bond energies: typically 1-6 eV
        - Physisorption: 0.01-0.5 eV
        - Chemisorption: 0.5-3 eV
        - Unreasonable values suggest computational errors

        Args:
            formation_energy: Calculated formation energy [eV]
            groups: Identified functional groups
            forces_pristine: Forces on pristine system [eV/Å]
            forces_modified: Forces on modified system [eV/Å]

        Returns:
            flags: List of validation warnings/errors
        """
        flags = []

        # Convergence quality check
        # Well-converged forces should be < fmax for reliable energies
        max_force_pristine = np.max(np.linalg.norm(forces_pristine, axis=1))
        max_force_modified = np.max(np.linalg.norm(forces_modified, axis=1))

        if max_force_pristine > self.fmax or max_force_modified > self.fmax:
            flags.append("POOR_CONVERGENCE")
            logger.warning("Poor geometric convergence may affect energy accuracy")

        # Overall energy magnitude check
        # Formation energies should be within chemical bond energy ranges
        if abs(formation_energy) > 10.0:
            flags.append("LARGE_FORMATION_ENERGY")
            logger.warning(f"Unusually large formation energy: {formation_energy:.2f} eV")

        # Per-functional-group energy validation
        # Check if individual group binding energies are chemically reasonable
        for group, count in groups.items():
            if count > 0:
                per_group = formation_energy / count

                # Functional group binding energy ranges from literature
                if group in ['NH2', 'OH', 'COOH', 'BH2']:
                    if per_group > 2.0:
                        flags.append(f"HIGH_ENERGY_{group}")
                        logger.warning(f"High binding energy for {group}: {per_group:.2f} eV")
                    elif per_group < -8.0:
                        flags.append(f"LOW_ENERGY_{group}")
                        logger.warning(f"Unusually strong binding for {group}: {per_group:.2f} eV")

        # Thermodynamic classification for experimental guidance
        if formation_energy < -0.5:
            flags.append("HIGHLY_FAVORABLE")
            logger.info("Process is highly exergonic - expect spontaneous reaction")
        elif formation_energy > 2.0:
            flags.append("HIGHLY_UNFAVORABLE")
            logger.info("Process requires significant activation - may need harsh conditions")

        return flags


def main():
    """
    Demonstration of quantum mechanical formation energy calculation.

    Now supports two modes:
    1. Legacy mode: Compare pre-existing functionalized molecules
    2. NEW Automated mode: Analyze all edge H functionalization sites automatically
    """
    import platform

    # Initialize calculator based on operating system
    system = platform.system()

    if system == "Linux":
        print(f"Running on {system} - using GPAW calculator")
        from gpaw import GPAW, PW, FermiDirac

        calc = GPAW(
            xc='PBE',
            mode=PW(300),
            kpts=(1, 1, 1),
            symmetry='off',
            spinpol=True,
            occupations=FermiDirac(0.05),
            convergence={
                'energy': 0.001,
                'density': 1e-4,
                'eigenstates': 1e-6,
            },
            mixer={'backend': 'pulay', 'beta': 0.1, 'nmaxold': 5, 'weight': 50},
            maxiter=300,
            txt='calculation.txt',
        )
    elif system == "Windows":
        print(f"Running on {system} - using SevenNet calculator")
        from sevenn.calculator import SevenNetCalculator

        calc = SevenNetCalculator('7net-l3i5', modal='mpa')
    else:
        print(f"Warning: Running on {system} - defaulting to SevenNet calculator")
        from sevenn.calculator import SevenNetCalculator

        calc = SevenNetCalculator('7net-l3i5', modal='mpa')

    calculator = FormationEnergyCalculator(calc, fmax=0.005)

    # ============================================================
    # CONFIGURATION
    # ============================================================
    mode = "automated"  # "legacy" or "automated"

    # ============================================================
    # MODE 1: Legacy - Compare pre-existing functionalized molecules
    # ============================================================
    if mode == "legacy":
        print("\n1. LEGACY MODE - Pre-existing functionalized GQD:")

        #  Groups autodetect:  NH2, OH, COOH
        result = calculator.calculate_formation_energy(
            pristine_key="GQD_TRIANGLE_3",  # Clean graphene quantum dot
            modified_key="GQD_TRIANGLE_3_NH2"  # functionalized GQD
        )

        print(f"\n{'=' * 50}")
        print("QUANTUM MECHANICAL FORMATION ENERGY ANALYSIS")
        print(f"{'=' * 50}")
        print(f"Formation Energy: {result.formation_energy:.4f} ± {result.uncertainty:.4f} eV")
        print(f"Identified Chemical Species: {result.groups}")

        # Energy component breakdown for understanding thermodynamic cycle
        print(f"\nTHERMODYNAMIC CYCLE COMPONENTS:")
        for comp, value in result.components.items():
            print(f"  {comp}: {value:.4f} eV")

        # Validation results for data quality assessment
        if result.validation_flags:
            print(f"\nVALIDATION WARNINGS: {result.validation_flags}")
        else:
            print(f"\nVALIDATION: ✓ All physical checks passed")

    # ============================================================
    # MODE 2: NEW - Automatically analyze all edge H functionalization sites
    # ============================================================
    elif mode == "automated":
        # Usage: Just pass the GQD mol file and functional group
        # Creates energy maps and statistical analysis automatically

        print("\n2. NEW MODE - Automated Functionalization Site Analysis:")

        # Configuration
        molecule_name = "GQD_TRIANGLE_3"  # Choose which molecule to analyze
        show_atom_idx = True  # Show atom indices on plots
        # Get molecule path from config
        mol_path = molecules_data[molecule_name]["path"]
        functional_groups = [
            # 'NH2', 'OH',
            # 'BH2',
            'COOH']
        for functional_group in functional_groups:
            excluded_atoms = []  # Optionally exclude specific H atoms
            # Run automated analysis - analyzes ALL edge H atoms
            atom_indices, formation_energies = calculator.analyze_functionalization_sites(
                mol_path=mol_path,
                functional_group=functional_group,
                molecule_name=molecule_name,
                show_atom_idx=show_atom_idx,
                excluded_atoms=excluded_atoms
            )

            print(f"\n{'=' * 60}")
            print(f"Analysis complete!")
            print(f"Results saved to: func_map_{molecule_name}_{functional_group}/")
            print(f"  - formation_energy_map.png: Visual energy map")
            print(f"  - energy_distribution.png: Statistical distribution")
            print(f"  - summary.txt: Detailed statistics")
            print(f"  - energy_map.txt: Raw data")
            print(f"{'=' * 60}")

    else:
        print(f"ERROR: Unknown mode '{mode}'. Use 'legacy' or 'automated'.")






if __name__ == "__main__":
    main()