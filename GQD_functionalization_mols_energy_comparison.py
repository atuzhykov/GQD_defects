import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from ase.build import molecule
from ase.io import read
from ase.optimize import FIRE

from configs import molecules_data

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
        }

        # Functional group recognition patterns for automated chemical analysis
        # Format: [atomic_composition, expected_bond_lengths]
        # Bond lengths from experimental crystallography and quantum chemistry
        self.fg_patterns = {
            'NH2': (['N', 'H', 'H'], [('N', 'H', 1.02), ('H', 'N', 1.02)]),  # Amine group
            'OH': (['O', 'H'], [('O', 'H', 0.97)]),  # Hydroxyl group
            'COOH': (['C', 'O', 'O', 'H'], [('C', 'O', 1.21), ('C', 'O', 1.36), ('O', 'H', 0.97)]),  # Carboxyl
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

        # Geometry optimization using FIRE algorithm
        # Minimizes total energy: E(R₁, R₂, ..., R_N) → minimum
        # Satisfies force balance: F_i = -∇_i E = 0 for all atoms i
        dyn = FIRE(atoms, logfile=None)
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
        reference_molecules = {
            'NH2': 'NH3',  # Ammonia as NH₂ precursor
            'OH': 'H2O',  # Water as OH precursor
            'COOH': 'HCOOH'  # Formic acid as COOH reference
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
                    dyn = FIRE(mol, logfile=None)
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
                # Use direct molecular reference (no correction needed)
                mu = base_energy
                sigma = 0.1  # Typical molecular calculation uncertainty

            return mu, sigma

        # Fallback for unknown functional groups
        logger.warning(f"Unknown functional group: {group_name}")
        return 0.0, 0.5  # Large uncertainty indicates unknown chemistry

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
                if group in ['NH2', 'OH', 'COOH']:
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

    Example calculation workflow:
    1. Initialize DFT calculator with appropriate parameters
    2. Create formation energy calculator with tight convergence
    3. Calculate formation energy for specific functionalization
    4. Analyze results and validate against chemical intuition
    """
    from sevenn.calculator import SevenNetCalculator


    calc = SevenNetCalculator('7net-l3i5', modal='mpa')

    calculator = FormationEnergyCalculator(calc, fmax=0.005)

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






if __name__ == "__main__":
    main()