import os
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from ase.io import read
from ase.optimize import BFGS
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
class DopingResult:
    """
    Doping calculation result with proper cell size normalization.

    Physical interpretation:
    - formation_energy: Total ΔE_f for the cell [eV]
    - formation_energy_per_dopant: ΔE_f normalized per dopant atom [eV/dopant]
    - uncertainty: Statistical error from DFT and chemical potential uncertainties [eV]
    - components: Energy breakdown for analysis [eV]
    - dopant_info: Dopant type, count, and concentration
    - validation_flags: Physical reasonableness checks
    """
    formation_energy: float  # Total formation energy for cell [eV]
    formation_energy_per_dopant: float  # Per-dopant energy (cell-size independent) [eV]
    uncertainty: float  # Total uncertainty [eV]
    components: Dict[str, float]  # Energy components [eV]
    dopant_info: Dict[str, any]  # Dopant details
    validation_flags: List[str]  # Validation warnings


class DopingFormationCalculator:
    """
    Doping formation energy calculator with proper cell size normalization.

    Theoretical framework:
    For substitutional doping in cell with N atoms and n dopants:
    ΔE_f = E(doped) - E(pristine) + n_removed × μ(host) - n_added × μ(dopant)

    Cell-size independent formation energy:
    ΔE_f_per_dopant = ΔE_f / n_dopants

    This allows comparison between different cell sizes and concentrations.
    """

    def __init__(self, calculator, fmax=0.05):
        """
        Initialize calculator with quantum mechanical parameters.

        Args:
            calculator: DFT calculator (e.g., SevenNet)
            fmax: Force convergence criterion [eV/Å]
        """
        self.calc = calculator
        self.fmax = fmax
        self.energy_cache = {}  # Cache for calculated energies

        # Chemical potentials with uncertainties [eV]
        # Based on reference states (bulk or molecular)
        self.chemical_potentials = {
            # Host atoms
            'C': (-9.085, 0.035),  # Graphene/graphite reference
            'H': (-3.396, 0.015),  # H2/2 reference

            # Common dopants
            'N': (-8.328, 0.025),  # N2/2 reference
            'B': (-6.678, 0.030),  # Bulk boron reference
            'P': (-5.641, 0.030),  # White phosphorus reference
            'S': (-4.136, 0.030),  # Orthorhombic sulfur
            'O': (-4.952, 0.020),  # O2/2 reference
            'F': (-1.912, 0.025),  # F2/2 reference
            'Li': (-1.82, 0.05),
        }

    def calculate_doping_energy(self, pristine_key: str, doped_key: str) -> DopingResult:
        """
        Calculate doping formation energy with cell size normalization.

        Process:
        1. Calculate total energies for pristine and doped systems
        2. Identify dopant type and count
        3. Apply chemical potential corrections
        4. Normalize per dopant for cell-size independence
        5. Propagate uncertainties
        6. Validate results

        Args:
            pristine_key: Key for pristine structure in molecules_data
            doped_key: Key for doped structure in molecules_data

        Returns:
            DopingResult with normalized formation energy
        """

        # Step 1: Calculate energies
        E_pristine, forces_pristine = self._get_energy(pristine_key)
        E_doped, forces_doped = self._get_energy(doped_key)

        # Step 2: Analyze doping configuration
        dopant_info = self._analyze_doping(pristine_key, doped_key)

        # Step 3: Calculate chemical potential corrections
        mu_correction, mu_uncertainty = self._calculate_chemical_potentials(dopant_info)

        # Step 4: Formation energy calculation
        # Total formation energy for the cell
        formation_energy_total = E_doped - E_pristine + mu_correction

        # Normalized per dopant (cell-size independent)
        n_dopants = dopant_info['n_dopants']
        formation_energy_per_dopant = formation_energy_total / n_dopants if n_dopants > 0 else formation_energy_total

        # Step 5: Uncertainty propagation
        # σ²(total) = σ²(DFT) + σ²(μ)
        dft_uncertainty = 0.001  # 1 meV DFT convergence error
        uncertainty = np.sqrt(
            (mu_uncertainty) ** 2 +
            (dft_uncertainty * 2) ** 2  # Two DFT calculations
        )
        uncertainty_per_dopant = uncertainty / n_dopants if n_dopants > 0 else uncertainty

        # Step 6: Validation
        validation_flags = self._validate_result(
            formation_energy_per_dopant, dopant_info, forces_pristine, forces_doped
        )

        # Energy components for analysis
        components = {
            'E_doped': E_doped,
            'E_pristine': E_pristine,
            'delta_E': E_doped - E_pristine,
            'mu_correction': mu_correction,
            'mu_host': dopant_info.get('mu_host_total', 0),
            'mu_dopant': dopant_info.get('mu_dopant_total', 0),
        }

        # Print detailed breakdown
        self._print_analysis(
            dopant_info, components, formation_energy_total,
            formation_energy_per_dopant, uncertainty_per_dopant
        )

        return DopingResult(
            formation_energy=formation_energy_total,
            formation_energy_per_dopant=formation_energy_per_dopant,
            uncertainty=uncertainty_per_dopant,
            components=components,
            dopant_info=dopant_info,
            validation_flags=validation_flags
        )

    def _get_energy(self, mol_key: str) -> Tuple[float, np.ndarray]:
        """
        Calculate optimized energy for a structure.

        Args:
            mol_key: Key in molecules_data

        Returns:
            energy: Total energy [eV]
            forces: Atomic forces [eV/Å]
        """
        # Use cache if available
        if mol_key in self.energy_cache:
            return self.energy_cache[mol_key]

        # Load structure
        atoms = read(molecules_data[mol_key]["mol_path"])
        atoms.calc = self.calc

        # Geometry optimization
        dyn = BFGS(atoms, logfile=None)
        dyn.run(fmax=self.fmax)

        # Get properties
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        # Check convergence
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > self.fmax:
            logger.warning(f"Poor convergence for {mol_key}: max_force = {max_force:.4f}")

        logger.info(f"Energy for {mol_key}: {energy:.4f} eV (max_force: {max_force:.4f})")

        # Cache result
        self.energy_cache[mol_key] = (energy, forces)

        return energy, forces

    def _analyze_doping(self, pristine_key: str, doped_key: str) -> Dict:
        """
        Analyze doping configuration by comparing structures.

        Args:
            pristine_key: Pristine structure key
            doped_key: Doped structure key

        Returns:
            Dictionary with dopant information
        """
        # Load structures
        pristine = read(molecules_data[pristine_key]["mol_path"])
        doped = read(molecules_data[doped_key]["mol_path"])

        # Get compositions
        comp_pristine = self._get_composition(pristine)
        comp_doped = self._get_composition(doped)

        # Find changes
        dopant_info = {
            'dopant_element': None,
            'n_dopants': 0,
            'host_removed': None,
            'n_host_removed': 0,
            'type': None,  # substitutional, interstitial, or vacancy
            'concentration': 0.0,
            'cell_size': len(pristine),
        }

        # Identify what changed
        all_elements = set(comp_pristine.keys()) | set(comp_doped.keys())

        for elem in all_elements:
            n_pristine = comp_pristine.get(elem, 0)
            n_doped = comp_doped.get(elem, 0)
            delta = n_doped - n_pristine

            if delta > 0 and elem not in ['C', 'H']:  # Dopant added
                dopant_info['dopant_element'] = elem
                dopant_info['n_dopants'] = delta
            elif delta < 0 and elem == 'C':  # Carbon removed
                dopant_info['host_removed'] = elem
                dopant_info['n_host_removed'] = abs(delta)

        # Classify doping type
        if dopant_info['n_dopants'] > 0 and dopant_info['n_host_removed'] > 0:
            dopant_info['type'] = 'substitutional'
        elif dopant_info['n_dopants'] > 0 and dopant_info['n_host_removed'] == 0:
            dopant_info['type'] = 'interstitial'
        elif dopant_info['n_dopants'] == 0 and dopant_info['n_host_removed'] > 0:
            dopant_info['type'] = 'vacancy'

        # Calculate concentration (% of non-H atoms)
        n_heavy_atoms = sum(comp_doped.get(elem, 0) for elem in comp_doped if elem != 'H')
        if n_heavy_atoms > 0:
            dopant_info['concentration'] = (dopant_info['n_dopants'] / n_heavy_atoms) * 100

        logger.info(f"Doping type: {dopant_info['type']}")
        logger.info(f"Dopant: {dopant_info['dopant_element']} ({dopant_info['n_dopants']} atoms)")
        logger.info(f"Concentration: {dopant_info['concentration']:.2f}%")

        return dopant_info

    def _calculate_chemical_potentials(self, dopant_info: Dict) -> Tuple[float, float]:
        """
        Calculate chemical potential corrections.

        For substitutional: μ_correction = n_removed × μ(host) - n_added × μ(dopant)
        For interstitial: μ_correction = -n_added × μ(dopant)
        For vacancy: μ_correction = n_removed × μ(host)

        Args:
            dopant_info: Doping configuration

        Returns:
            mu_total: Total chemical potential correction [eV]
            uncertainty: Combined uncertainty [eV]
        """
        mu_total = 0.0
        uncertainty_sq = 0.0

        # Host atom contribution (for substitutional and vacancy)
        if dopant_info['n_host_removed'] > 0:
            host_elem = dopant_info['host_removed'] or 'C'
            if host_elem in self.chemical_potentials:
                mu_host, sigma_host = self.chemical_potentials[host_elem]
                mu_host_total = dopant_info['n_host_removed'] * mu_host
                mu_total += mu_host_total
                uncertainty_sq += (dopant_info['n_host_removed'] * sigma_host) ** 2
                dopant_info['mu_host_total'] = mu_host_total

        # Dopant contribution (for substitutional and interstitial)
        if dopant_info['n_dopants'] > 0 and dopant_info['dopant_element']:
            dopant_elem = dopant_info['dopant_element']
            if dopant_elem in self.chemical_potentials:
                mu_dopant, sigma_dopant = self.chemical_potentials[dopant_elem]
                mu_dopant_total = dopant_info['n_dopants'] * mu_dopant
                mu_total -= mu_dopant_total  # Negative because dopants are added
                uncertainty_sq += (dopant_info['n_dopants'] * sigma_dopant) ** 2
                dopant_info['mu_dopant_total'] = mu_dopant_total

        return mu_total, np.sqrt(uncertainty_sq)

    def _get_composition(self, atoms) -> Dict[str, int]:
        """Get atomic composition of structure."""
        composition = {}
        for symbol in atoms.get_chemical_symbols():
            composition[symbol] = composition.get(symbol, 0) + 1
        return composition

    def _validate_result(self, formation_energy_per_dopant: float, dopant_info: Dict,
                         forces_pristine: np.ndarray, forces_doped: np.ndarray) -> List[str]:
        """
        Validate results against known physics.

        Args:
            formation_energy_per_dopant: Per-dopant formation energy [eV]
            dopant_info: Doping configuration
            forces_pristine: Forces on pristine system [eV/Å]
            forces_doped: Forces on doped system [eV/Å]

        Returns:
            List of validation flags
        """
        flags = []

        # Convergence check
        max_force_pristine = np.max(np.linalg.norm(forces_pristine, axis=1))
        max_force_doped = np.max(np.linalg.norm(forces_doped, axis=1))

        if max_force_pristine > self.fmax or max_force_doped > self.fmax:
            flags.append("POOR_CONVERGENCE")

        # Energy range checks
        if dopant_info['type'] == 'substitutional':
            if formation_energy_per_dopant < -3.0:
                flags.append("VERY_FAVORABLE")
            elif formation_energy_per_dopant > 8.0:
                flags.append("VERY_UNFAVORABLE")

        # Concentration warning
        if dopant_info['concentration'] > 10:
            flags.append("HIGH_CONCENTRATION")
            logger.warning(f"High dopant concentration: {dopant_info['concentration']:.1f}%")

        # Thermodynamic classification
        if formation_energy_per_dopant < 0:
            flags.append("SPONTANEOUS")
        elif formation_energy_per_dopant < 2.0:
            flags.append("THERMALLY_ACCESSIBLE")
        else:
            flags.append("HIGH_BARRIER")

        return flags

    def _print_analysis(self, dopant_info: Dict, components: Dict,
                        formation_energy_total: float, formation_energy_per_dopant: float,
                        uncertainty: float) -> None:
        """Print detailed analysis of results."""
        print(f"\n{'=' * 60}")
        print("DOPING FORMATION ENERGY ANALYSIS")
        print(f"{'=' * 60}")

        print(f"\nSystem Information:")
        print(f"  Cell size: {dopant_info['cell_size']} atoms")
        print(f"  Doping type: {dopant_info['type']}")
        print(f"  Dopant: {dopant_info['dopant_element']} ({dopant_info['n_dopants']} atoms)")
        print(f"  Host removed: {dopant_info['host_removed']} ({dopant_info['n_host_removed']} atoms)")
        print(f"  Concentration: {dopant_info['concentration']:.2f}%")

        print(f"\nEnergy Components:")
        print(f"  E(pristine): {components['E_pristine']:.4f} eV")
        print(f"  E(doped):    {components['E_doped']:.4f} eV")
        print(f"  ΔE:          {components['delta_E']:.4f} eV")

        print(f"\nChemical Potentials:")
        if 'mu_host_total' in dopant_info:
            print(f"  μ(host) × n:   +{dopant_info['mu_host_total']:.4f} eV")
        if 'mu_dopant_total' in dopant_info:
            print(f"  μ(dopant) × n: -{dopant_info['mu_dopant_total']:.4f} eV")
        print(f"  Total correction: {components['mu_correction']:.4f} eV")

        print(f"\n{'=' * 60}")
        print(f"RESULTS:")
        print(f"  ΔE_f (total cell):     {formation_energy_total:.4f} eV")
        print(f"  ΔE_f (per dopant):     {formation_energy_per_dopant:.4f} ± {uncertainty:.4f} eV")
        print(f"{'=' * 60}")

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
        results_dir = f"doping_map_{molecule_name}_{dopant_element}{suffix}"
        os.makedirs(results_dir, exist_ok=True)

        # Load base structure
        base_atoms = read(mol_path)

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

        # Calculate pristine energy
        base_atoms.calc = self.calc
        print("Relaxing base structure...")
        dyn = BFGS(base_atoms, logfile=None)
        dyn.run(fmax=self.fmax)
        E_pristine = base_atoms.get_potential_energy()

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
                mu_host, sigma_host = self.chemical_potentials[host_element]
                mu_dopant, sigma_dopant = self.chemical_potentials[dopant_element]

                # For substitutional doping: ΔE_f = E_doped - E_pristine + μ_host - μ_dopant
                formation_energy = E_doped - E_pristine + mu_host - mu_dopant

                formation_energies.append(formation_energy)
                print(f"  Atom {atom_idx} ({host_element}) formation energy: {formation_energy:.3f} eV")

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
                      edgecolors='black', linewidths=1)

            if show_atom_idx:
                ax.text(atom.position[0], atom.position[1], str(i), fontsize=8)

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
    Demonstration of the improved doping analyzer.

    Now supports two modes:
    1. Legacy mode: Compare pre-existing doped molecules
    2. NEW mode: Automatically analyze all doping sites for a given GQD and dopant element
    """
    import platform
    import sys

    # Initialize calculator based on operating system
    system = platform.system()

    if system == "Linux":
        print(f"Running on {system} - using GPAW calculator")
        from gpaw import GPAW, PW, FermiDirac

        calc = GPAW(
            xc='PBE',
            mode=PW(300),  # Reduced from 400 - still reasonable for C-C bond breaking
            kpts=(1, 1, 1),
            symmetry='off',
            spinpol=True,  # Keep this - essential for radicals
            occupations=FermiDirac(0.05),  # Increased smearing - faster SCF convergence
            convergence={
                'energy': 0.001,  # Relaxed from 0.0005 (1 meV is fine for formation energies)
                'density': 1e-4,  # Much looser - 1e-6 is overkill
                'eigenstates': 1e-6,  # Looser from 1e-8
            },
            mixer={'backend': 'pulay', 'beta': 0.1, 'nmaxold': 5, 'weight': 50},  # Faster mixing
            maxiter=300,  # Explicit limit to catch non-convergence earlier
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

    # Create calculator
    calculator = DopingFormationCalculator(calc, fmax=0.05)

    # ============================================================
    # CONFIGURATION
    # ============================================================
    mode = "automated"  # "legacy" or "automated"

    # ============================================================
    # MODE 1: Legacy - Compare pre-existing doped molecules
    # ============================================================
    if mode == "legacy":
        # Common dopants: 'N' 'B' 'P' 'S' 'O' 'F'
        print("\n1. LEGACY MODE - DOPED GQD:")

        pristine_key = "GQD_TRIANGLE_3"
        doped_key = "GQD_TRIANGLE_3_min_C_added_N"

        result = calculator.calculate_doping_energy(
            pristine_key=pristine_key,
            doped_key=doped_key
        )

        # Comparison
        print(f"\n{'=' * 60}")
        print("COMPARISON (per-dopant energies):")
        print(f"  N-doping: {result.formation_energy_per_dopant:.3f} ± {result.uncertainty:.3f} eV")
        print(f"\n⚠️ Note: concentration is {result.dopant_info['concentration']:.1f}%")
        print(f"{'=' * 60}")

    # ============================================================
    # MODE 2: NEW - Automatically analyze all doping sites
    # ============================================================
    elif mode == "automated":
        # Usage: Just pass the GQD mol file and dopant element
        # Creates energy maps and statistical analysis automatically

        print("\n2. NEW MODE - Automated Doping Site Analysis:")

        # Configuration
        molecule_name = "GQD_TRIANGLE_3"  # Choose which molecule to analyze
        dopant_element = "N"   # Choose dopant: 'Li', 'N', 'B', 'P', 'S', 'O', 'F'
        show_atom_idx = True    # Show atom indices on plots
        excluded_atoms = []     # Optionally exclude specific atoms
        replace_H = False       # Set True to also analyze H replacement (wider energy map)

        # Get molecule path from config
        mol_path = molecules_data[molecule_name]["path"]

        # Run automated analysis
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
        print(f"Results saved to: doping_map_{molecule_name}_{dopant_element}/")
        print(f"  - formation_energy_map.png: Visual energy map")
        print(f"  - energy_distribution.png: Statistical distribution")
        print(f"  - summary.txt: Detailed statistics")
        print(f"  - energy_map.txt: Raw data")
        print(f"{'=' * 60}")

    else:
        print(f"ERROR: Unknown mode '{mode}'. Use 'legacy' or 'automated'.")


if __name__ == "__main__":
    main()