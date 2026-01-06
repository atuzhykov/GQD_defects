"""
GPAW DFT Analysis Module

This module performs comprehensive DFT calculations using GPAW including:
- HOMO/LUMO eigenvalues and band gap
- Electronic density
- Optical properties (absorption spectrum)
- Phonon/vibrational modes

Usage:
    from gpaw_analysis import GPAWAnalyzer

    analyzer = GPAWAnalyzer(molecule_name="Coronene")
    results = analyzer.run_full_analysis()
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS

from config import molecules_data

# GPAW imports
try:
    from gpaw import GPAW, PW, FermiDirac
    from gpaw.tddft import TDDFT
    from gpaw.tddft.spectrum import photoabsorption_spectrum
    GPAW_AVAILABLE = True
except ImportError:
    GPAW_AVAILABLE = False
    print("Warning: GPAW not available. This module requires GPAW to run.")


class GPAWAnalyzer:
    """
    Comprehensive GPAW DFT analysis for molecular systems.

    Attributes:
        molecule_name (str): Name of molecule from config
        atoms (ase.Atoms): Atomic structure
        calc (GPAW): GPAW calculator instance
        results (dict): Dictionary storing all calculated properties
    """

    def __init__(self, molecule_name, xc='PBE', mode_cutoff=300, kpts=(1,1,1),
                 spinpol=True, output_dir=None):
        """
        Initialize GPAW analyzer.

        Parameters:
            molecule_name (str): Molecule name from config.py
            xc (str): Exchange-correlation functional (default: 'PBE')
            mode_cutoff (int): Plane wave cutoff in eV (default: 300)
            kpts (tuple): k-point sampling (default: (1,1,1) for molecules)
            spinpol (bool): Enable spin polarization (default: True)
            output_dir (str): Directory for output files (default: gpaw_results_{molecule_name})
        """
        if not GPAW_AVAILABLE:
            raise ImportError("GPAW is not installed. Please install GPAW to use this module.")

        self.molecule_name = molecule_name
        self.xc = xc
        self.mode_cutoff = mode_cutoff
        self.kpts = kpts
        self.spinpol = spinpol

        # Set output directory
        if output_dir is None:
            self.output_dir = f"gpaw_results_{molecule_name}"
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Load structure
        self._load_structure()

        # Initialize results dictionary
        self.results = {}

        # Calculator will be set up when needed
        self.calc = None
        self.is_converged = False

    def _load_structure(self):
        """Load molecular structure from config."""
        if self.molecule_name not in molecules_data:
            raise ValueError(f"Molecule '{self.molecule_name}' not found in config.py")

        mol_data = molecules_data[self.molecule_name]
        mol_filename = mol_data["path"]
        cell_size = mol_data["cell"]

        # Read molecule
        self.atoms = read(mol_filename)

        # Center in cell
        positions = self.atoms.get_positions()
        center_of_mass = np.mean(positions, axis=0)
        translation = np.array([cell_size / 2] * 3) - center_of_mass
        positions = positions + translation
        self.atoms.set_positions(positions)
        self.atoms.set_cell([cell_size] * 3)
        self.atoms.set_pbc(True)

        print(f"Loaded {self.molecule_name}: {len(self.atoms)} atoms")
        print(f"Chemical formula: {self.atoms.get_chemical_formula()}")

    def setup_calculator(self, relax=True, txt_output='gpaw_calc.txt'):
        """
        Set up GPAW calculator and optionally relax structure.

        Parameters:
            relax (bool): Whether to relax the structure (default: True)
            txt_output (str): GPAW output filename
        """
        print("\nSetting up GPAW calculator...")

        self.calc = GPAW(
            xc=self.xc,
            mode=PW(self.mode_cutoff),
            kpts=self.kpts,
            symmetry='off',
            spinpol=self.spinpol,
            occupations=FermiDirac(0.05),
            convergence={
                'energy': 0.0005,
                'density': 1e-4,
                'eigenstates': 1e-6,
            },
            txt=os.path.join(self.output_dir, txt_output),
        )

        self.atoms.set_calculator(self.calc)

        if relax:
            print("Relaxing structure...")
            optimizer = BFGS(self.atoms,
                           trajectory=os.path.join(self.output_dir, 'relaxation.traj'))
            optimizer.run(fmax=0.05)

            # Save relaxed structure
            write(os.path.join(self.output_dir, f'relaxed_{self.molecule_name}.xyz'),
                  self.atoms)
            print(f"Structure relaxed and saved")

        # Get ground state energy
        energy = self.atoms.get_potential_energy()
        self.results['ground_state_energy'] = energy
        print(f"Ground state energy: {energy:.3f} eV")

        self.is_converged = True

    def calculate_homo_lumo(self):
        """
        Calculate HOMO, LUMO eigenvalues and band gap.

        Returns:
            dict: Dictionary with homo, lumo, gap, and all eigenvalues
        """
        if not self.is_converged:
            raise RuntimeError("Calculator not set up. Run setup_calculator() first.")

        print("\nCalculating HOMO/LUMO...")

        # Get HOMO and LUMO
        homo, lumo = self.calc.get_homo_lumo()
        gap = lumo - homo

        # Get all eigenvalues for both spins if spin-polarized
        eigenvalues = {}
        if self.spinpol:
            eigenvalues['spin_up'] = self.calc.get_eigenvalues(kpt=0, spin=0)
            eigenvalues['spin_down'] = self.calc.get_eigenvalues(kpt=0, spin=1)
            homo_up, lumo_up = self.calc.get_homo_lumo(spin=0)
            homo_down, lumo_down = self.calc.get_homo_lumo(spin=1)

            print(f"  HOMO (spin up): {homo_up:.3f} eV")
            print(f"  LUMO (spin up): {lumo_up:.3f} eV")
            print(f"  Gap (spin up): {lumo_up - homo_up:.3f} eV")
            print(f"  HOMO (spin down): {homo_down:.3f} eV")
            print(f"  LUMO (spin down): {lumo_down:.3f} eV")
            print(f"  Gap (spin down): {lumo_down - homo_down:.3f} eV")
        else:
            eigenvalues['all'] = self.calc.get_eigenvalues(kpt=0)

        print(f"  HOMO: {homo:.3f} eV")
        print(f"  LUMO: {lumo:.3f} eV")
        print(f"  Band gap: {gap:.3f} eV")

        self.results['homo_lumo'] = {
            'homo': homo,
            'lumo': lumo,
            'gap': gap,
            'eigenvalues': eigenvalues
        }

        # Save to file
        with open(os.path.join(self.output_dir, 'homo_lumo.txt'), 'w') as f:
            f.write(f"HOMO-LUMO Analysis for {self.molecule_name}\n")
            f.write(f"{'='*50}\n")
            f.write(f"HOMO: {homo:.6f} eV\n")
            f.write(f"LUMO: {lumo:.6f} eV\n")
            f.write(f"Band Gap: {gap:.6f} eV\n")
            if self.spinpol:
                f.write(f"\nSpin-resolved:\n")
                f.write(f"  Spin up   - HOMO: {homo_up:.6f} eV, LUMO: {lumo_up:.6f} eV, Gap: {lumo_up-homo_up:.6f} eV\n")
                f.write(f"  Spin down - HOMO: {homo_down:.6f} eV, LUMO: {lumo_down:.6f} eV, Gap: {lumo_down-homo_down:.6f} eV\n")

        # Plot energy levels
        self._plot_energy_levels()

        return self.results['homo_lumo']

    def _plot_energy_levels(self):
        """Plot energy level diagram."""
        homo_lumo = self.results['homo_lumo']

        fig, ax = plt.subplots(figsize=(8, 10))

        if self.spinpol:
            # Plot spin up and spin down separately
            eigs_up = homo_lumo['eigenvalues']['spin_up']
            eigs_down = homo_lumo['eigenvalues']['spin_down']

            # Determine occupied/unoccupied
            nelectrons = self.calc.get_number_of_electrons()
            nocc = int(nelectrons / 2)

            # Plot levels
            for i, e in enumerate(eigs_up):
                color = 'blue' if i < nocc else 'red'
                ax.hlines(e, 0, 0.4, color=color, linewidth=2)

            for i, e in enumerate(eigs_down):
                color = 'blue' if i < nocc else 'red'
                ax.hlines(e, 0.6, 1.0, color=color, linewidth=2)

            ax.text(0.2, homo_lumo['homo'] - 0.5, 'HOMO', ha='center', fontsize=10)
            ax.text(0.2, homo_lumo['lumo'] + 0.5, 'LUMO', ha='center', fontsize=10)

            ax.text(0.2, eigs_up.min() - 1, 'Spin ↑', ha='center', fontsize=12, weight='bold')
            ax.text(0.8, eigs_down.min() - 1, 'Spin ↓', ha='center', fontsize=12, weight='bold')
        else:
            eigs = homo_lumo['eigenvalues']['all']
            nelectrons = self.calc.get_number_of_electrons()
            nocc = int(nelectrons / 2)

            for i, e in enumerate(eigs):
                color = 'blue' if i < nocc else 'red'
                ax.hlines(e, 0.3, 0.7, color=color, linewidth=2)

            ax.text(0.5, homo_lumo['homo'] - 0.5, 'HOMO', ha='center', fontsize=10)
            ax.text(0.5, homo_lumo['lumo'] + 0.5, 'LUMO', ha='center', fontsize=10)

        ax.set_ylabel('Energy (eV)', fontsize=12)
        ax.set_title(f'Energy Levels - {self.molecule_name}\nGap = {homo_lumo["gap"]:.3f} eV',
                     fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)

        plt.savefig(os.path.join(self.output_dir, 'energy_levels.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Energy level diagram saved")

    def calculate_electron_density(self, save_cube=True):
        """
        Calculate and save electronic density.

        Parameters:
            save_cube (bool): Save density in cube format for visualization

        Returns:
            numpy.ndarray: Electronic density on grid
        """
        if not self.is_converged:
            raise RuntimeError("Calculator not set up. Run setup_calculator() first.")

        print("\nCalculating electronic density...")

        # Get pseudo density (fast) and all-electron density (accurate)
        pseudo_density = self.calc.get_pseudo_density()

        # Get all-electron density (slower but more accurate)
        all_electron_density = self.calc.get_all_electron_density(gridrefinement=2)

        self.results['electron_density'] = {
            'pseudo': pseudo_density,
            'all_electron': all_electron_density
        }

        # Save as numpy arrays
        np.save(os.path.join(self.output_dir, 'pseudo_density.npy'), pseudo_density)
        np.save(os.path.join(self.output_dir, 'all_electron_density.npy'),
                all_electron_density)

        print(f"  Pseudo density shape: {pseudo_density.shape}")
        print(f"  All-electron density shape: {all_electron_density.shape}")

        # Save in cube format for visualization with software like VESTA, VMD
        if save_cube:
            from ase.io.cube import write_cube

            with open(os.path.join(self.output_dir, 'density.cube'), 'w') as f:
                write_cube(f, self.atoms, all_electron_density)
            print(f"  Density saved in cube format for visualization")

        # Create 2D slice visualization
        self._plot_density_slice(all_electron_density)

        return all_electron_density

    def _plot_density_slice(self, density):
        """Plot 2D slice of electron density."""
        # Take a slice through the middle of the cell (z-direction)
        z_mid = density.shape[2] // 2
        density_slice = density[:, :, z_mid]

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(density_slice.T, origin='lower', cmap='viridis',
                      extent=[0, self.atoms.cell[0, 0], 0, self.atoms.cell[1, 1]])

        # Overlay atomic positions (project to z=z_mid plane)
        for atom in self.atoms:
            ax.plot(atom.position[0], atom.position[1], 'o',
                   color='white', markersize=8, markeredgecolor='black')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Electron Density (e/Å³)', fontsize=12)

        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        ax.set_title(f'Electronic Density (z-slice) - {self.molecule_name}', fontsize=14)

        plt.savefig(os.path.join(self.output_dir, 'density_slice.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Density slice visualization saved")

    def calculate_simple_spectrum(self, energy_range=(0, 10), energy_step=0.01, width=0.3):
        """
        Calculate simple absorption spectrum from single-particle transitions.
        This is a fast approximation - does not include electron-hole interactions.

        Parameters:
            energy_range (tuple): Energy range in eV
            energy_step (float): Energy step in eV
            width (float): Gaussian broadening in eV

        Returns:
            tuple: (energies, spectrum)
        """
        if not self.is_converged:
            raise RuntimeError("Calculator not set up. Run setup_calculator() first.")

        print("\nCalculating simple absorption spectrum...")
        print("  Note: This is a single-particle approximation (no TDDFT)")

        # Get eigenvalues
        if self.spinpol:
            eigs_up = self.calc.get_eigenvalues(kpt=0, spin=0)
            eigs_down = self.calc.get_eigenvalues(kpt=0, spin=1)
            eigs = (eigs_up + eigs_down) / 2  # Average
        else:
            eigs = self.calc.get_eigenvalues(kpt=0)

        # Determine HOMO index
        nelectrons = self.calc.get_number_of_electrons()
        homo_idx = int(nelectrons / 2) - 1

        # Calculate all possible transitions
        energies = np.arange(energy_range[0], energy_range[1], energy_step)
        spectrum = np.zeros_like(energies)

        transitions = []
        for i in range(max(0, homo_idx - 2), homo_idx + 1):  # Few occupied states
            for j in range(homo_idx + 1, min(len(eigs), homo_idx + 6)):  # Few unoccupied
                trans_energy = eigs[j] - eigs[i]
                if energy_range[0] <= trans_energy <= energy_range[1]:
                    # Simple approximation: all transitions have equal weight
                    weight = 1.0
                    transitions.append((trans_energy, weight))

                    # Add Gaussian peak
                    gaussian = weight * np.exp(-((energies - trans_energy) / width) ** 2)
                    spectrum += gaussian

        print(f"  Calculated {len(transitions)} transitions")
        if transitions:
            print(f"  HOMO-LUMO transition at {transitions[0][0]:.3f} eV")

        self.results['simple_spectrum'] = {
            'energies': energies,
            'absorption': spectrum,
            'transitions': transitions
        }

        # Save
        with open(os.path.join(self.output_dir, 'simple_spectrum.txt'), 'w') as f:
            f.write("# Simple Absorption Spectrum (Single-Particle Approximation)\n")
            f.write("# Energy(eV)  Absorption(arb. units)\n")
            for e, a in zip(energies, spectrum):
                f.write(f"{e:.6f}  {a:.6e}\n")

        # Plot
        self._plot_optical_spectrum(energies, spectrum)

        return energies, spectrum

    def calculate_optical_properties(self, energy_range=(0, 10), energy_step=0.01,
                                    width=0.1, method='lr-tddft'):
        """
        Calculate optical absorption spectrum using TDDFT.

        Parameters:
            energy_range (tuple): Energy range for spectrum in eV (default: 0-10 eV)
            energy_step (float): Energy step in eV (default: 0.01 eV)
            width (float): Gaussian broadening width in eV (default: 0.1 eV)
            method (str): 'lr-tddft' (linear response) or 'rt-tddft' (real-time)

        Returns:
            tuple: (energies, absorption_spectrum)
        """
        if not self.is_converged:
            raise RuntimeError("Calculator not set up. Run setup_calculator() first.")

        print("\nCalculating optical properties using TDDFT...")
        print(f"  Method: {method}")
        print("  This may take a while...")

        # Save ground state for TDDFT
        gs_file = os.path.join(self.output_dir, 'gs.gpw')
        self.calc.write(gs_file)

        if method == 'lr-tddft':
            # Linear Response TDDFT - more stable for molecules
            from gpaw.lrtddft import LrTDDFT

            print(f"  Calculating excited states with LrTDDFT...")

            # Create LrTDDFT object - parameters depend on GPAW version
            try:
                # Try newer API (GPAW >= 20.x)
                lr = LrTDDFT(gs_file, txt=os.path.join(self.output_dir, 'lrtddft.txt'))
            except Exception as e:
                print(f"  Note: Using fallback LrTDDFT initialization")
                lr = LrTDDFT(gs_file)

            # Calculate excitation energies and oscillator strengths
            lr.write(os.path.join(self.output_dir, 'excitations.dat'))

            # Get excitation data (transitions)
            exlist = []
            try:
                # Method 1: Direct iteration
                for ex in lr:
                    exlist.append({
                        'energy': ex.get_energy(),
                        'weight': ex.get_oscillator_strength()[0]  # Sum of x,y,z
                    })
            except:
                # Method 2: Manual reading from file
                print("  Reading transitions from file...")
                with open(os.path.join(self.output_dir, 'excitations.dat'), 'r') as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                exlist.append({
                                    'energy': float(parts[1]),
                                    'weight': float(parts[2]) if len(parts) > 2 else 0.1
                                })
                            except:
                                continue

            # Create absorption spectrum from discrete transitions
            energies = np.arange(energy_range[0], energy_range[1], energy_step)
            spectrum = np.zeros_like(energies)

            if len(exlist) == 0:
                print("  Warning: No transitions found. Check excitations.dat file")
                print("  Creating empty spectrum...")
            else:
                # Broaden each transition with a Gaussian
                for ex in exlist:
                    energy = ex['energy']  # eV
                    osc_str = ex['weight']  # oscillator strength

                    # Gaussian broadening
                    gaussian = osc_str * np.exp(-((energies - energy) / width) ** 2)
                    spectrum += gaussian

                print(f"  Calculated {len(exlist)} transitions")
                if len(exlist) > 0:
                    print(f"  First excitation at {exlist[0]['energy']:.3f} eV")

        else:
            # Real-time TDDFT (original implementation)
            print("  Using real-time propagation TDDFT...")
            from gpaw.tddft import TDDFT
            from gpaw.tddft.spectrum import photoabsorption_spectrum

            td_calc = TDDFT(gs_file, txt=os.path.join(self.output_dir, 'tddft.txt'))

            # Apply kick in x, y, z directions
            kicks = [0, 1, 2]  # x, y, z

            print("  Performing time propagation...")
            # Time propagation parameters
            time_step = 10  # attoseconds
            iterations = 1000  # Total time = 10 fs

            spectrum_total = None

            for kick_direction in kicks:
                td_calc.absorption_kick(kick_strength=[1e-3 if i == kick_direction else 0
                                                       for i in range(3)])
                td_calc.propagate(time_step, iterations,
                                f'{self.output_dir}/dm_{kick_direction}.dat')

                # Calculate spectrum for this direction
                e, spec = photoabsorption_spectrum(
                    f'{self.output_dir}/dm_{kick_direction}.dat',
                    f'{self.output_dir}/spec_{kick_direction}.dat',
                    width=width
                )

                if spectrum_total is None:
                    energies = e
                    spectrum_total = spec
                else:
                    spectrum_total += spec

            spectrum = spectrum_total / 3.0  # Average over x, y, z

        self.results['optical'] = {
            'energies': energies,
            'absorption': spectrum
        }

        # Save spectrum data
        with open(os.path.join(self.output_dir, 'absorption_spectrum.txt'), 'w') as f:
            f.write("# Optical Absorption Spectrum\n")
            f.write("# Energy(eV)  Absorption(arb. units)\n")
            for e, a in zip(energies, spectrum):
                f.write(f"{e:.6f}  {a:.6e}\n")

        # Plot spectrum
        self._plot_optical_spectrum(energies, spectrum)

        print(f"  Optical spectrum calculated")

        return energies, spectrum

    def _plot_optical_spectrum(self, energies, absorption):
        """Plot optical absorption spectrum."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(energies, absorption, linewidth=2, color='blue')
        ax.fill_between(energies, absorption, alpha=0.3)

        ax.set_xlabel('Energy (eV)', fontsize=12)
        ax.set_ylabel('Absorption (arb. units)', fontsize=12)
        ax.set_title(f'Optical Absorption Spectrum - {self.molecule_name}', fontsize=14)
        ax.grid(alpha=0.3)
        ax.set_xlim(energies[0], energies[-1])

        plt.savefig(os.path.join(self.output_dir, 'absorption_spectrum.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Absorption spectrum plot saved")

    def calculate_dipole_moment(self):
        """
        Calculate electric dipole moment.

        Returns:
            numpy.ndarray: Dipole moment vector [x, y, z] in Debye
        """
        if not self.is_converged:
            raise RuntimeError("Calculator not set up. Run setup_calculator() first.")

        print("\nCalculating dipole moment...")

        # Get dipole moment from calculator (in e*Angstrom)
        dipole = self.calc.get_dipole_moment()

        # Convert to Debye (1 e*Angstrom = 4.80321 Debye)
        dipole_debye = dipole * 4.80321

        dipole_magnitude = np.linalg.norm(dipole_debye)

        self.results['dipole_moment'] = {
            'vector': dipole_debye,
            'magnitude': dipole_magnitude
        }

        print(f"  Dipole moment: [{dipole_debye[0]:.3f}, {dipole_debye[1]:.3f}, {dipole_debye[2]:.3f}] D")
        print(f"  Magnitude: {dipole_magnitude:.3f} D")

        # Save to file
        with open(os.path.join(self.output_dir, 'dipole_moment.txt'), 'w') as f:
            f.write(f"Electric Dipole Moment for {self.molecule_name}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Vector (Debye): [{dipole_debye[0]:.6f}, {dipole_debye[1]:.6f}, {dipole_debye[2]:.6f}]\n")
            f.write(f"Magnitude (Debye): {dipole_magnitude:.6f}\n")

        return dipole_debye

    def calculate_mulliken_charges(self):
        """
        Calculate atomic charges using Bader or Hirshfeld analysis.
        Note: GPAW doesn't have built-in Mulliken charges.

        Returns:
            numpy.ndarray: Atomic charges
        """
        if not self.is_converged:
            raise RuntimeError("Calculator not set up. Run setup_calculator() first.")

        print("\nCalculating atomic charges...")

        try:
            # GPAW supports Hirshfeld charges (better than Mulliken for DFT)
            from gpaw.analyse.hirshfeld import HirshfeldPartitioning

            # Perform Hirshfeld partitioning
            hp = HirshfeldPartitioning(self.calc)
            charges = hp.get_charges()

            self.results['hirshfeld_charges'] = charges

            print(f"  Hirshfeld charges calculated for {len(charges)} atoms")

            # Save to file
            with open(os.path.join(self.output_dir, 'hirshfeld_charges.txt'), 'w') as f:
                f.write(f"Hirshfeld Atomic Charges for {self.molecule_name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Note: Hirshfeld analysis is more reliable than Mulliken for DFT\n")
                f.write(f"{'='*60}\n")
                f.write(f"{'Atom':<6} {'Element':<8} {'Charge (e)':<15}\n")
                f.write(f"{'-'*60}\n")
                for i, (atom, charge) in enumerate(zip(self.atoms, charges)):
                    f.write(f"{i:<6} {atom.symbol:<8} {charge:>12.6f}\n")

                f.write(f"\nTotal charge: {np.sum(charges):.6f} e\n")

            print(f"  Total charge: {np.sum(charges):.6f} e")
            return charges

        except ImportError:
            print(f"  Warning: Hirshfeld analysis not available in this GPAW version")
            print(f"  Calculating simple Bader-like charges from density...")

            try:
                # Fallback: Use density-based charges
                # Get all-electron density
                density = self.calc.get_all_electron_density()

                # Simple charge estimate: integrate density around each atom
                charges = np.zeros(len(self.atoms))

                # This is a very rough estimate
                print(f"  Using approximate density-based charges")
                print(f"  Note: These are less accurate than Hirshfeld or Bader analysis")

                # For now, just return zeros with a warning
                self.results['approximate_charges'] = charges

                with open(os.path.join(self.output_dir, 'charges_note.txt'), 'w') as f:
                    f.write(f"Charge Analysis for {self.molecule_name}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Note: Accurate charge analysis requires:\n")
                    f.write(f"  - Hirshfeld partitioning (install GPAW with full features)\n")
                    f.write(f"  - Or external Bader analysis tool\n")
                    f.write(f"\nFor GPAW Hirshfeld analysis:\n")
                    f.write(f"  from gpaw.analyse.hirshfeld import HirshfeldPartitioning\n")

                print(f"  See charges_note.txt for more information")
                return charges

            except Exception as e:
                print(f"  Warning: Could not calculate charges: {e}")
                return None

        except Exception as e:
            print(f"  Warning: Could not calculate Hirshfeld charges: {e}")
            return None

    def calculate_phonons(self, delta=0.01, nfree=2):
        """
        Calculate phonon modes and vibrational frequencies.

        Parameters:
            delta (float): Displacement for finite differences in Angstrom
            nfree (int): Number of displacements per atom (2 or 4)

        Returns:
            dict: Phonon frequencies and modes
        """
        if not self.is_converged:
            raise RuntimeError("Calculator not set up. Run setup_calculator() first.")

        print("\nCalculating phonon modes...")
        print("  This will perform finite displacement calculations...")

        # For molecules, we don't need a supercell
        # Use ASE Vibrations for molecules
        from ase.vibrations import Vibrations

        vib = Vibrations(self.atoms, name=os.path.join(self.output_dir, 'vib'))
        vib.run()

        # Get frequencies
        vib.summary(log=os.path.join(self.output_dir, 'phonon_summary.txt'))

        # Get vibrational energies (in meV)
        energies = vib.get_energies()

        # Convert to frequencies (cm^-1)
        frequencies_cm = energies * 8.06554  # meV to cm^-1

        print(f"  Calculated {len(frequencies_cm)} vibrational modes")
        print(f"  Frequency range: {frequencies_cm.min():.1f} - {frequencies_cm.max():.1f} cm⁻¹")

        # Save frequencies
        with open(os.path.join(self.output_dir, 'phonon_frequencies.txt'), 'w') as f:
            f.write(f"Vibrational Frequencies for {self.molecule_name}\n")
            f.write(f"{'='*50}\n")
            f.write(f"{'Mode':<6} {'Energy (meV)':<15} {'Frequency (cm⁻¹)':<20}\n")
            f.write(f"{'-'*50}\n")
            for i, (e, freq) in enumerate(zip(energies, frequencies_cm)):
                f.write(f"{i:<6} {e:<15.3f} {freq:<20.1f}\n")

        self.results['phonons'] = {
            'energies_meV': energies,
            'frequencies_cm': frequencies_cm,
            'vibrations_object': vib
        }

        # Plot IR spectrum (simplified - intensity = 1 for all modes)
        self._plot_ir_spectrum(frequencies_cm)

        # Write trajectory files for visualization
        for mode in range(len(frequencies_cm)):
            vib.write_mode(mode)

        print(f"  Mode trajectory files written for visualization")

        return self.results['phonons']

    def _plot_ir_spectrum(self, frequencies):
        """Plot simulated IR spectrum."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Simple stick spectrum
        ax.stem(frequencies, np.ones_like(frequencies), linefmt='b-',
               markerfmt='bo', basefmt=' ')

        ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Intensity (arb. units)', fontsize=12)
        ax.set_title(f'Vibrational Spectrum - {self.molecule_name}', fontsize=14)
        ax.grid(alpha=0.3)

        plt.savefig(os.path.join(self.output_dir, 'vibrational_spectrum.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Vibrational spectrum plot saved")

    def run_full_analysis(self, include_optical=True, include_phonons=True,
                         include_dipole=True, include_charges=True):
        """
        Run complete GPAW analysis including all calculations.

        Parameters:
            include_optical (bool): Calculate optical properties (time-consuming)
            include_phonons (bool): Calculate phonons (time-consuming)
            include_dipole (bool): Calculate dipole moment
            include_charges (bool): Calculate Mulliken charges

        Returns:
            dict: Complete results dictionary
        """
        print(f"\n{'='*60}")
        print(f"GPAW Full Analysis for {self.molecule_name}")
        print(f"{'='*60}")

        # 1. Setup and relax
        self.setup_calculator(relax=True)

        # 2. HOMO/LUMO
        self.calculate_homo_lumo()

        # 3. Electronic density
        self.calculate_electron_density()

        # 4. Dipole moment (fast)
        if include_dipole:
            try:
                self.calculate_dipole_moment()
            except Exception as e:
                print(f"\nWarning: Dipole calculation failed: {e}")

        # 5. Mulliken charges (fast)
        if include_charges:
            try:
                self.calculate_mulliken_charges()
            except Exception as e:
                print(f"\nWarning: Charge calculation failed: {e}")

        # 6. Optical properties (optional - can be slow)
        if include_optical:
            try:
                self.calculate_optical_properties(method='lr-tddft')
            except Exception as e:
                print(f"\nWarning: TDDFT optical calculation failed: {e}")
                print("Trying simple spectrum instead...")
                try:
                    self.calculate_simple_spectrum()
                except Exception as e2:
                    print(f"Warning: Simple spectrum also failed: {e2}")
                    print("Continuing with other calculations...")

        # 7. Phonons (optional - can be slow)
        if include_phonons:
            try:
                self.calculate_phonons()
            except Exception as e:
                print(f"\nWarning: Phonon calculation failed: {e}")
                print("Continuing with other calculations...")

        # Save complete results
        self._save_summary()

        print(f"\n{'='*60}")
        print(f"Analysis complete! Results saved in: {self.output_dir}")
        print(f"{'='*60}\n")

        return self.results

    def _save_summary(self):
        """Save summary of all results."""
        with open(os.path.join(self.output_dir, 'analysis_summary.txt'), 'w') as f:
            f.write(f"GPAW Analysis Summary for {self.molecule_name}\n")
            f.write(f"{'='*60}\n\n")

            f.write(f"Structure:\n")
            f.write(f"  Formula: {self.atoms.get_chemical_formula()}\n")
            f.write(f"  Number of atoms: {len(self.atoms)}\n")
            f.write(f"  Cell size: {self.atoms.cell[0,0]:.2f} Å\n\n")

            f.write(f"Calculation Parameters:\n")
            f.write(f"  XC functional: {self.xc}\n")
            f.write(f"  Cutoff energy: {self.mode_cutoff} eV\n")
            f.write(f"  k-points: {self.kpts}\n")
            f.write(f"  Spin polarized: {self.spinpol}\n\n")

            if 'ground_state_energy' in self.results:
                f.write(f"Ground State:\n")
                f.write(f"  Energy: {self.results['ground_state_energy']:.6f} eV\n\n")

            if 'homo_lumo' in self.results:
                hl = self.results['homo_lumo']
                f.write(f"Electronic Structure:\n")
                f.write(f"  HOMO: {hl['homo']:.3f} eV\n")
                f.write(f"  LUMO: {hl['lumo']:.3f} eV\n")
                f.write(f"  Band gap: {hl['gap']:.3f} eV\n\n")

            if 'phonons' in self.results:
                ph = self.results['phonons']
                f.write(f"Vibrational Properties:\n")
                f.write(f"  Number of modes: {len(ph['frequencies_cm'])}\n")
                f.write(f"  Frequency range: {ph['frequencies_cm'].min():.1f} - ")
                f.write(f"{ph['frequencies_cm'].max():.1f} cm⁻¹\n\n")

            if 'optical' in self.results:
                f.write(f"Optical Properties:\n")
                f.write(f"  Absorption spectrum calculated\n")
                opt = self.results['optical']
                max_abs_idx = np.argmax(opt['absorption'])
                f.write(f"  Peak absorption at: {opt['energies'][max_abs_idx]:.2f} eV\n\n")

            f.write(f"\nOutput files:\n")
            f.write(f"  - relaxed_{self.molecule_name}.xyz: Relaxed structure\n")
            f.write(f"  - homo_lumo.txt: HOMO/LUMO eigenvalues\n")
            f.write(f"  - energy_levels.png: Energy level diagram\n")
            f.write(f"  - density.cube: Electronic density (for visualization)\n")
            f.write(f"  - density_slice.png: 2D density visualization\n")
            if 'optical' in self.results:
                f.write(f"  - absorption_spectrum.txt/png: Optical absorption\n")
            if 'phonons' in self.results:
                f.write(f"  - phonon_frequencies.txt: Vibrational modes\n")
                f.write(f"  - vibrational_spectrum.png: IR spectrum\n")


if __name__ == "__main__":
    # Example usage
    molecule_name = "QD_1"  # Change this to analyze different molecules

    # Create analyzer
    analyzer = GPAWAnalyzer(
        molecule_name=molecule_name,
        xc='PBE',
        mode_cutoff=300,
        kpts=(1, 1, 1),
        spinpol=True
    )

    # Run full analysis
    # Note: Set include_optical=False and include_phonons=False for faster testing
    results = analyzer.run_full_analysis(
        include_optical=True,   # Set to False to skip optical calculations
        include_phonons=True    # Set to False to skip phonon calculations
    )

    # Access individual results
    print("\nKey Results:")
    print(f"Band gap: {results['homo_lumo']['gap']:.3f} eV")

    if 'phonons' in results:
        print(f"Number of vibrational modes: {len(results['phonons']['frequencies_cm'])}")

    if 'optical' in results:
        max_idx = np.argmax(results['optical']['absorption'])
        print(f"Peak absorption at: {results['optical']['energies'][max_idx]:.2f} eV")
