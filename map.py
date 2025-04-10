import os
from itertools import combinations
import torch
torch.backends.cudnn.enabled = False
torch._dynamo.config.suppress_errors = True
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.optimize import FIRE
from ase.visualize import view
from matplotlib.colors import LinearSegmentedColormap
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from scipy import stats

from config import molecules_data
from utils import delete_atom_by_idx, rotate_bond_transform, calculate_formation_energy, MU_C

import matplotlib as mpl
# Set global font properties for all plots
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12


mpl.rcParams['axes.titlesize'] = 16  # Plot titles
mpl.rcParams['axes.labelsize'] = 14  # Axis labels
mpl.rcParams['xtick.labelsize'] = 12  # X-axis tick labels
mpl.rcParams['ytick.labelsize'] = 12  # Y-axis tick labels
mpl.rcParams['legend.fontsize'] = 12  # Legend text
mpl.rcParams['figure.titlesize'] = 18  # Figure titles
def setup_structure(molecule_name):
    """Set up and relax the initial structure"""
    # Get parameters from config
    mol_filename = molecules_data[molecule_name]["path"]
    cell = molecules_data[molecule_name]["cell"]

    # Read molecule and center it in the cell
    atoms = read(mol_filename)
    positions = atoms.get_positions()
    center_of_mass = np.mean(positions, axis=0)
    translation = np.array([cell / 2] * 3) - center_of_mass
    positions = [list(np.array(pos) + translation) for pos in positions]
    atoms.set_positions(positions)
    atoms.set_cell([cell] * 3)
    atoms.set_pbc(True)

    # Set up calculator
    atoms.set_calculator(calc)

    # Perform initial relaxation
    print("Relaxing base structure...")
    optimizer = FIRE(atoms)
    optimizer.run(fmax=0.05)

    # Get base energy
    base_relaxed = atoms.copy()
    base_relaxed.calc = calc
    base_energy = base_relaxed.get_potential_energy()
    print(f"Base structure energy: {base_energy:.3f} eV")

    return base_relaxed, base_energy


def analyze_vacancies(base_relaxed, base_energy, molecule_name, show_atom_idx=True):
    """Analyze single vacancy formation energies"""
    print("\n=== Running Vacancy Analysis ===\n")
    results_dir = f"vacancy_map_{molecule_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize lists to store results
    formation_energies = []
    atom_indices = []

    # Loop through each atom
    for atom_idx in range(len(base_relaxed)):
        # Skip non-carbon atoms
        if base_relaxed[atom_idx].symbol != 'C':
            continue

        print(f"Processing atom {atom_idx} of {len(base_relaxed)} (symbol: {base_relaxed[atom_idx].symbol})")
        atom_indices.append(atom_idx)

        try:
            # Create a vacancy by removing this atom - always start from the base relaxed structure
            vacancy_atoms = delete_atom_by_idx(base_relaxed.copy(), atom_idx)

            # Set up calculator for vacancy structure
            vacancy_atoms.set_calculator(calc)

            # Relax the vacancy structure
            print(f"  Relaxing vacancy structure...")
            optimizer = FIRE(vacancy_atoms)
            optimizer.run(fmax=0.05)


            # Calculate formation energy
            formation_energy = calculate_formation_energy(base_relaxed, vacancy_atoms, calc, MU_C)

            formation_energies.append(formation_energy)
            print(f"  Atom {atom_idx} formation energy: {formation_energy:.3f} eV")

        except Exception as e:
            print(f"  Error processing atom {atom_idx}: {e}")
            formation_energies.append(None)

    # Create output files and visualizations
    _process_vacancy_results(results_dir, base_relaxed, atom_indices, formation_energies, molecule_name, show_atom_idx)

    return atom_indices, formation_energies


def analyze_divacancies(base_relaxed, base_energy, molecule_name, max_distance=1.5, show_atom_idx=True):
    """Analyze divacancy formation energies"""
    print("\n=== Running Divacancy Analysis ===\n")
    results_dir = f"divacancy_map_{molecule_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize lists to store results
    formation_energies = []
    divacancy_pairs = []

    # Find all carbon atoms
    carbon_indices = [i for i, atom in enumerate(base_relaxed) if atom.symbol == 'C']

    # Create a list of potential divacancy pairs (filter by bond distance)
    potential_pairs = []
    for idx1, idx2 in combinations(carbon_indices, 2):
        distance = base_relaxed.get_distance(idx1, idx2)
        if distance <= max_distance:
            potential_pairs.append((idx1, idx2))

    print(f"Found {len(potential_pairs)} potential divacancy pairs within {max_distance} A")

    # Loop through each pair
    for pair_idx, (atom_idx1, atom_idx2) in enumerate(potential_pairs):
        print(f"Processing divacancy pair {pair_idx + 1}/{len(potential_pairs)}: atoms {atom_idx1} and {atom_idx2}")
        divacancy_pairs.append((atom_idx1, atom_idx2))

        try:
            # Create a divacancy by removing these atoms - always start from the base relaxed structure
            divacancy_atoms = delete_atom_by_idx(base_relaxed.copy(), [atom_idx1, atom_idx2])

            # Set up calculator for divacancy structure
            divacancy_atoms.set_calculator(calc)

            # Relax the divacancy structure
            print(f"  Relaxing divacancy structure...")
            optimizer = FIRE(divacancy_atoms)
            optimizer.run(fmax=0.05)

            # Get divacancy energy
            divacancy_energy = divacancy_atoms.get_potential_energy()

            # Calculate formation energy
            mu_C = base_energy / len(base_relaxed)  # Chemical potential of carbon
            formation_energy = divacancy_energy - (len(divacancy_atoms) * mu_C)

            formation_energies.append(formation_energy)
            print(f"  Divacancy {atom_idx1}-{atom_idx2} formation energy: {formation_energy:.3f} eV")

        except Exception as e:
            print(f"  Error processing divacancy {atom_idx1}-{atom_idx2}: {e}")
            formation_energies.append(None)

    # Create output files and visualizations
    _process_divacancy_results(results_dir, base_relaxed, divacancy_pairs, formation_energies, molecule_name,
                               carbon_indices, show_atom_idx)

    return divacancy_pairs, formation_energies


def analyze_stone_wales(base_relaxed, base_energy, molecule_name, max_distance=1.8, show_atom_idx=True):
    """Analyze Stone-Wales transformation formation energies"""
    print("\n=== Running Stone-Wales Analysis ===\n")
    results_dir = f"stw_map_{molecule_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize lists to store results
    formation_energies = []
    bond_pairs = []

    # Find all carbon atoms
    carbon_indices = [i for i, atom in enumerate(base_relaxed) if atom.symbol == 'C']

    # Create a list of potential bond pairs for rotation (filter by bond distance)
    potential_pairs = []
    for idx1, idx2 in combinations(carbon_indices, 2):
        distance = base_relaxed.get_distance(idx1, idx2)
        if distance <= max_distance:
            potential_pairs.append((idx1, idx2))

    print(f"Found {len(potential_pairs)} potential bonds within {max_distance} A")

    # Loop through each bond pair
    for pair_idx, (atom_idx1, atom_idx2) in enumerate(potential_pairs):
        print(f"Processing Stone-Wales bond {pair_idx + 1}/{len(potential_pairs)}: atoms {atom_idx1} and {atom_idx2}")
        bond_pairs.append((atom_idx1, atom_idx2))

        try:
            # Create a Stone-Wales defect by rotating the bond - always start from the base relaxed structure
            stw_atoms = base_relaxed.copy()

            # Apply the bond rotation transformation (90 degrees by default)
            rotation_transform = rotate_bond_transform(atom_idx1, atom_idx2, angle_degrees=90)
            stw_atoms = rotation_transform(stw_atoms)

            # Set up calculator for STW structure
            stw_atoms.set_calculator(calc)

            # Relax the STW structure
            print(f"  Relaxing Stone-Wales structure...")
            optimizer = FIRE(stw_atoms)
            optimizer.run(fmax=0.05)

            # Get STW energy
            stw_energy = stw_atoms.get_potential_energy()

            # Calculate formation energy (difference from base)
            formation_energy = stw_energy - base_energy

            formation_energies.append(formation_energy)
            print(f"  Stone-Wales {atom_idx1}-{atom_idx2} formation energy: {formation_energy:.3f} eV")

        except Exception as e:
            print(f"  Error processing Stone-Wales {atom_idx1}-{atom_idx2}: {e}")
            formation_energies.append(None)

    # Create output files and visualizations
    _process_stw_results(results_dir, base_relaxed, bond_pairs, formation_energies, molecule_name, carbon_indices,
                         show_atom_idx)

    return bond_pairs, formation_energies


def _process_vacancy_results(results_dir, base_relaxed, atom_indices, formation_energies, molecule_name,
                             show_atom_idx=True):
    """Process and visualize vacancy results"""
    # Create a dictionary mapping atom indices to formation energies
    energy_map = {idx: energy for idx, energy in zip(atom_indices, formation_energies)}

    # Save formation energies with atom indices
    np.savez(os.path.join(results_dir, "formation_energies.npz"),
             indices=np.array(atom_indices),
             energies=np.array(formation_energies))

    # Write energy map to text file for easy reference
    with open(os.path.join(results_dir, "energy_map.txt"), 'w') as f:
        f.write("Atom_Index Formation_Energy(eV)\n")
        for idx, energy in zip(atom_indices, formation_energies):
            if energy is not None:
                f.write(f"{idx} {energy:.3f}\n")
            else:
                f.write(f"{idx} None\n")

    # Filter out None values for visualization and analysis
    valid_data = [(idx, energy) for idx, energy in zip(atom_indices, formation_energies) if energy is not None]
    if valid_data:
        valid_atom_idx, valid_energies = zip(*valid_data)

        # Calculate statistics
        min_energy = min(valid_energies)
        max_energy = max(valid_energies)
        min_idx = valid_atom_idx[valid_energies.index(min_energy)]
        max_idx = valid_atom_idx[valid_energies.index(max_energy)]
        avg_energy = np.mean(valid_energies)
        std_energy = np.std(valid_energies)

        # Create a custom colormap (red = high energy, blue = low energy)
        cmap = LinearSegmentedColormap.from_list("formation_energy",
                                                 ["blue", "green", "yellow", "red"])

        # Plot the structure with atoms colored by formation energy
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot atoms colored by formation energy
        for i, atom in enumerate(base_relaxed):
            if i in energy_map and energy_map[i] is not None:
                # Normalize energy for coloring
                norm_energy = (energy_map[i] - min_energy) / (
                            max_energy - min_energy) if max_energy > min_energy else 0.5
                color = cmap(norm_energy)
            else:
                # Gray for non-carbon atoms or failed calculations
                color = 'gray'

            ax.scatter(atom.position[0], atom.position[1], c=color, s=100,
                       edgecolors='black', linewidths=1)

            # Add atom index for reference if enabled
            if show_atom_idx:
                ax.text(atom.position[0], atom.position[1], str(i), fontsize=8)

        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_energy, max_energy))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Formation Energy (eV)')

        # Set plot properties
        ax.set_aspect('equal')
        ax.set_title(f'Vacancy Formation Energy Map - {molecule_name}')
        ax.set_xlabel('X (A)')
        ax.set_ylabel('Y (A)')

        # Save the plot
        plt.savefig(os.path.join(results_dir, "formation_energy_map.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Create Gaussian distribution plot and save to file
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        # Plot histogram of formation energies
        hist, bins, _ = ax2.hist(valid_energies, bins=20, density=True, alpha=0.6, color='skyblue')

        # Plot Gaussian fit
        x = np.linspace(min_energy - 0.5, max_energy + 0.5, 1000)
        gaussian = stats.norm.pdf(x, avg_energy, std_energy)
        ax2.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian Fit\nμ={avg_energy:.3f} eV\nσ={std_energy:.3f} eV')

        # Add lines for mean and ±1 std dev
        ax2.axvline(avg_energy, color='k', linestyle='--', alpha=0.5, label='Mean')
        ax2.axvline(avg_energy + std_energy, color='k', linestyle=':', alpha=0.5, label='+1σ')
        ax2.axvline(avg_energy - std_energy, color='k', linestyle=':', alpha=0.5, label='-1σ')

        # Set plot properties
        ax2.set_xlabel('Formation Energy (eV)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title(f'Vacancy Formation Energy Distribution - {molecule_name}')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Save the distribution plot
        plt.savefig(os.path.join(results_dir, "energy_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Save distribution data for MATLAB
        with open(os.path.join(results_dir, "energy_distribution.txt"), 'w') as f:
            f.write("# Vacancy Formation Energy Distribution\n")
            f.write("# Mean: {:.6f} eV\n".format(avg_energy))
            f.write("# Std Dev: {:.6f} eV\n".format(std_energy))
            f.write("# Min: {:.6f} eV (atom {}) \n".format(min_energy, min_idx))
            f.write("# Max: {:.6f} eV (atom {}) \n".format(max_energy, max_idx))
            f.write("# Format: energy_value probability_density\n")
            for i in range(len(x)):
                f.write("{:.6f} {:.6f}\n".format(x[i], gaussian[i]))

        # Write summary file with statistics
        with open(os.path.join(results_dir, "summary.txt"), 'w') as f:
            f.write(f"Vacancy Formation Energy Analysis for {molecule_name}\n")
            f.write(f"----------------------------------------\n")
            f.write(f"Total carbon atoms analyzed: {len(atom_indices)}\n")
            f.write(f"Valid calculations: {len(valid_energies)}\n")
            f.write(f"Minimum formation energy: {min_energy:.3f} eV at atom index {min_idx}\n")
            f.write(f"Maximum formation energy: {max_energy:.3f} eV at atom index {max_idx}\n")
            f.write(f"Average formation energy: {avg_energy:.3f} eV\n")
            f.write(f"Standard deviation: {std_energy:.3f} eV\n")

            # Sort atoms by formation energy
            sorted_data = sorted(valid_data, key=lambda x: x[1])

            # List top 5 lowest formation energy sites
            f.write(f"\nTop 5 lowest formation energy sites (easiest to remove):\n")
            for i in range(min(5, len(sorted_data))):
                idx, energy = sorted_data[i]
                f.write(f"  Atom {idx}: {energy:.3f} eV\n")

            # List top 5 highest formation energy sites
            f.write(f"\nTop 5 highest formation energy sites (hardest to remove):\n")
            for i in range(min(5, len(sorted_data))):
                idx, energy = sorted_data[-(i + 1)]
                f.write(f"  Atom {idx}: {energy:.3f} eV\n")

            f.write(f"----------------------------------------\n")
    else:
        print("No valid vacancy calculations were completed.")


def _process_divacancy_results(results_dir, base_relaxed, divacancy_pairs, formation_energies, molecule_name,
                               carbon_indices, show_atom_idx=True):
    """Process and visualize divacancy results"""
    # Create a dictionary mapping divacancy pairs to formation energies
    energy_map = {pair: energy for pair, energy in zip(divacancy_pairs, formation_energies)}

    # Save formation energies with divacancy pairs
    np.savez(os.path.join(results_dir, "divacancy_energies.npz"),
             pairs=np.array(divacancy_pairs),
             energies=np.array(formation_energies))

    # Write energy map to text file for easy reference
    with open(os.path.join(results_dir, "energy_map.txt"), 'w') as f:
        f.write("Atom1_Index Atom2_Index Distance(A) Formation_Energy(eV)\n")
        for (idx1, idx2), energy in zip(divacancy_pairs, formation_energies):
            distance = base_relaxed.get_distance(idx1, idx2)
            if energy is not None:
                f.write(f"{idx1} {idx2} {distance:.3f} {energy:.3f}\n")
            else:
                f.write(f"{idx1} {idx2} {distance:.3f} None\n")

    # Filter out None values for visualization and analysis
    valid_data = [(pair, energy) for pair, energy in zip(divacancy_pairs, formation_energies) if energy is not None]
    if valid_data:
        valid_pairs, valid_energies = zip(*valid_data)

        # Calculate statistics
        min_energy = min(valid_energies)
        max_energy = max(valid_energies)
        min_pair = valid_pairs[valid_energies.index(min_energy)]
        max_pair = valid_pairs[valid_energies.index(max_energy)]
        avg_energy = np.mean(valid_energies)
        std_energy = np.std(valid_energies)

        # Create a custom colormap for edges (red = high energy, blue = low energy)
        cmap = LinearSegmentedColormap.from_list("formation_energy",
                                                 ["blue", "green", "yellow", "red"])

        # Plot the structure with divacancy pairs colored by formation energy
        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot all atoms first as gray circles
        for i, atom in enumerate(base_relaxed):
            ax.scatter(atom.position[0], atom.position[1], c='lightgray', s=100,
                       edgecolors='black', linewidths=1)
            # Add atom index for reference if enabled
            if show_atom_idx:
                ax.text(atom.position[0], atom.position[1], str(i), fontsize=8)

        # Plot the divacancy pairs as colored lines
        for (idx1, idx2), energy in energy_map.items():
            if energy is not None:
                # Normalize energy for coloring
                norm_energy = (energy - min_energy) / (max_energy - min_energy) if max_energy > min_energy else 0.5
                color = cmap(norm_energy)

                # Get atom positions
                pos1 = base_relaxed.positions[idx1]
                pos2 = base_relaxed.positions[idx2]

                # Draw line between atoms
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color=color, linewidth=2.5, alpha=0.7)

                # Mark the pair atoms with larger markers
                ax.scatter([pos1[0], pos2[0]], [pos1[1], pos2[1]], c=color, s=150,
                           edgecolors='black', linewidths=1.5, zorder=10)

        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_energy, max_energy))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Formation Energy (eV)')

        # Set plot properties
        ax.set_aspect('equal')
        ax.set_title(f'Divacancy Formation Energy Map - {molecule_name}')
        ax.set_xlabel('X (A)')
        ax.set_ylabel('Y (A)')

        # Save the plot
        plt.savefig(os.path.join(results_dir, "divacancy_energy_map.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Create another visualization showing pairs with lowest energies
        fig2, ax2 = plt.subplots(figsize=(14, 12))

        # Plot all atoms first as gray circles
        for i, atom in enumerate(base_relaxed):
            ax2.scatter(atom.position[0], atom.position[1], c='lightgray', s=100,
                        edgecolors='black', linewidths=1)
            # Add atom index for reference if enabled
            if show_atom_idx:
                ax2.text(atom.position[0], atom.position[1], str(i), fontsize=8)

        # Sort divacancies by formation energy
        sorted_data = sorted(valid_data, key=lambda x: x[1])

        # Plot only the top 10 lowest formation energy divacancies
        top_n = min(10, len(sorted_data))
        for i in range(top_n):
            (idx1, idx2), energy = sorted_data[i]

            # Get atom positions
            pos1 = base_relaxed.positions[idx1]
            pos2 = base_relaxed.positions[idx2]

            # Draw line between atoms with label
            line = ax2.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                            linewidth=3, label=f"{idx1}-{idx2}: {energy:.3f} eV")

            # Mark the pair atoms with larger markers of same color
            color = line[0].get_color()
            ax2.scatter([pos1[0], pos2[0]], [pos1[1], pos2[1]], c=color, s=150,
                        edgecolors='black', linewidths=1.5, zorder=10)

            # Add rank number at midpoint
            mid_x = (pos1[0] + pos2[0]) / 2
            mid_y = (pos1[1] + pos2[1]) / 2
            ax2.text(mid_x, mid_y, f"#{i + 1}", fontsize=12,
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

        # Set plot properties
        ax2.set_aspect('equal')
        ax2.set_title(f'Top {top_n} Lowest Energy Divacancies - {molecule_name}')
        ax2.set_xlabel('X (A)')
        ax2.set_ylabel('Y (A)')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        # Save the plot
        plt.savefig(os.path.join(results_dir, "top_divacancies.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Create Gaussian distribution plot
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        # Plot histogram of formation energies
        hist, bins, _ = ax3.hist(valid_energies, bins=20, density=True, alpha=0.6, color='skyblue')

        # Plot Gaussian fit
        x = np.linspace(min_energy - 0.5, max_energy + 0.5, 1000)
        gaussian = stats.norm.pdf(x, avg_energy, std_energy)
        ax3.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian Fit\nμ={avg_energy:.3f} eV\nσ={std_energy:.3f} eV')

        # Add lines for mean and ±1 std dev
        ax3.axvline(avg_energy, color='k', linestyle='--', alpha=0.5, label='Mean')
        ax3.axvline(avg_energy + std_energy, color='k', linestyle=':', alpha=0.5, label='+1σ')
        ax3.axvline(avg_energy - std_energy, color='k', linestyle=':', alpha=0.5, label='-1σ')

        # Set plot properties
        ax3.set_xlabel('Formation Energy (eV)')
        ax3.set_ylabel('Probability Density')
        ax3.set_title(f'Divacancy Formation Energy Distribution - {molecule_name}')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # Save the distribution plot
        plt.savefig(os.path.join(results_dir, "energy_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Save distribution data for MATLAB
        with open(os.path.join(results_dir, "energy_distribution.txt"), 'w') as f:
            f.write("# Divacancy Formation Energy Distribution\n")
            f.write("# Mean: {:.6f} eV\n".format(avg_energy))
            f.write("# Std Dev: {:.6f} eV\n".format(std_energy))
            f.write("# Min: {:.6f} eV (atoms {}) \n".format(min_energy, min_pair))
            f.write("# Max: {:.6f} eV (atoms {}) \n".format(max_energy, max_pair))
            f.write("# Format: energy_value probability_density\n")
            for i in range(len(x)):
                f.write("{:.6f} {:.6f}\n".format(x[i], gaussian[i]))

        # Write summary file with statistics
        with open(os.path.join(results_dir, "summary.txt"), 'w') as f:
            f.write(f"Divacancy Formation Energy Analysis for {molecule_name}\n")
            f.write(f"----------------------------------------\n")
            f.write(f"Total divacancy pairs analyzed: {len(divacancy_pairs)}\n")
            f.write(f"Valid calculations: {len(valid_energies)}\n")
            f.write(f"Minimum formation energy: {min_energy:.3f} eV for atoms {min_pair}\n")
            f.write(f"Maximum formation energy: {max_energy:.3f} eV for atoms {max_pair}\n")
            f.write(f"Average formation energy: {avg_energy:.3f} eV\n")
            f.write(f"Standard deviation: {std_energy:.3f} eV\n")

            # List top 10 lowest formation energy sites
            f.write(f"\nTop 10 lowest formation energy divacancies (easiest to form):\n")
            for i in range(min(10, len(sorted_data))):
                (idx1, idx2), energy = sorted_data[i]
                distance = base_relaxed.get_distance(idx1, idx2)
                f.write(f"  #{i + 1} Atoms {idx1}-{idx2} (distance: {distance:.2f} A): {energy:.3f} eV\n")

            # List top 10 highest formation energy sites
            f.write(f"\nTop 10 highest formation energy divacancies (hardest to form):\n")
            for i in range(min(10, len(sorted_data))):
                (idx1, idx2), energy = sorted_data[-(i + 1)]
                distance = base_relaxed.get_distance(idx1, idx2)
                f.write(f"  #{i + 1} Atoms {idx1}-{idx2} (distance: {distance:.2f} A): {energy:.3f} eV\n")

            f.write(f"----------------------------------------\n")
    else:
        print("No valid divacancy calculations were completed.")


def _process_stw_results(results_dir, base_relaxed, bond_pairs, formation_energies, molecule_name, carbon_indices,
                         show_atom_idx=True):
    """Process and visualize Stone-Wales results"""
    # Create a dictionary mapping bond pairs to formation energies
    energy_map = {pair: energy for pair, energy in zip(bond_pairs, formation_energies)}

    # Save formation energies with bond pairs
    np.savez(os.path.join(results_dir, "stw_energies.npz"),
             pairs=np.array(bond_pairs),
             energies=np.array(formation_energies))

    # Write energy map to text file for easy reference
    with open(os.path.join(results_dir, "energy_map.txt"), 'w') as f:
        f.write("Atom1_Index Atom2_Index Distance(A) Formation_Energy(eV)\n")
        for (idx1, idx2), energy in zip(bond_pairs, formation_energies):
            distance = base_relaxed.get_distance(idx1, idx2)
            if energy is not None:
                f.write(f"{idx1} {idx2} {distance:.3f} {energy:.3f}\n")
            else:
                f.write(f"{idx1} {idx2} {distance:.3f} None\n")

    # Filter out None values for visualization and analysis
    valid_data = [(pair, energy) for pair, energy in zip(bond_pairs, formation_energies) if energy is not None]
    if valid_data:
        valid_pairs, valid_energies = zip(*valid_data)

        # Calculate statistics
        min_energy = min(valid_energies)
        max_energy = max(valid_energies)
        min_pair = valid_pairs[valid_energies.index(min_energy)]
        max_pair = valid_pairs[valid_energies.index(max_energy)]
        avg_energy = np.mean(valid_energies)
        std_energy = np.std(valid_energies)

        # Create a custom colormap for bonds (red = high energy, blue = low energy)
        cmap = LinearSegmentedColormap.from_list("formation_energy",
                                                 ["blue", "green", "yellow", "red"])

        # Plot the structure with STW bond pairs colored by formation energy
        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot all atoms first
        for i, atom in enumerate(base_relaxed):
            ax.scatter(atom.position[0], atom.position[1], c='lightgray', s=100,
                       edgecolors='black', linewidths=1)
            # Add atom index for reference if enabled
            if show_atom_idx:
                ax.text(atom.position[0], atom.position[1], str(i), fontsize=8)

        # Plot bonds between all carbon atoms (to show the structure)
        for idx1, idx2 in combinations(carbon_indices, 2):
            distance = base_relaxed.get_distance(idx1, idx2)
            if distance <= 1.8:  # Use standard C-C bond distance
                pos1 = base_relaxed.positions[idx1]
                pos2 = base_relaxed.positions[idx2]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color='lightgray',
                        linewidth=1, alpha=0.5, zorder=1)

        # Plot the STW bond pairs as colored lines
        for (idx1, idx2), energy in energy_map.items():
            if energy is not None:
                # Normalize energy for coloring
                norm_energy = (energy - min_energy) / (max_energy - min_energy) if max_energy > min_energy else 0.5
                color = cmap(norm_energy)

                # Get atom positions
                pos1 = base_relaxed.positions[idx1]
                pos2 = base_relaxed.positions[idx2]

                # Draw line between atoms
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color=color, linewidth=3, alpha=0.8, zorder=5)

                # Mark the pair atoms
                ax.scatter([pos1[0], pos2[0]], [pos1[1], pos2[1]], c=color, s=150,
                           edgecolors='black', linewidths=1.5, zorder=10)

        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_energy, max_energy))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Formation Energy (eV)')

        # Set plot properties
        ax.set_aspect('equal')
        ax.set_title(f'Stone-Wales Formation Energy Map - {molecule_name}')
        ax.set_xlabel('X (A)')
        ax.set_ylabel('Y (A)')

        # Save the plot
        plt.savefig(os.path.join(results_dir, "stw_energy_map.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Create another visualization showing the top 10 lowest energy STW defects
        fig2, ax2 = plt.subplots(figsize=(14, 12))

        # Plot all atoms first
        for i, atom in enumerate(base_relaxed):
            ax2.scatter(atom.position[0], atom.position[1], c='lightgray', s=100,
                        edgecolors='black', linewidths=1)
            # Add atom index for reference if enabled
            if show_atom_idx:
                ax2.text(atom.position[0], atom.position[1], str(i), fontsize=8)

        # Plot all bonds lightly
        for idx1, idx2 in combinations(carbon_indices, 2):
            distance = base_relaxed.get_distance(idx1, idx2)
            if distance <= 1.8:
                pos1 = base_relaxed.positions[idx1]
                pos2 = base_relaxed.positions[idx2]
                ax2.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color='lightgray',
                         linewidth=1, alpha=0.5, zorder=1)

        # Sort STW defects by formation energy
        sorted_data = sorted(valid_data, key=lambda x: x[1])

        # Plot only the top 10 lowest formation energy Stone-Wales defects
        top_n = min(10, len(sorted_data))
        for i in range(top_n):
            (idx1, idx2), energy = sorted_data[i]

            # Get atom positions
            pos1 = base_relaxed.positions[idx1]
            pos2 = base_relaxed.positions[idx2]

            # Draw line between atoms with label
            line = ax2.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                            linewidth=4, label=f"{idx1}-{idx2}: {energy:.3f} eV")

            # Mark the pair atoms with larger markers of same color
            color = line[0].get_color()
            ax2.scatter([pos1[0], pos2[0]], [pos1[1], pos2[1]], c=color, s=150,
                        edgecolors='black', linewidths=1.5, zorder=10)

            # Add rank number at midpoint
            mid_x = (pos1[0] + pos2[0]) / 2
            mid_y = (pos1[1] + pos2[1]) / 2
            ax2.text(mid_x, mid_y, f"#{i + 1}", fontsize=12,
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

        # Set plot properties
        ax2.set_aspect('equal')
        ax2.set_title(f'Top {top_n} Lowest Energy Stone-Wales Defects - {molecule_name}')
        ax2.set_xlabel('X (A)')
        ax2.set_ylabel('Y (A)')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        # Save the plot
        plt.savefig(os.path.join(results_dir, "top_stw_defects.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Create Gaussian distribution plot
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        # Plot histogram of formation energies
        hist, bins, _ = ax3.hist(valid_energies, bins=20, density=True, alpha=0.6, color='skyblue')

        # Plot Gaussian fit
        x = np.linspace(min_energy - 0.5, max_energy + 0.5, 1000)
        gaussian = stats.norm.pdf(x, avg_energy, std_energy)
        ax3.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian Fit\nμ={avg_energy:.3f} eV\nσ={std_energy:.3f} eV')

        # Add lines for mean and ±1 std dev
        ax3.axvline(avg_energy, color='k', linestyle='--', alpha=0.5, label='Mean')
        ax3.axvline(avg_energy + std_energy, color='k', linestyle=':', alpha=0.5, label='+1σ')
        ax3.axvline(avg_energy - std_energy, color='k', linestyle=':', alpha=0.5, label='-1σ')

        # Set plot properties
        ax3.set_xlabel('Formation Energy (eV)')
        ax3.set_ylabel('Probability Density')
        ax3.set_title(f'Stone-Wales Formation Energy Distribution - {molecule_name}')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # Save the distribution plot
        plt.savefig(os.path.join(results_dir, "energy_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Save distribution data for MATLAB
        with open(os.path.join(results_dir, "energy_distribution.txt"), 'w') as f:
            f.write("# Stone-Wales Formation Energy Distribution\n")
            f.write("# Mean: {:.6f} eV\n".format(avg_energy))
            f.write("# Std Dev: {:.6f} eV\n".format(std_energy))
            f.write("# Min: {:.6f} eV (atoms {}) \n".format(min_energy, min_pair))
            f.write("# Max: {:.6f} eV (atoms {}) \n".format(max_energy, max_pair))
            f.write("# Format: energy_value probability_density\n")
            for i in range(len(x)):
                f.write("{:.6f} {:.6f}\n".format(x[i], gaussian[i]))

        # Write summary file with statistics
        with open(os.path.join(results_dir, "summary.txt"), 'w') as f:
            f.write(f"Stone-Wales Formation Energy Analysis for {molecule_name}\n")
            f.write(f"----------------------------------------\n")
            f.write(f"Total bond pairs analyzed: {len(bond_pairs)}\n")
            f.write(f"Valid calculations: {len(valid_energies)}\n")
            f.write(f"Minimum formation energy: {min_energy:.3f} eV for bond {min_pair}\n")
            f.write(f"Maximum formation energy: {max_energy:.3f} eV for bond {max_pair}\n")
            f.write(f"Average formation energy: {avg_energy:.3f} eV\n")
            f.write(f"Standard deviation: {std_energy:.3f} eV\n")

            # List top 10 lowest formation energy sites
            f.write(f"\nTop 10 lowest formation energy Stone-Wales defects (easiest to form):\n")
            for i in range(min(10, len(sorted_data))):
                (idx1, idx2), energy = sorted_data[i]
                distance = base_relaxed.get_distance(idx1, idx2)
                f.write(f"  #{i + 1} Bond {idx1}-{idx2} (distance: {distance:.2f} A): {energy:.3f} eV\n")

            # List top 10 highest formation energy sites
            f.write(f"\nTop 10 highest formation energy Stone-Wales defects (hardest to form):\n")
            for i in range(min(10, len(sorted_data))):
                (idx1, idx2), energy = sorted_data[-(i + 1)]
                distance = base_relaxed.get_distance(idx1, idx2)
                f.write(f"  #{i + 1} Bond {idx1}-{idx2} (distance: {distance:.2f} A): {energy:.3f} eV\n")

            f.write(f"----------------------------------------\n")
    else:
        print("No valid Stone-Wales calculations were completed.")


if __name__ == "__main__":
    # Configuration - modify these values directly
    molecule_name = "GQD_HEXAGON_8_8"  # Choose which molecule to analyze (e.g., QD_4, QD_7)
    defect_type = "vacancy"     # Choose 'vacancy', 'divacancy', 'stw', or 'all'
    show_atom_idx = True    # Set to False to hide atom indices in visualizations
    div_dist = 1.5          # Maximum distance for divacancy consideration (Angstrom)
    stw_dist = 1.8          # Maximum distance for Stone-Wales bond consideration (Angstrom)
    DEBUG_MODE = False      # Set to True to enable debug mode (no calculations)

    if DEBUG_MODE:
        print("Debug mode enabled - no calculations will be performed")
        view(read(molecules_data[molecule_name]["path"]))
    else:
        # GPU setup
        device = "cuda"
        orbff = pretrained.orb_v3_conservative_inf_omat(
            device=device,
            precision="float32-highest",  # or "float32-highest" / "float64
        )
        calc = ORBCalculator(orbff, device=device)

        # Setup initial structure
        base_relaxed, base_energy = setup_structure(molecule_name)

        # Run selected analysis
        if defect_type == 'vacancy' or defect_type == 'all':
            analyze_vacancies(base_relaxed, base_energy, molecule_name, show_atom_idx)

        if defect_type == 'divacancy' or defect_type == 'all':
            analyze_divacancies(base_relaxed, base_energy, molecule_name,
                               max_distance=div_dist, show_atom_idx=show_atom_idx)

        if defect_type == 'stw' or defect_type == 'all':
            analyze_stone_wales(base_relaxed, base_energy, molecule_name,
                               max_distance=stw_dist, show_atom_idx=show_atom_idx)

        print("\nAnalysis complete!")