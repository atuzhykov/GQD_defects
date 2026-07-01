"""
orientational_defect.py
Bond-rotation (orientational-defect) energy-barrier scan for GQDs.

Ported from PINN-PoC/main.py into the orientational_defects project and aligned
with its conventions:
  * molecule path + cell size come from config.py
  * the calculator comes from GQD_basic_defects.setup_calculator()
    (SevenNet on Windows, GPAW/PBE on Linux) — same settings as the defect maps
  * plot style / DPI are inherited from settings.py
  * FMAX is inherited from settings.py; the MD constants live below
  * results are written to a top-level
        rotation_scan_<mol>_step_<step>_deg_cell_size_<cell>_fmax_<FMAX>_<calc>/
    folder with the same layout the *_map scripts use: relaxed_structures_xyz/
    and structure_images/ subfolders, a copied input_<basename>, and
    energy_distribution.png/.txt, summary.txt, energy_barriers.txt/.npz outputs.

Because config.py entries only carry `path` + `cell`, the rotation atom indices
(axis_bond / atoms_to_rotate / atoms_to_fix) are defined here per molecule,
mirroring the in-script `axis_atoms` pattern of GQD_basic_defects.py. Set
DEBUG_MODE = True first to open the structure in the ASE viewer and read off the
atom indices before committing to a run.
"""

import datetime
import json
import os
import shutil

import numpy as np
from ase import units
from ase.constraints import FixAtoms
from ase.io import read, write, Trajectory
from ase.md import Langevin
from ase.optimize import FIRE
from ase.visualize import view
from matplotlib import pyplot as plt

# Applies the project matplotlib style (fonts, sizes, DPI) on import.
import settings
from settings import FMAX
from config import molecules_data
from GQD_basic_defects import setup_calculator
from od_utils import (
    rotate_submolecule,
    get_gauss_distribution_function,
    plot_gauss_distribution,
    write_traj_xyz_and_plot_energy,
    record_energy,
)

# ============================================================================
# SCAN / MD SETTINGS
# ============================================================================
TOTAL_ANGLE = 360
angle_step = 10

# Molecular Dynamic setting
# The desired temperature, in Kelvin.
temperature_K = 300
# A friction coefficient in inverse ASE time units.
friction = 5e-3
# The time step in ASE time units.
timestep = 2.0 * units.fs
# Number of molecular dynamics steps to be run.
steps = 50

# FIRE settings
# Convergence criterion of the forces on atoms (inherited from settings.py).
fmax = FMAX

# ============================================================================
# ROTATION PARAMETERS (per config.py molecule key)
# ============================================================================
# axis_bond        : two atom indices defining the rotation axis
# atoms_to_rotate  : indices of the sub-group rotated about that axis
# atoms_to_fix     : indices held fixed during relaxation
# Fill these in for the molecule you select in `groups` below; use DEBUG_MODE
# to discover the indices with the ASE viewer.
rotation_params = {
    # Hydroxyl O-H torsion about the C-O axis (1-indexed in the .mol as
    # C1-O25-H37 -> 0-indexed C0-O24-H36). Rotating H36 drives the dihedral;
    # everything else relaxes.
    "GQD_HEX_2_2_OH": {
        "axis_bond": [0, 24],
        "atoms_to_rotate": [36],
        "atoms_to_fix": [],
    },
}

# ============================================================================
# RUN CONFIGURATION
# ============================================================================
# Set True to just open the structure in the ASE viewer to read atom indices.
DEBUG_MODE = False
# Way of structure optimization. True if Langevin MD, False if FIRE optimizer.
use_molecular_dynamics = False
# Store relaxation checkpoints per angle step.
store_checkpoints = False

# config.py keys to process.
groups = ["GQD_HEX_2_2_OH"]


def _results_dir(group, calc_name, cell):
    """Top-level results folder, matching the *_map naming convention."""
    suffix = f"_{calc_name}" if calc_name else ""
    return (f"rotation_scan_{group}_step_{angle_step}_deg_"
            f"cell_size_{int(cell)}_fmax_{fmax}{suffix}")


def _write_outputs(results_dir, group, calc_name, diff_predictions):
    """Write the barrier profile, Gaussian distribution, table, npz and summary."""
    images_dir = os.path.join(results_dir, "structure_images")
    angles = np.arange(0, TOTAL_ANGLE + 1, angle_step).tolist()

    # Barrier profile (analog of *_map's formation_energy_map.png)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(angles, diff_predictions, '-o', label='Change in Predictions', color='orange')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Change in Predicted Energy (eV)')
    ax.set_title(f'Differential {calc_name} Predictions for {group}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "energy_barrier_profile.png"))
    plt.close(fig)

    # Gaussian distribution (energy_distribution.png/.txt, as in the maps)
    mu, sigma, gauss_func = get_gauss_distribution_function(diff_predictions)
    plot_gauss_distribution(diff_predictions, sigma, gauss_func,
                            os.path.join(results_dir, "energy_distribution.png"))

    barrier = float(np.max(diff_predictions))
    max_angle = angles[int(np.argmax(diff_predictions))]
    with open(os.path.join(results_dir, "energy_distribution.txt"), 'w') as f:
        f.write("# Rotational Energy-Barrier Distribution\n")
        f.write(f"# Mean: {mu:.6f} eV\n")
        f.write(f"# Std Dev: {sigma:.6f} eV\n")
        f.write(f"# Barrier (max): {barrier:.6f} eV at {max_angle} deg\n")
        f.write("# Format: angle_deg delta_energy_eV\n")
        for a, e in zip(angles, diff_predictions):
            f.write(f"{a} {e:.6f}\n")

    # Machine-readable table + npz (analog of energy_map.txt / formation_energies.npz)
    np.savetxt(os.path.join(results_dir, "energy_barriers.txt"),
               np.column_stack([angles, diff_predictions]),
               header="angle_deg delta_energy_eV")
    np.savez(os.path.join(results_dir, "energy_barriers.npz"),
             angles=np.array(angles), delta_energy=np.array(diff_predictions))

    with open(os.path.join(results_dir, "summary.txt"), 'w') as f:
        f.write(f"Rotational Energy-Barrier Analysis for {group}\n")
        f.write("----------------------------------------\n")
        f.write(f"Calculator: {calc_name}\n")
        f.write(f"Total angle: {TOTAL_ANGLE} deg, step: {angle_step} deg "
                f"({len(angles)} points)\n")
        f.write(f"Rotation barrier (max): {barrier:.3f} eV at {max_angle} deg\n")
        f.write(f"Mean delta energy: {mu:.3f} eV\n")
        f.write(f"Standard deviation: {sigma:.3f} eV\n")
        f.write("----------------------------------------\n")

    return images_dir


def main():
    calc, calc_name = setup_calculator()

    for group in groups:
        predictions = []
        mol_path = molecules_data[group]["path"]
        cell = molecules_data[group]["cell"]
        axis_bond = rotation_params[group]["axis_bond"]
        atoms_to_rotate = rotation_params[group]["atoms_to_rotate"]
        atoms_to_fix = rotation_params[group]["atoms_to_fix"]

        if DEBUG_MODE:
            view(read(mol_path))
            continue

        results_dir = _results_dir(group, calc_name, cell)
        structures_xyz_dir = os.path.join(results_dir, "relaxed_structures_xyz")
        images_dir = os.path.join(results_dir, "structure_images")
        os.makedirs(structures_xyz_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        # Copy the input structure and dump the run configuration.
        shutil.copy(mol_path, os.path.join(results_dir, f"input_{os.path.basename(mol_path)}"))
        with open(os.path.join(results_dir, f"{group}_info.json"), 'w') as json_file:
            json.dump({**molecules_data[group], **rotation_params[group],
                       "angle_step": angle_step, "total_angle": TOTAL_ANGLE,
                       "fmax": fmax, "calculator": calc_name}, json_file, indent=4)

        ase_atoms = read(mol_path)
        positions = ase_atoms.get_positions()
        center_of_mass = np.mean(positions, axis=0)
        translation = np.array([cell / 2, cell / 2, cell / 2]) - center_of_mass
        positions = [list(np.array(pos) + translation) for pos in positions]
        ase_atoms.set_positions(positions)
        ase_atoms.set_cell([cell, cell, cell])
        ase_atoms.set_pbc(True)
        c = FixAtoms(indices=atoms_to_fix + atoms_to_rotate)
        ase_atoms.set_constraint(c)
        ase_atoms.calc = calc
        write(os.path.join(images_dir, f'{group}.png'), ase_atoms)

        # initial structure optimization
        if use_molecular_dynamics:
            dyn = Langevin(ase_atoms, timestep, temperature_K=temperature_K, friction=friction)
            dyn.run(steps)
        else:
            dyn = FIRE(ase_atoms)
            dyn.run(fmax=fmax)

        rotated_mols = [ase_atoms]
        predictions.append(ase_atoms.get_potential_energy())
        angle = angle_step
        while angle <= TOTAL_ANGLE:
            energies = []
            print(f"{angle} deg from {TOTAL_ANGLE} deg with step: {angle_step} deg was processed for {group}")
            ase_atoms = rotate_submolecule(ase_atoms, angle_step, axis_bond, atoms_to_rotate, cell)
            ase_atoms.calc = calc
            ase_atoms.set_constraint(c)

            # Structure optimization block: proceed and store energies minimization
            # plot and atoms positions trajectories for OVITO.
            traj = None
            if store_checkpoints:
                traj_filename = os.path.join(structures_xyz_dir,
                                             f"{calc_name}_{group}_step_{angle}_trajectory.traj")
                traj = Trajectory(traj_filename, 'w', ase_atoms)
            if use_molecular_dynamics:
                dyn = Langevin(ase_atoms, timestep, temperature_K=temperature_K, friction=friction)
                if store_checkpoints:
                    dyn.attach(record_energy, interval=1, atoms=ase_atoms, energies=energies)
                    dyn.attach(traj.write, interval=1)
                dyn.run(steps)
            else:
                dyn = FIRE(ase_atoms)
                if store_checkpoints:
                    dyn.attach(record_energy, interval=1, atoms=ase_atoms, energies=energies)
                    dyn.attach(traj.write, interval=1)
                dyn.run(fmax=fmax)

            if store_checkpoints:
                traj.write(ase_atoms)
                traj.close()
                write_traj_xyz_and_plot_energy(
                    traj_filename,
                    os.path.join(structures_xyz_dir, f"{calc_name}_{group}_step_{angle}_trajectory.xyz"),
                    energies)

            rotated_mols.append(ase_atoms.copy())
            predictions.append(ase_atoms.get_potential_energy())

            angle += angle_step

        # for OVITO animation
        write(os.path.join(structures_xyz_dir,
                           f"{calc_name}_with_{os.path.splitext(os.path.basename(mol_path))[0]}_rotated.xyz"),
              rotated_mols, 'extxyz')

        min_pred = np.min([predictions])
        diff_predictions = [item - min_pred for item in predictions]
        _write_outputs(results_dir, group, calc_name, diff_predictions)
        print(f"Results written to: {results_dir}")


if __name__ == "__main__":
    main()
