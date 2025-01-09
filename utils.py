import datetime
import os

import ase
from ase.constraints import FixAtoms
from ase.io import Trajectory
from ase.optimize import FIRE
from ase.io import read
import numpy as np

def delete_atom_by_idx(atoms, idx):
    """
    Delete atom by index from ASE Atoms object

    Parameters:
    atoms: ASE Atoms object
    idx: integer or list of integers - atom indices to delete

    Returns:
    ASE Atoms object with deleted atoms
    """
    if isinstance(idx, int):
        idx = [idx]
    mask = np.ones(len(atoms), dtype=bool)
    mask[idx] = False
    return atoms[mask]


def move_atom_along_axis(atoms, atom_idx, axis_atom1_idx, axis_atom2_idx, distance):
    """
    Move atom along axis defined by two other atoms

    Parameters:
    atoms: ASE Atoms object
    atom_idx: integer - index of atom to move
    axis_atom1_idx: integer - first atom defining axis
    axis_atom2_idx: integer - second atom defining axis
    distance: float - distance to move (positive or negative)

    Returns:
    ASE Atoms object with moved atom
    """
    # Get positions
    pos1 = atoms.positions[axis_atom1_idx]
    pos2 = atoms.positions[axis_atom2_idx]

    # Calculate axis vector and normalize it
    axis = pos2 - pos1
    axis = axis / np.linalg.norm(axis)

    # Move atom
    atoms.positions[atom_idx] += axis * distance

    return atoms


def delete_atoms_transform(indices):
    def transform(atoms):
        return delete_atom_by_idx(atoms, indices)

    return transform


def substitute_atoms_transform(indices, symbol):
    def transform(atoms):
        modified_atoms = atoms.copy()
        for index in indices:
            modified_atoms[index].symbol = symbol
        return modified_atoms

    return transform


def move_atom_transform(atom_idx, axis_atom1_idx, axis_atom2_idx, distance):
    def transform(atoms):
        return move_atom_along_axis(atoms, atom_idx, axis_atom1_idx, axis_atom2_idx, distance)

    return transform


def get_distance(atoms, idx1, idx2):
    return atoms.get_distance(idx1, idx2)


def save_structure(atoms, filename, format='xyz'):
    """
    Save ASE Atoms structure as PDB or XYZ file

    Parameters:
    atoms: ASE Atoms object
    filename: str - output filename
    format: str - 'pdb' or 'xyz'
    """
    if not filename.endswith(f'.{format}'):
        filename += f'.{format}'

    atoms.write(filename, format=format)
    print(f"Structure saved as {filename}")


def write_traj_xyz(traj_path, output_file):
    # Load the trajectory file
    atoms_list = read(traj_path, ':')
    with open(output_file, 'w') as f:
        for atoms in atoms_list:
            f.write(f"{len(atoms)}\n")
            f.write(f"Lattice=\"{atoms.cell[0][0]} {atoms.cell[0][1]} {atoms.cell[0][2]} "
                    f"{atoms.cell[1][0]} {atoms.cell[1][1]} {atoms.cell[1][2]} "
                    f"{atoms.cell[2][0]} {atoms.cell[2][1]} {atoms.cell[2][2]}\" "
                    f"Properties=species:S:1:pos:R:3 pbc=\"{atoms.pbc[0]} {atoms.pbc[1]} {atoms.pbc[2]}\"\n")
            for atom in atoms:
                f.write(f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n")


def calculate_formation_energy(perfect, vacancy, calculator):
    """
    More accurate formation energy calculation
    """
    perfect.set_calculator(calculator)
    vacancy.set_calculator(calculator)

    E_perfect = perfect.get_potential_energy()
    E_vacancy = vacancy.get_potential_energy()

    # Chemical potential approach
    mu_Si = E_perfect / len(perfect)

    E_form = E_vacancy - (len(vacancy) * mu_Si)
    return E_form


def track_core_structure(fmax, atoms, transforms, calc, central_atom_index=0, radius=8,
                         task_name="core_tracking", fix_ends=False, fixed_atoms=[]):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(f"experiments", f"{task_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)


    fixed_indices = []
    if fix_ends:
        current_atoms = atoms.copy()
        central_pos = current_atoms.positions[central_atom_index]
        all_distances = np.linalg.norm(current_atoms.positions - central_pos, axis=1)
        fixed_indices = [i for i in range(len(atoms))
                         if all_distances[i] > radius and i != central_atom_index]

    c = FixAtoms(indices=fixed_indices+fixed_atoms)
    atoms.set_constraint(c)

    initial_distances = atoms.get_distances(central_atom_index, range(len(atoms)))
    initial_core_mask = initial_distances <= radius

    # Create trajectory files
    full_traj = Trajectory(os.path.join(experiment_dir, 'full_trajectory.traj'), 'w')
    core_traj = Trajectory(os.path.join(experiment_dir, 'core_trajectory.traj'), 'w')
    initial_core_traj = Trajectory(os.path.join(experiment_dir, 'initial_core.traj'), 'w')

    # Write the initial state to the trajectory files
    full_traj.write(atoms)
    core_traj.write(atoms[initial_core_mask])

    initial_core_atoms = atoms[initial_core_mask]
    initial_core_traj.write(initial_core_atoms)

    # Apply transformations
    modified_atoms = atoms.copy()
    for transform in transforms:
        modified_atoms = transform(modified_atoms)

    # Create a mask for the core atoms after transformations
    distances = modified_atoms.get_distances(central_atom_index, range(len(modified_atoms)))
    core_mask = distances <= radius

    modified_atoms.set_calculator(calc)

    # Define the save_core_state function
    def save_core_state(atoms=modified_atoms):
        # Use the mask to get the core atoms
        core_atoms = atoms[core_mask]

        # Save structures
        full_traj.write(atoms)
        core_traj.write(core_atoms)


    dyn = FIRE(modified_atoms)
    dyn.attach(save_core_state, interval=1)

    dyn.run(fmax=fmax)


    save_core_state(modified_atoms)

    # Close trajectories
    full_traj.close()
    core_traj.close()

    # Convert to xyz
    write_traj_xyz(os.path.join(experiment_dir, 'full_trajectory.traj'),
                   os.path.join(experiment_dir, 'full_trajectory.xyz'))
    write_traj_xyz(os.path.join(experiment_dir, 'core_trajectory.traj'),
                   os.path.join(experiment_dir, 'core_trajectory.xyz'))

    formation_energy = calculate_formation_energy(atoms, modified_atoms, calc)
    with open(os.path.join(experiment_dir, "analysis_output.txt"), "w", encoding="utf-8") as file:
        file.write("Calculation Setup Analysis:\n")
        file.write(f"Formation Energy: {formation_energy:.3f} eV")
    save_structure(modified_atoms, os.path.join(experiment_dir, f"{task_name}"))
    return modified_atoms


def rotate_bond_transform(atom1_idx, atom2_idx, angle_degrees=90):
    """
    Rotate a bond between two atoms by specified angle around the bond midpoint.

    Parameters:
    atom1_idx: int - index of first atom
    atom2_idx: int - index of second atom
    angle_degrees: float - rotation angle in degrees (default 90° for STW defects)

    Returns:
    transform function
    """

    def transform(atoms):
        modified_atoms = atoms.copy()

        # Get positions of the two atoms
        pos1 = modified_atoms.positions[atom1_idx]
        pos2 = modified_atoms.positions[atom2_idx]

        # Calculate midpoint
        midpoint = (pos1 + pos2) / 2

        # Calculate rotation axis (perpendicular to the bond)
        bond_vector = pos2 - pos1
        # Using z-axis as rotation axis for 2D structure
        rotation_axis = np.array([0, 0, 1])

        # Convert angle to radians
        angle_rad = np.radians(angle_degrees)

        # Create rotation matrix
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        # Translate to origin, rotate, translate back
        pos1_centered = pos1 - midpoint
        pos2_centered = pos2 - midpoint

        pos1_rotated = np.dot(rotation_matrix, pos1_centered) + midpoint
        pos2_rotated = np.dot(rotation_matrix, pos2_centered) + midpoint

        # Update positions
        modified_atoms.positions[atom1_idx] = pos1_rotated
        modified_atoms.positions[atom2_idx] = pos2_rotated

        return modified_atoms

    return transform