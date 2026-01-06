import datetime
import os
import platform
import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, Trajectory
from ase.optimize import BFGS
from ase.visualize import view

from config import molecules_data
from utils import (
    rotate_bond_transform,
    delete_atoms_transform,
    move_atom_transform,
    get_distance, calculate_formation_energy, write_traj_xyz, save_structure
)

# ============= DEBUG MODE =============
# Set True to visualize atom indices, False to run calculations
DEBUG_MODE = False

# ============= RELAXATION SETTINGS =============
# Set True to use previously saved relaxed structure (skip relaxation)
# Set False to always relax from scratch
USE_SAVED_RELAXED = True
RELAXED_STRUCTURES_DIR = "relaxed_structures"  # Directory for saved structures


# ============= CALCULATOR SETUP =============
def setup_calculator():
    """Initialize calculator based on operating system."""
    system = platform.system()

    if system == "Linux":
        print(f"Running on {system} - using GPAW calculator")
        from gpaw import GPAW, PW, FermiDirac

        calc = GPAW(
            xc='PBE',  # GGA-PBE як у Wei2012, Şahan2019
            mode=PW(400),  # 400-500 eV за Wei2012, Hawthorne2025
            kpts=(1, 1, 1),  # Застосувати "30 Å rule" (Wei2012)
            symmetry='off',
            spinpol=True,  # ОБОВ'ЯЗКОВО для вакансій (Wei2012, Şahan2019)
            occupations=FermiDirac(0.05),
            convergence={
                'energy': 1e-5,  # 10⁻⁵ eV для Ef точності 0.1 eV (Li2005)
                'density': 1e-5,
                'eigenstates': 1e-7,
            },
            mixer={'backend': 'pulay', 'beta': 0.05, 'nmaxold': 8, 'weight': 50},
            maxiter=500,
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

    return calc


# Initialize calculator
calc = setup_calculator()


# ============= MOLECULE SETUP =============
# Choose which molecule to work with from config.py
molecule_name = "GQD_HEXAGON_3_3"
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

if DEBUG_MODE:
    view(atoms)


# ============= DEFECT CONFIGURATION =============
# Choose which atoms define the defect axis
# Use DEBUG_MODE to visualize atom indices
axis_atoms = (32, 16)  # Change these based on your molecule

# ============= TRANSFORM SELECTION =============
# Enable/disable specific defect types by setting to True/False
ENABLE_VACANCY = True          # Single atom removal
ENABLE_DIVACANCY = False        # Remove both atoms in pair
ENABLE_SPLIT_VACANCY = False    # Move one atom to middle, remove other
ENABLE_STONE_WALES = False      # Bond rotation (Stone-Wales defect)


# ============= TRANSFORM DEFINITIONS =============
transforms_config = []

# 1. Single Vacancy - removes first atom in pair
if ENABLE_VACANCY:
    transforms_config.append({
        'name': 'vacancy',
        'transforms': [delete_atoms_transform(axis_atoms[0])]
    })

# 2. Divacancy - removes both atoms in pair
if ENABLE_DIVACANCY:
    transforms_config.append({
        'name': 'divacancy',
        'transforms': [delete_atoms_transform([axis_atoms[0], axis_atoms[1]])]
    })

# 3. Split Vacancy - move one atom to bond midpoint, then remove other
if ENABLE_SPLIT_VACANCY:
    transforms_config.append({
        'name': 'split_vacancy',
        'transforms': [
            move_atom_transform(
                atom_idx=axis_atoms[0],
                axis_atom1_idx=axis_atoms[0],
                axis_atom2_idx=axis_atoms[1],
                distance=get_distance(atoms, axis_atoms[0], axis_atoms[1]) / 2
            ),
            delete_atoms_transform([axis_atoms[1]])
        ]
    })

# 4. Stone-Wales Transformation - 90° bond rotation
if ENABLE_STONE_WALES:
    transforms_config.append({
        'name': 'stone_wales',
        'transforms': [
            rotate_bond_transform(
                atom1_idx=axis_atoms[0],
                atom2_idx=axis_atoms[1],
                angle_degrees=90
            )
        ]
    })


# ============= EXECUTION =============
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


    dyn = BFGS(modified_atoms)
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

    formation_energy = calculate_formation_energy(atoms, modified_atoms, calc, -9.214)
    with open(os.path.join(experiment_dir, "analysis_output.txt"), "w", encoding="utf-8") as file:
        file.write("Calculation Setup Analysis:\n")
        file.write(f"Formation Energy: {formation_energy:.3f} eV")
    save_structure(modified_atoms, os.path.join(experiment_dir, f"{task_name}"))
    return modified_atoms


if not DEBUG_MODE:
    # Set up calculator and optimization parameters
    fmax = 0.5  # Maximum force criterion for optimization

    # Create directory for relaxed structures if it doesn't exist
    os.makedirs(RELAXED_STRUCTURES_DIR, exist_ok=True)

    # Determine calculator type for file naming
    calc_name = "GPAW" if platform.system() == "Linux" else "SevenNet"
    relaxed_file = os.path.join(
        RELAXED_STRUCTURES_DIR,
        f"relaxed_{molecule_name}_{calc_name}_fmax_{fmax}.xyz"
    )

    # Check if we should use saved relaxed structure
    if USE_SAVED_RELAXED and os.path.exists(relaxed_file):
        print(f"\n{'='*60}")
        print(f"Loading saved relaxed structure from:")
        print(f"  {relaxed_file}")
        print(f"{'='*60}\n")

        # Load relaxed structure
        atoms = read(relaxed_file)
        atoms.set_calculator(calc)

        # Verify it's properly relaxed
        forces = atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        energy = atoms.get_potential_energy()

        print(f"Loaded structure info:")
        print(f"  Energy: {energy:.6f} eV")
        print(f"  Max force: {max_force:.6f} eV/Å")
        print(f"  Convergence criterion (fmax): {fmax} eV/Å")

        if max_force > fmax:
            print(f"\n⚠️  WARNING: Loaded structure not fully converged!")
            print(f"  Re-relaxing from saved structure...")
            dyn = BFGS(atoms)
            dyn.run(fmax=fmax)
            atoms.write(relaxed_file)
            print(f"  ✓ Re-relaxed and updated: {relaxed_file}")
        else:
            print(f"  ✓ Structure is properly relaxed")
    else:
        # Relax pristine structure from scratch
        if USE_SAVED_RELAXED:
            print(f"\n⚠️  Saved relaxed structure not found: {relaxed_file}")
            print(f"Relaxing pristine structure from scratch...\n")
        else:
            print(f"\nRelaxing pristine structure from scratch...")

        atoms.set_calculator(calc)
        dyn = BFGS(atoms)
        dyn.run(fmax=fmax)

        # Save relaxed structure
        atoms.write(relaxed_file)
        print(f"\n{'='*60}")
        print(f"✓ Pristine structure relaxed and saved to:")
        print(f"  {relaxed_file}")
        print(f"{'='*60}\n")

        # Also save to root directory for compatibility
        atoms.write(f"relaxed_{molecule_name}.xyz")

    # Run each enabled transformation
    print(f"\n{'='*60}")
    print(f"Running {len(transforms_config)} defect calculations...")
    print(f"{'='*60}\n")

    for config in transforms_config:
        task_name = config['name']
        transform_list = config['transforms']

        print(f"\n{'='*60}")
        print(f"Processing: {task_name}")
        print(f"{'='*60}")

        # Create full task name with metadata
        full_task_name = (
            f"{task_name}_"
            f"fmax_{fmax}_"
            f"{os.path.splitext(os.path.basename(mol_filename))[0]}"
        )

        # Track and optimize defect structure
        modified = track_core_structure(
            fmax=fmax,
            atoms=atoms.copy(),
            transforms=transform_list,
            calc=calc,
            task_name=full_task_name
        )

        print(f"✓ Completed: {task_name}")

    print(f"\n{'='*60}")
    print(f"All calculations completed!")
    print(f"Results saved in experiments/ directory")
    print(f"{'='*60}")

else:
    print("\n" + "="*60)
    print("DEBUG MODE - Showing atom indices")
    print("="*60)
    print(f"Molecule: {molecule_name}")
    print(f"Selected axis atoms: {axis_atoms}")
    print(f"Total atoms: {len(atoms)}")
    print(f"\nRelaxation settings:")
    print(f"  USE_SAVED_RELAXED: {USE_SAVED_RELAXED}")
    print(f"  Relaxed structures directory: {RELAXED_STRUCTURES_DIR}")
    print("\nEnabled transforms:")
    for config in transforms_config:
        print(f"  - {config['name']}")
    print("\nSet DEBUG_MODE = False to run calculations")
    print("="*60)
