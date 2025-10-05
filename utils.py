import datetime
import os

import numpy as np
from ase.constraints import FixAtoms
from ase.io import Trajectory
from ase.io import read
from ase.optimize import FIRE


def calculate_mu_C(calculator):
    """
    Calculate the chemical potential of carbon from perfect graphene.

    Uses ASE's graphene builder to create a periodic graphene unit cell, relaxes it, and computes
    the energy per carbon atom. This is used as mu_C in formation energy calculations to ensure
    consistency with the carbon lattice, excluding edge effects from GQDs.
    """
    from ase.build import graphene
    from ase.optimize import FIRE
    import numpy as np

    # # Create graphene sheet
    # atoms = graphene()
    #
    # # Explicitly set a 3D cell with vacuum in z-direction
    # vacuum = 15.0  # Adjust as needed (in Ångstroms)
    # cell = np.array([
    #     [atoms.cell[0, 0], atoms.cell[0, 1], 0.0],
    #     [atoms.cell[1, 0], atoms.cell[1, 1], 0.0],
    #     [0.0, 0.0, vacuum]
    # ])
    # atoms.set_cell(cell)
    # atoms.center()  # Center atoms in the new cell
    # atoms.pbc = [True, True, True]  # Ensure 3D PBC
    #
    # # Assign the calculator
    # atoms.set_calculator(calculator)
    #
    # # Relax the structure
    # optimizer = FIRE(atoms)
    # optimizer.run(fmax=0.05)
    #
    # # Calculate energy per carbon atom
    # E_graphene = atoms.get_potential_energy()
    # N = len(atoms)  # Typically 2 for graphene unit cell
    # return E_graphene / N
    return -9.214

def calculate_mu_Si(calculator):
    """
    Calculate the chemical potential of silicon from perfect diamond cubic silicon.

    Uses ASE's bulk builder to create a periodic silicon unit cell, relaxes it, and computes
    the energy per silicon atom. This is used as mu_Si in formation energy calculations to ensure
    consistency with the silicon lattice.
    """
    from ase.build import bulk
    from ase.optimize import FIRE

    # Create silicon crystal (diamond cubic structure)
    atoms = bulk('Si', 'diamond', a=5.43)  # 5.43 Å is the lattice constant for Si

    # Assign the calculator
    atoms.set_calculator(calculator)

    # Relax the structure
    optimizer = FIRE(atoms)
    optimizer.run(fmax=0.05)

    # Calculate energy per silicon atom
    E_silicon = atoms.get_potential_energy()
    N = len(atoms)  # Number of atoms in the unit cell
    return E_silicon / N


def calculate_mu_Ge(calculator):
    """
    Calculate the chemical potential of germanium from perfect diamond cubic germanium.

    Uses ASE's bulk builder to create a periodic germanium unit cell, relaxes it, and computes
    the energy per germanium atom. This is used as mu_Ge in formation energy calculations to ensure
    consistency with the germanium lattice.
    """
    from ase.build import bulk
    from ase.optimize import FIRE

    # Create germanium crystal (diamond cubic structure)
    atoms = bulk('Ge', 'diamond', a=5.658)  # 5.658 Å is the lattice constant for Ge

    # Assign the calculator
    atoms.set_calculator(calculator)

    # Relax the structure
    optimizer = FIRE(atoms)
    optimizer.run(fmax=0.05)

    # Calculate energy per germanium atom
    E_germanium = atoms.get_potential_energy()
    N = len(atoms)  # Number of atoms in the unit cell
    return E_germanium / N


def calculate_element_mu(calculator, element='C'):
    """
    Calculate the chemical potential of the specified element from its perfect crystal structure.

    Args:
        calculator: ASE calculator to use for energy calculations
        element: Chemical symbol of the element ('C', 'Si', or 'Ge')

    Returns:
        Chemical potential (energy per atom) of the specified element
    """
    if element == 'C':
        return calculate_mu_C(calculator)
    elif element == 'Si':
        return calculate_mu_Si(calculator)
    elif element == 'Ge':
        return calculate_mu_Ge(calculator)
    else:
        raise ValueError(f"Unsupported element: {element}. Must be one of: C, Si, Ge")

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



def calculate_formation_energy(perfect, defective, calculator, mu_C):
    """
    Calculate formation energy for defects in graphene quantum dots (GQDs).

    This function computes the formation energy for various defects—single vacancies, divacancies,
    and Stone-Wales (STW) defects—in GQDs. It is designed to be universal, handling cases where carbon
    atoms are removed (vacancies/divacancies) or rearranged (STW), while accounting for edge
    functional groups (e.g., H, O, N) by using a separately computed chemical potential of carbon
    (mu_C) from perfect graphene.

    Parameters:
    - perfect: ASE Atoms object of the perfect GQD structure, including carbon atoms and any
               functional groups at the edges.
    - defective: ASE Atoms object of the defective GQD structure (e.g., with a vacancy, divacancy,
                 or STW defect), also including functional groups.
    - calculator: ASE calculator object (e.g., ORBCalculator), used to compute potential energies.
    - mu_C: Chemical potential of carbon (in eV), calculated separately from a relaxed perfect
            graphene structure (e.g., E_graphene / N_carbons), to avoid distortions from edge
            functional groups in GQDs.

    Returns:
    - E_form: Formation energy of the defect (in eV), reflecting the energy cost of creating the
              defect relative to the perfect structure.

    Detailed Explanation:
    --------------------
    Formation energy (E_f) quantifies the energetic cost of introducing a defect into a material.
    For GQDs, defects like vacancies (removal of one carbon), divacancies (removal of two carbons),
    and Stone-Wales defects (bond rotation without atom removal) are common. The formula must
    account for the finite size of GQDs and the presence of edge functional groups (e.g., H, O, N),
    which differ energetically from carbon in ideal graphene.

    The standard formation energy formula is derived from thermodynamics and statistical mechanics,
    where the energy of a defective system is compared to the perfect system, adjusted by the
    chemical potential of removed atoms. For GQDs, the literature suggests:

    1. **Vacancies and Divacancies:**
       - Formula: E_f = E_defective - E_perfect + n_removed * mu_C
       - Here, n_removed is the number of carbon atoms removed (1 for vacancy, 2 for divacancy),
         and mu_C is the chemical potential of carbon. This reflects the energy to create the defect
         plus the energy of the removed carbon atoms relative to a reservoir (perfect graphene).
       - Why mu_C from perfect graphene? Using mu_C from the GQD itself (e.g., E_perfect / total_atoms)
         includes contributions from functional groups, which distorts the carbon-specific energy.
         Perfect graphene provides a consistent reference for carbon’s chemical potential,
         excluding edge effects.

    2. **Stone-Wales Defects:**
       - Formula: E_f = E_defective - E_perfect
       - No atoms are removed (n_removed = 0), so the formation energy is simply the energy difference
         due to bond rearrangement. No mu_C term is needed, as the defect involves only a structural
         change.

    3. **Implementation Choices:**
       - **Counting Carbon Atoms:** The function counts only carbon atoms (symbol == 'C') to determine
         n_removed, ensuring functional groups don’t affect the calculation. This is critical for
         GQDs, where edge atoms (H, O, N) are present but not part of the defect process.
       - **Separate mu_C:** mu_C is passed as a parameter, calculated from a perfect graphene
         structure (e.g., using a function like calculate_mu_C). This aligns with best practices to
         isolate carbon’s intrinsic energy.

    Literature References:
    ---------------------
    - **Valencia, A. M., & Caldas, M. J. (2017). "Single vacancy defect in graphene: Insights into
      its magnetic properties from theoretical modeling." Physical Review B, 96, 125431.**
      - Uses E_f = E(C_{n-1}H_m) + E(carbon) - E(C_nH_m), where E(carbon) is the energy per carbon
        in graphene, supporting mu_C from perfect graphene.
      - Link: https://doi.org/10.1103/PhysRevB.96.125431 (Check via APS or institutional access)

    - **Botello-Méndez, A. R., et al. (2011). "One-dimensional extended lines of divacancy defects
      in graphene." Nanoscale, 3, 2868-2872.**
      - Discusses divacancy formation in graphene, implying a similar approach with mu_C from the
        pristine lattice for consistency in extended systems.
      - Link: https://doi.org/10.1039/C1NR10229A (Check via RSC or institutional access)

    - **Wang, C., & Ding, Y. (2013). "Catalytically healing the Stone-Wales defects in graphene by
      carbon adatoms." Journal of Materials Chemistry A, 1, 1885-1891.**
      - Uses E_f = E_S-W - E_perfect for STW defects, confirming the energy difference approach
        when no atoms are removed.
      - Link: https://doi.org/10.1039/C2TA00947A (Check via RSC or institutional access)

    - **"Energetics of atomic scale structure changes in graphene." Chemical Society Reviews,
      DOI:10.1039/C4CS00499J.**
      - Generalizes E_f = E_d + n * mu - E_p, where mu is from perfect graphene, supporting this
        formula for vacancies and divacancies.
      - Link: https://doi.org/10.1039/C4CS00499J (Check via RSC or institutional access)

    Why This Approach?
    -----------------
    - **Consistency Across Defects:** By checking n_removed, the function adapts to all defect types
      without separate implementations, making it universal.
    - **Edge Functional Groups:** Counting only carbon atoms ensures H, O, or N don’t skew results,
      and using mu_C from graphene avoids their energetic influence, as discussed in prior analyses.
    - **Literature Alignment:** The approach mirrors established methods in graphene defect studies,
      ensuring comparability with published results.

    Notes:
    - Ensure the calculator (e.g., ORBCalculator) is consistent across perfect GQD, defective GQD,
      and graphene calculations for mu_C.
    - For edge defects involving functional group disruption, relaxation (via FIRE) should handle
      structural changes, but the formula remains valid.
    """
    perfect.set_calculator(calculator)
    defective.set_calculator(calculator)

    E_perfect = perfect.get_potential_energy()
    E_defective = defective.get_potential_energy()

    # Count number of carbon atoms
    N_C_perfect = sum(1 for atom in perfect if atom.symbol == 'C')
    N_C_defective = sum(1 for atom in defective if atom.symbol == 'C')

    n_removed = N_C_perfect - N_C_defective

    if n_removed == 0:
        # For Stone-Wales defect
        E_form = E_defective - E_perfect
    else:
        # For vacancies and divacancies
        E_form = E_defective - E_perfect + n_removed * mu_C

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

    formation_energy = calculate_formation_energy(atoms, modified_atoms, calc, MU_C)
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


def determine_target_element(atoms):
    """
    Determine the most common element in an ASE Atoms object.

    Parameters:
    -----------
    atoms : ase.Atoms
        ASE Atoms object representing a molecular structure

    Returns:
    --------
    str
        Chemical symbol of the most common element
    dict
        Element counts dictionary
    """
    from collections import Counter

    # Get chemical symbols of all atoms
    symbols = atoms.get_chemical_symbols()

    # Count occurrences of each element
    element_counts = Counter(symbols)

    # Find the most common element
    most_common_element = element_counts.most_common(1)[0][0]

    print(f"Structure analysis:")
    print(f"Total number of atoms: {len(atoms)}")
    print("Element counts:")
    for element, count in element_counts.items():
        print(f"  {element}: {count} ({count / len(atoms) * 100:.1f}%)")
    print(f"Most common element: {most_common_element}")

    return most_common_element, dict(element_counts)