import numpy as np
from ase.io import read
from ase.optimize import FIRE
from ase.visualize import view
from matplotlib import pyplot as plt
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from config import molecules_data
from utils import move_atom_transform, track_core_structure

# set True if you just want to see certain atoms indices, otherwise (for launch calculation set False)
DEBUG_MODE = False


molecule_name = "QD_6"
mol_finename = molecules_data[molecule_name]["path"]
cell = molecules_data[molecule_name]["cell"]
atoms = read(mol_finename)
positions = atoms.get_positions()
center_of_mass = np.mean(positions, axis=0)
translation = np.array([cell / 2, cell / 2, cell / 2]) - center_of_mass
positions = [list(np.array(pos) + translation) for pos in positions]
atoms.set_positions(positions)
atoms.set_cell([cell, cell, cell])
atoms.set_pbc(True)
view(atoms)
axis_atoms = (11, 12)
start_point = 1.3
end_point = 1.4
step = 0.1

is_dual_inverse_bond_transforms = True # put False for single_inverse_bond_transforms
if not DEBUG_MODE:
    device = "cuda"
    orbff = pretrained.orb_v2(device=device)
    calc = ORBCalculator(orbff, device=device)
    atoms.set_calculator(calc)
    fmax = 0.05
    dyn = FIRE(atoms)
    dyn.run(fmax=fmax)


    distances = np.arange(start_point, end_point, step)
    plt.figure(figsize=(8, 6))
    energies = []
    for distance in distances:
        if is_dual_inverse_bond_transforms:
            modified = track_core_structure(
                fmax=fmax,
                atoms=atoms.copy(),
                transforms=[
                    move_atom_transform(
                        atom_idx=axis_atoms[0],
                        axis_atom1_idx=axis_atoms[0],
                        axis_atom2_idx=axis_atoms[1],
                        distance=-distance),
                    move_atom_transform(
                        atom_idx=axis_atoms[1],
                        axis_atom1_idx=axis_atoms[0],
                        axis_atom2_idx=axis_atoms[1],
                        distance=distance)
                ],
                fixed_atoms=list(axis_atoms),
                fix_ends=False,
                calc=ORBCalculator(orbff, device=device),
                task_name=f"dual_inverse_distance_{distance:2f}_fmax_{fmax}")
            energies.append(modified.get_potential_energy())
        else:
            modified = track_core_structure(
                fmax=fmax,
                atoms=atoms.copy(),
                transforms=[
                    move_atom_transform(
                        atom_idx=axis_atoms[0],
                        axis_atom1_idx=axis_atoms[0],
                        axis_atom2_idx=axis_atoms[1],
                        distance=-distance),
                ],
                fixed_atoms=axis_atoms[0],
                fix_ends=True,
                central_atom_index=axis_atoms[0],
                calc=ORBCalculator(orbff, device=device),
                task_name=f"single_inverse_distance_{distance:2f}_fmax_{fmax}")
            energies.append(modified.get_potential_energy())


    energy_min = min(energies)
    relative_energies = [e - energy_min for e in energies]
    plt.plot(distances, relative_energies, marker='o', label=f'fmax = {fmax} eV/Å')

    energy_min = min(energies)
    relative_energies = [e - energy_min for e in energies]

    with open(f'{molecule_name}_energy_vs_distance_{fmax}.txt', 'w') as f:
        f.write("Displacement(A) RelativeEnergy(eV)\n")  # Header
        for d, e in zip(distances, relative_energies):
            f.write(f"{d:.6f} {e:.6f}\n")

    plt.plot(distances, relative_energies, marker='o')
    plt.xlabel('Displacement (Å)')
    plt.ylabel('Relative Energy (eV)')
    plt.title('Potential Energy Surface')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{molecule_name}_fmax_{fmax}.png')
    plt.close()
