import os
import numpy as np
from ase.io import read
from ase.optimize import FIRE
from ase.visualize import view
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from config import molecules_data
from utils import rotate_bond_transform, delete_atoms_transform, move_atom_transform, get_distance, track_core_structure
import torch
torch.backends.cudnn.enabled = False
torch._dynamo.config.suppress_errors = True
# put True if you just want to see certain atoms indices, otherwise (for launch calculation, put False)
DEBUG_MODE = False

device = "cuda"
orbff = pretrained.orb_v3_conservative_inf_omat(
    device=device,
    precision="float32-high",  # or "float32-highest" / "float64
)
calc = ORBCalculator(orbff, device=device)




# ============= MOLECULE SETUP =============
# Choose which molecule to work with from config.py
molecule_name = "GQD_HEXAGON_10_10_func"
mol_finename = molecules_data[molecule_name]["path"]
cell = molecules_data[molecule_name]["cell"]

# Read molecule and center it in the cell
atoms = read(mol_finename)
positions = atoms.get_positions()
center_of_mass = np.mean(positions, axis=0)
translation = np.array([cell / 2] * 3) - center_of_mass
positions = [list(np.array(pos) + translation) for pos in positions]
atoms.set_positions(positions)
atoms.set_cell([cell] * 3)
atoms.set_pbc(True)
view(atoms)
# ============= DEFINE TRANSFORMATIONS =============
# Choose which atoms to work with
# For example, if you want to transform atoms 11 and 12:
axis_atoms = (15, 16)  # Change these numbers based on your molecule

# 1. Simple removal of one atom
remove_atoms_transforms = [
    delete_atoms_transform(axis_atoms[0])  # Removes first atom in pair
]

# 2. Remove both atoms in the pair
di_vacancy_transform = [
    delete_atoms_transform([axis_atoms[0], axis_atoms[1]])
]

# 3. Split vacancy transformation
# Moves one atom to middle of bond, then removes other atom
split_vacancy_transform = [
    move_atom_transform(
        atom_idx=axis_atoms[0],
        axis_atom1_idx=axis_atoms[0],
        axis_atom2_idx=axis_atoms[1],
        distance=get_distance(atoms, axis_atoms[0], axis_atoms[1]) / 2
    ),
    delete_atoms_transform([axis_atoms[1]])
]

# 4. Stone-Wales transformation (bond rotation)
stw_transform = [
    rotate_bond_transform(atom1_idx=axis_atoms[0], atom2_idx=axis_atoms[1])
]

# ============= CHOOSE WHICH TRANSFORMATIONS TO RUN =============
# Uncomment/comment transformations you want to use
task_names = [
    "remove_atoms_transforms",
    "di_vacancy_transform",
    "split_vacancy_transform",
    "stw_transform"

]

transforms_list = [
    remove_atoms_transforms,
    di_vacancy_transform,
    split_vacancy_transform,
    stw_transform
]

if not DEBUG_MODE:
    # ============= OPTIMIZATION AND RUNNING =============
    # Set up calculator and optimize structure
    atoms.set_calculator(calc)
    fmax = 0.05  # Maximum force criterion for optimization
    dyn = FIRE(atoms)
    dyn.run(fmax=fmax)
    atoms.write(f"relaxed_{molecule_name}.xyz")
    # Run each transformation
    for task_name, transform_list in zip(task_names, transforms_list):
        modified = track_core_structure(
            fmax=fmax,
            atoms=atoms.copy(),
            transforms=transform_list,
            calc=calc,
            task_name=task_name + f"_fmax_{fmax}_{os.path.splitext(os.path.basename(mol_finename))[0]}"
        )
