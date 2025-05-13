import torch
from ase import Atoms
from ase.optimize import FIRE
from ase.visualize import view
import numpy as np
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
torch.backends.cudnn.enabled = False
torch._dynamo.config.suppress_errors = True
# Define supercell dimensions
M = 9  # Number of repetitions in x-direction
N = 9  # Number of repetitions in y-direction

# Choose the element: "Si" for silicene, "Ge" for germanene
element = "Si"  # Change to "Si" for silicene

# Define material parameters
if element == "Si":
    # Silicene parameters
    a = 3.90  # Lattice constant (Å)
    d = 2.23  # Si–Si bond length (Å)
    buckling = 0.44  # Buckling height (Å)
    material_name = "silicene"
    atomic_number = 14  # For Jmol (Si)
elif element == "Ge":
    # Germanene parameters
    a = 3.62  # Lattice constant (Å)
    d = 2.2  # Ge–Ge bond length (Å)
    buckling = 0.64  # Buckling height (Å)
    material_name = "germanene"
    atomic_number = 32  # For Jmol (Ge)
else:
    raise ValueError("Element must be 'Si' (silicene) or 'Ge' (germanene)")

# Define the unit cell for a honeycomb lattice
# Lattice vectors for a hexagonal lattice
a1 = [a, 0, 0]
a2 = [a/2, a * np.sqrt(3)/2, 0]
a3 = [0, 0, 10]

# Positions of the two atoms in the unit cell
# Atom 1 (sublattice A) at (0, 0, 0)
# Atom 2 (sublattice B) at (a/3, a*sqrt(3)/6, buckling)
atoms = Atoms(
    symbols=f'{element}2',
    positions=[
        (0, 0, 0),                     # Sublattice A
        (a/3, a * np.sqrt(3)/6, buckling)  # Sublattice B
    ],
    cell=[a1, a2, a3],
    pbc=[True, True, False]
)

# Replicate the unit cell to form an MxN supercell
atoms = atoms.repeat((M, N, 1))

# Center the structure (optional, for better visualization)
atoms.center()

# Perform relaxation
device = "cuda"
orbff = pretrained.orb_v3_conservative_inf_omat(
    device=device,
    precision="float32-highest",  # or "float32-highest" / "float64
)
calc = ORBCalculator(orbff, device=device)
atoms.calc = calc
atoms.pbc = False
print(f"Relaxing {material_name} structure...")
optimizer = FIRE(atoms)
optimizer.run(fmax=0.05, steps=200)  # Convergence criterion: forces < 0.05 eV/Å
print("Relaxation completed!")

# Visualize the structure in ASE
view(atoms)

# Save the structure for Jmol
atoms.write(f'{material_name}_{M}x{N}.xyz')
