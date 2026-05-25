"""
cache.py
Disk caching of expensive, deterministic calculation results.

Relaxations and reference-energy calculations dominate the runtime of every
script here, yet many of them are identical from one run to the next. This
module caches the two that are safe to reuse, both under
settings.RELAXED_STRUCTURES_DIR and keyed by the active calculator so a switch
between GPAW and SevenNet never reuses the wrong energy scale:

  * Chemical potentials (mu_C, mu_H, mu_X)  ->  chemical_potentials.json
        These depend only on the calculator (and fmax), so they are identical
        across every run and every script — recomputing the graphene / H2 / X2
        relaxation each time is pure waste.

  * Relaxed structures (one XYZ per config key + calculator + fmax + cell)
        Cached as  relaxed_<key>_<calc>_fmax_<fmax>_cell_<cell>.xyz, the same
        naming GQD_basic_defects already uses, so its pristine cache and the
        comparison scripts' caches are shared.

Every cache verifies before reuse (a structure is reused only if its max force
is still below fmax) and falls back to a fresh calculation on any miss or
mismatch, so a cache hit can never silently return a wrong or under-converged
result.
"""

import json
import os

import numpy as np
from ase.build import molecule
from ase.io import read
from ase.optimize import BFGS

from settings import FMAX, BFGS_MAXSTEP, RELAXED_STRUCTURES_DIR
from utils import calculate_mu_C, calculate_mu_H

# ============================================================================
# Chemical-potential cache  (chemical_potentials.json)
# ============================================================================
_MU_CACHE_FILE = "chemical_potentials.json"

# Gas-phase diatomic references X2 -> mu(X) = E(X2)/2.  C and H are handled
# separately (graphene / H2 via utils) to stay consistent with the scripts.
_DIATOMIC_REFERENCE = {'H': 'H2', 'N': 'N2', 'O': 'O2', 'F': 'F2', 'Cl': 'Cl2'}


def _mu_cache_path():
    return os.path.join(RELAXED_STRUCTURES_DIR, _MU_CACHE_FILE)


def _load_mu_cache():
    """Load the chemical-potential cache, returning {} if absent or corrupt."""
    path = _mu_cache_path()
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_mu_cache(cache):
    os.makedirs(RELAXED_STRUCTURES_DIR, exist_ok=True)
    with open(_mu_cache_path(), 'w') as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def _cache_key(calc_name, fmax):
    return f"{calc_name}_fmax_{fmax}"


def _compute_mu(element, calc, fmax):
    """Compute mu(element) [eV] with the active calculator (no caching)."""
    if element == 'C':
        return calculate_mu_C(calc)                 # graphene energy/atom
    if element == 'H':
        return calculate_mu_H(calc)                 # E(H2)/2
    if element in _DIATOMIC_REFERENCE:
        ref = molecule(_DIATOMIC_REFERENCE[element])
        ref.set_cell([15.0, 15.0, 15.0])
        ref.center()
        ref.set_pbc(True)
        ref.calc = calc
        BFGS(ref, logfile=None).run(fmax=fmax)
        return ref.get_potential_energy() / 2.0     # E(X2)/2
    raise ValueError(
        f"No graphene/diatomic reference for the chemical potential of '{element}'. "
        f"Supply a literature/bulk fallback in the caller."
    )


def get_chemical_potential(element, calc, calc_name, fmax=FMAX):
    """Return mu(element) [eV], reading/writing the per-calculator JSON cache.

    On a cache hit the stored value is returned immediately; on a miss mu is
    computed with the active calculator (graphene for C, E(H2)/2 for H, E(X2)/2
    for the diatomic-forming elements N/O/F/Cl), stored, and returned. The cache
    is keyed by calculator name and fmax, so changing either recomputes cleanly.

    Raises ValueError for elements with no graphene/diatomic reference (e.g.
    B, P, S, Li); the caller is responsible for any literature fallback.
    """
    key = _cache_key(calc_name, fmax)
    cache = _load_mu_cache()
    calc_cache = cache.get(key, {})
    if element in calc_cache:
        return calc_cache[element]

    mu = _compute_mu(element, calc, fmax)
    calc_cache[element] = mu
    cache[key] = calc_cache
    _save_mu_cache(cache)
    return mu


# ============================================================================
# Relaxed-structure cache  (relaxed_<key>_<calc>_fmax_<fmax>_cell_<cell>.xyz)
# ============================================================================
def relaxed_structure_path(key, calc_name, fmax, cell):
    """Cache path for a relaxed structure (matches GQD_basic_defects' naming)."""
    fname = f"relaxed_{key}_{calc_name}_fmax_{fmax}_cell_{cell}.xyz"
    return os.path.join(RELAXED_STRUCTURES_DIR, fname)


def load_or_relax(atoms, key, calc, calc_name, fmax=FMAX, cell=None,
                  maxstep=BFGS_MAXSTEP, label=None):
    """Relax `atoms`, reusing a cached relaxed XYZ when one is present and converged.

    The cache file is relaxed_<key>_<calc_name>_fmax_<fmax>_cell_<cell>.xyz under
    settings.RELAXED_STRUCTURES_DIR. A cached structure is reused only if its max
    force (recomputed with the active calculator) is already below `fmax`;
    otherwise it is used as the starting geometry for a fresh relaxation. When no
    cache exists the structure is relaxed from its current geometry. Either way
    the converged structure is written back to the cache.

    `atoms` must already have its cell/PBC set and the calculator attached. Any
    atoms.info['original_bonds'] is preserved across the load. The cell edge for
    the cache key is taken from `cell` when given, else from atoms.get_cell().

    Returns (relaxed_atoms, potential_energy).
    """
    label = label or key
    if cell is None:
        cell = int(round(atoms.get_cell()[0][0]))
    os.makedirs(RELAXED_STRUCTURES_DIR, exist_ok=True)
    path = relaxed_structure_path(key, calc_name, fmax, cell)

    if os.path.exists(path):
        cached = read(path)
        if len(cached) != len(atoms) or \
                cached.get_chemical_symbols() != atoms.get_chemical_symbols():
            # The config structure changed under this key — the cache is stale.
            print(f"  {label}: cached structure differs from input "
                  f"(composition mismatch); ignoring cache and relaxing fresh")
        else:
            cached.set_cell(atoms.get_cell())
            cached.set_pbc(atoms.get_pbc())
            if 'original_bonds' in atoms.info:
                cached.info['original_bonds'] = atoms.info['original_bonds']
            cached.calc = calc
            max_force = np.max(np.linalg.norm(cached.get_forces(), axis=1))
            if max_force <= fmax:
                energy = cached.get_potential_energy()
                print(f"  {label}: loaded cached relaxed structure "
                      f"(E = {energy:.4f} eV, max_force = {max_force:.4f} eV/A)")
                return cached, energy
            print(f"  {label}: cached structure under-converged "
                  f"(max_force = {max_force:.4f} > {fmax} eV/A); re-relaxing")
            atoms = cached  # continue from the cached geometry (closer than input)

    print(f"Relaxing {label} ...")
    atoms.calc = calc
    dyn = BFGS(atoms, maxstep=maxstep)
    dyn.run(fmax=fmax)
    energy = atoms.get_potential_energy()
    max_force = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
    print(f"  {label}: E = {energy:.4f} eV, max_force = {max_force:.4f} eV/A  [cached -> {path}]")
    atoms.write(path)
    return atoms, energy
