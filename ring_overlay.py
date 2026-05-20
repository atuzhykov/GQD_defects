"""
ring_overlay.py
Reusable ring-topology overlay for GQD defect structures.

Builds a C-only bond graph, finds rings with networkx, and draws the structure
with non-hexagonal rings (5/7/8/9-gons ...) highlighted as colored polygons.
The formation energy and ring topology are shown in the title.

Used by:
  - GQD_defects_mols_energy_comparison.py  (single relaxed-defect panel)
  - map.py                                 (per-site overlays in ring_overlays/)

Extracted from main_papers/paper1_wip/ITEST/overlay_rings.py, generalised to
accept an ASE Atoms object directly instead of reading hard-coded XYZ paths.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon

BOND_CUTOFF = 1.85  # Å, C-C only

RING_COLORS = {
    4:  ('#ff00ff', 'quadrilateral'),
    5:  ('#4477ff', 'pentagon'),
    7:  ('#ff8800', 'heptagon'),
    8:  ('#dd0000', 'octagon'),
    9:  ('#9900cc', 'nonagon'),
    10: ('#006600', '10-gon'),
    11: ('#884400', '11-gon'),
    12: ('#ff66aa', '12-gon'),
}
DEFAULT_COLOR = ('#888888', 'ring')


def _carbon_only(species, coords):
    idx = [i for i, s in enumerate(species) if s == 'C']
    return idx, coords[idx]


def _build_graph(c_coords):
    G = nx.Graph()
    G.add_nodes_from(range(len(c_coords)))
    for i in range(len(c_coords)):
        for j in range(i + 1, len(c_coords)):
            d = np.linalg.norm(c_coords[i] - c_coords[j])
            if d < BOND_CUTOFF:
                G.add_edge(i, j, dist=float(d))
    return G


def _find_rings(G):
    rings = set()
    try:
        for cycle in nx.minimum_cycle_basis(G):
            rings.add(frozenset(cycle))
    except Exception:
        pass
    for u, v in list(G.edges()):
        G2 = G.copy()
        G2.remove_edge(u, v)
        try:
            path = nx.shortest_path(G2, u, v)
            if 4 <= len(path) <= 12:
                rings.add(frozenset(path))
        except nx.NetworkXNoPath:
            pass
    return rings


def _ring_polygon_coords(ring_nodes, c_coords):
    pts = c_coords[sorted(ring_nodes)][:, :2]
    cx, cy = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    return pts[np.argsort(angles)]


def _topology_str(rings):
    non_hex = sorted(len(r) for r in rings if len(r) != 6)
    return '-'.join(str(s) for s in non_hex) if non_hex else '6-only'


def _draw(ax, species, all_coords, c_idx, c_coords, G, rings, ef, title):
    ax.set_aspect('equal')
    ax.set_facecolor('white')

    legend_patches = []
    seen_sizes = set()
    for ring in rings:
        sz = len(ring)
        if sz == 6:
            continue
        poly_xy = _ring_polygon_coords(np.array(sorted(ring)), c_coords)
        color, _ = RING_COLORS.get(sz, DEFAULT_COLOR)
        ax.add_patch(MplPolygon(poly_xy, closed=True, facecolor=color,
                                edgecolor=color, alpha=0.30, linewidth=2.5, zorder=1))
        poly_closed = np.vstack([poly_xy, poly_xy[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], color=color, linewidth=3.0, zorder=2)
        if sz not in seen_sizes:
            legend_patches.append(mpatches.Patch(facecolor=color, edgecolor=color,
                                                 alpha=0.5, label=f'{sz}-membered ring'))
            seen_sizes.add(sz)

    for u, v in G.edges():
        ax.plot([c_coords[u, 0], c_coords[v, 0]],
                [c_coords[u, 1], c_coords[v, 1]], 'k-', linewidth=2.5, zorder=3)

    for sp, pos in zip(species, all_coords):
        if sp == 'H':
            for j in c_idx:
                if np.linalg.norm(pos - all_coords[j]) < 1.3:
                    ax.plot([pos[0], all_coords[j, 0]], [pos[1], all_coords[j, 1]],
                            '-', color='#aaaaaa', linewidth=1.8, zorder=3)
    for sp, pos in zip(species, all_coords):
        if sp == 'H':
            ax.plot(pos[0], pos[1], 'o', color='white', markeredgecolor='gray',
                    markersize=10, zorder=4)
    for pos in c_coords:
        ax.plot(pos[0], pos[1], 'o', color='#555555', markersize=18, zorder=5)
        ax.plot(pos[0], pos[1], 'o', color='#cccccc', markersize=15, zorder=6)

    topo = _topology_str(rings)
    ef_str = f"$\\Delta E_f$ = {ef:.3f} eV" if ef is not None else "$\\Delta E_f$ = N/A"
    ax.set_title(f"{title}\n{ef_str}   topology: {topo}", fontsize=14, pad=6)
    if legend_patches:
        ax.legend(handles=legend_patches, loc='lower right', fontsize=11,
                  framealpha=0.9, edgecolor='gray')

    margin = 1.5
    ax.set_xlim(c_coords[:, 0].min() - margin, c_coords[:, 0].max() + margin)
    ax.set_ylim(c_coords[:, 1].min() - margin, c_coords[:, 1].max() + margin)
    ax.axis('off')


def save_ring_overlay(atoms, out_path, ef=None, title=None):
    """Render a single-panel ring-topology overlay for an ASE Atoms object.

    atoms:    ASE Atoms (relaxed defect structure)
    out_path: PNG output path
    ef:       formation energy [eV] shown in the title (optional)
    title:    panel title (optional)
    """
    species = atoms.get_chemical_symbols()
    all_coords = atoms.get_positions()
    c_idx, c_coords = _carbon_only(species, all_coords)
    if len(c_coords) == 0:
        print(f"  Warning: no carbon atoms — skipping ring overlay {out_path}")
        return
    G = _build_graph(c_coords)
    rings = _find_rings(G)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    _draw(ax, species, all_coords, c_idx, c_coords, G, rings, ef,
          title if title is not None else "")
    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
