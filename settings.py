"""
settings.py
Central, project-wide configuration constants.

Single source of truth for:
  * matplotlib plot style (fonts, sizes) — applied on import
  * figure / print quality (DPI for on-screen and saved figures)
  * relaxation / calculation constants (FMAX, BFGS_MAXSTEP, ...)

Import this module anywhere instead of re-declaring these values:

    import settings                       # applies the plot style on import
    from settings import FMAX, BFGS_MAXSTEP, SAVEFIG_DPI
    settings.apply_plot_style()           # re-apply explicitly if needed
    settings.print_settings()             # dump the active configuration

GQD_basic_defects re-exports FMAX / BFGS_MAXSTEP / USE_SAVED_RELAXED /
RELAXED_STRUCTURES_DIR from here, so existing
`from GQD_basic_defects import FMAX, BFGS_MAXSTEP` imports keep working.
"""

import matplotlib as mpl

# ============================================================================
# PLOT STYLE  (fonts / sizes)
# ============================================================================
FONT_FAMILY = 'Times New Roman'
FONT_SIZE = 12
AXES_TITLESIZE = 16      # Plot titles
AXES_LABELSIZE = 14      # Axis labels
XTICK_LABELSIZE = 12     # X-axis tick labels
YTICK_LABELSIZE = 12     # Y-axis tick labels
LEGEND_FONTSIZE = 12     # Legend text
FIGURE_TITLESIZE = 18    # Figure titles
ATOM_IDX_FONTSIZE = 8    # Atom-index labels on structure/energy-map plots
                         # (used when show_atom_idx=True); increase for larger text

# ============================================================================
# FIGURE / PRINT QUALITY
# ============================================================================
FIGURE_DPI = 150         # on-screen / interactive figure resolution
SAVEFIG_DPI = 300        # saved-figure resolution (publication / print quality)
SAVEFIG_BBOX = 'tight'   # trim surrounding whitespace when saving

# ============================================================================
# RELAXATION / CALCULATION CONSTANTS
# ============================================================================
FMAX = 0.05              # force-convergence criterion [eV/Å]
BFGS_MAXSTEP = 0.1       # max BFGS step size [Å]; ASE default 0.2, use 0.1 for
                         # GPAW defects to prevent SCF divergence
USE_SAVED_RELAXED = True            # reuse a previously saved relaxed structure
RELAXED_STRUCTURES_DIR = "relaxed_structures"  # where relaxed structures are cached


def apply_plot_style():
    """Apply the project matplotlib style to mpl.rcParams (idempotent)."""
    mpl.rcParams['font.family'] = FONT_FAMILY
    mpl.rcParams['font.size'] = FONT_SIZE
    mpl.rcParams['axes.titlesize'] = AXES_TITLESIZE
    mpl.rcParams['axes.labelsize'] = AXES_LABELSIZE
    mpl.rcParams['xtick.labelsize'] = XTICK_LABELSIZE
    mpl.rcParams['ytick.labelsize'] = YTICK_LABELSIZE
    mpl.rcParams['legend.fontsize'] = LEGEND_FONTSIZE
    mpl.rcParams['figure.titlesize'] = FIGURE_TITLESIZE
    # figure / print quality
    mpl.rcParams['figure.dpi'] = FIGURE_DPI
    mpl.rcParams['savefig.dpi'] = SAVEFIG_DPI
    mpl.rcParams['savefig.bbox'] = SAVEFIG_BBOX


def print_settings():
    """Print the active configuration details."""
    print(f"{'=' * 60}")
    print("PROJECT SETTINGS (settings.py)")
    print(f"{'=' * 60}")
    print("Plot style:")
    print(f"  font.family       = {FONT_FAMILY}")
    print(f"  font.size         = {FONT_SIZE}")
    print(f"  axes.titlesize    = {AXES_TITLESIZE}")
    print(f"  axes.labelsize    = {AXES_LABELSIZE}")
    print(f"  xtick.labelsize   = {XTICK_LABELSIZE}")
    print(f"  ytick.labelsize   = {YTICK_LABELSIZE}")
    print(f"  legend.fontsize   = {LEGEND_FONTSIZE}")
    print(f"  figure.titlesize  = {FIGURE_TITLESIZE}")
    print(f"  atom_idx_fontsize = {ATOM_IDX_FONTSIZE}")
    print("Figure / print quality:")
    print(f"  figure.dpi        = {FIGURE_DPI}")
    print(f"  savefig.dpi       = {SAVEFIG_DPI}")
    print(f"  savefig.bbox      = {SAVEFIG_BBOX}")
    print("Relaxation / calculation:")
    print(f"  FMAX                   = {FMAX} eV/A")
    print(f"  BFGS_MAXSTEP           = {BFGS_MAXSTEP} A")
    print(f"  USE_SAVED_RELAXED      = {USE_SAVED_RELAXED}")
    print(f"  RELAXED_STRUCTURES_DIR = {RELAXED_STRUCTURES_DIR}")
    print(f"{'=' * 60}")


# Apply the style as a side effect of import so every script that imports
# settings (directly or transitively) gets the consistent look automatically.
apply_plot_style()
