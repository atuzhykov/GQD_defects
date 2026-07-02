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
from cycler import cycler

# ============================================================================
# PLOT STYLE  (fonts / sizes)  — npj / Nature Partner Journal look
# ============================================================================
# Sizes are deliberately large: npj Carbon figures are printed small in a
# two-column layout, so the on-screen fonts have to be big to stay legible
# once the figure is scaled down. Bump these here and every script inherits it.
FONT_FAMILY = 'Times New Roman'
FONT_SIZE = 18           # base text size
AXES_TITLESIZE = 22      # Plot titles
AXES_LABELSIZE = 20      # Axis labels
XTICK_LABELSIZE = 17     # X-axis tick labels
YTICK_LABELSIZE = 17     # Y-axis tick labels
LEGEND_FONTSIZE = 16     # Legend text
FIGURE_TITLESIZE = 24    # Figure titles
ATOM_IDX_FONTSIZE = 8    # Atom-index labels on structure/energy-map plots
                         # (used when show_atom_idx=True); increase for larger text

# ============================================================================
# npj / NATURE FIGURE AESTHETIC
# ============================================================================
# Official Nature Publishing Group (ggsci "npg") categorical palette — the
# house colours of the Nature family that npj Carbon belongs to.
NPG_PALETTE = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
               '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
AXES_LINEWIDTH = 1.6     # frame / spine thickness
LINE_LINEWIDTH = 2.6     # data-line thickness
MARKER_SIZE = 8          # marker diameter
TICK_MAJOR_WIDTH = 1.4   # major-tick thickness
TICK_MAJOR_SIZE = 6      # major-tick length

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
    # npj / Nature aesthetic: clean open frame, inward ticks, thicker lines,
    # NPG house colour cycle, frameless legend.
    mpl.rcParams['axes.prop_cycle'] = cycler(color=NPG_PALETTE)
    mpl.rcParams['axes.linewidth'] = AXES_LINEWIDTH
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['lines.linewidth'] = LINE_LINEWIDTH
    mpl.rcParams['lines.markersize'] = MARKER_SIZE
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.major.width'] = TICK_MAJOR_WIDTH
    mpl.rcParams['ytick.major.width'] = TICK_MAJOR_WIDTH
    mpl.rcParams['xtick.major.size'] = TICK_MAJOR_SIZE
    mpl.rcParams['ytick.major.size'] = TICK_MAJOR_SIZE
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['axes.grid'] = False


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
    print("npj / Nature aesthetic:")
    print(f"  npg_palette[0]    = {NPG_PALETTE[0]} (+{len(NPG_PALETTE) - 1} more)")
    print(f"  axes.linewidth    = {AXES_LINEWIDTH}")
    print(f"  line.linewidth    = {LINE_LINEWIDTH}")
    print(f"  marker.size       = {MARKER_SIZE}")
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
