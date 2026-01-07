# Graphene Quantum Dot Defect Analysis

A computational chemistry toolkit for analyzing structural defects, doping, and functionalization in graphene quantum dots (GQDs) and 2D carbon-based materials using DFT and machine learning potentials.

## Overview

This project performs comprehensive atomic-scale analysis of defects in graphene quantum dots, including:
- **Vacancy analysis**: Single and double atom removal
- **Stone-Wales defects**: Bond rotation transformations
- **Doping studies**: Substitutional doping with B, N, P, etc.
- **Functionalization**: Edge and surface functionalization with chemical groups (NH2, COOH, OH, etc.)
- **Electronic properties**: HOMO/LUMO, band gap, optical absorption, electron density
- **Formation energy mapping**: Systematic analysis of all possible defect sites

## Key Features

- **Dual calculator support**: GPAW (DFT, Linux) and SevenNet (ML potential, Windows)
- **Automated defect mapping**: Systematic exploration of all possible defect configurations
- **Comprehensive visualization**: Energy maps, distribution plots, structure images
- **Formation energy calculations**: Quantitative comparison of defect stability
- **DFT electronic structure analysis**: HOMO/LUMO, optical spectra, phonons (GPAW only)

## Project Structure

```
.
├── map.py                              # Main defect mapping tool (systematic analysis)
├── GQD_basic_defects.py               # Single defect application tool
├── gpaw_analysis.py                   # GPAW DFT electronic structure analysis
├── GQD_dopants_mols_energy_comparison.py       # Doping comparison tool
├── GQD_functionalization_mols_energy_comparison.py  # Functionalization comparison
├── utils.py                           # Utility functions
├── config.py                          # Molecule database configuration
└── data/                              # Input molecular structures (.mol files)
```

## Installation

### Prerequisites

Choose your platform and calculator:

#### Linux with GPAW (DFT)
Recommended for accurate electronic structure calculations.

#### Windows with SevenNet (ML potential)
Faster calculations, suitable for large-scale screening.

### Environment Setup

1. **Install Miniconda/Anaconda**
   - Download from: https://docs.anaconda.com/miniconda/

2. **Create conda environment**
   ```bash
   conda create --name gqd_defects python=3.10 numpy=1.23.5
   conda activate gqd_defects
   ```

3. **Install ASE (Atomic Simulation Environment)**
   ```bash
   pip install ase
   ```

4. **Install calculator (choose one)**

   **Option A: GPAW (Linux only)**
   ```bash
   # Check CUDA version if using GPU
   nvidia-smi

   # Install PyTorch with appropriate CUDA version
   # For CUDA 11.8:
   conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
   # For CUDA 12.1:
   conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

   # Install GPAW
   pip install gpaw

   # Install GPAW datasets
   gpaw install-data <directory>
   ```

   **Option B: SevenNet (Windows)**
   ```bash
   # Install PyTorch
   conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 cpuonly -c pytorch

   # Install SevenNet
   pip install sevenn-calculator
   ```

5. **Install additional dependencies**
   ```bash
   pip install matplotlib scipy
   pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"
   ```

## Usage

### 1. Systematic Defect Mapping (map.py)

Automatically analyzes all possible defect sites in a molecule.

```python
# Edit configuration in map.py
molecule_name = "Coronene"      # Choose molecule from config.py
defect_type = "vacancy"         # Options: 'vacancy', 'divacancy', 'stw', 'all'
show_atom_idx = True           # Display atom indices in plots
DEBUG_MODE = False             # Set True to visualize structure first

# Run analysis
python map.py
```

**Output** (in `{defect_type}_map_{molecule_name}/`):
- `formation_energy_map.png`: Color-coded defect energy map
- `energy_distribution.png`: Statistical distribution with Gaussian fit
- `summary.txt`: Detailed statistics and rankings
- `energy_map.txt`: Formation energies for all sites
- `relaxed_structures_xyz/`: Relaxed structures in XYZ format
- `relaxed_structures_mol/`: Relaxed structures in MOL format with bond info
- `structure_images/`: Visualizations of each defect

### 2. Single Defect Analysis (GQD_basic_defects.py)

Apply specific defects to analyze individual configurations.

```python
# Configure in GQD_basic_defects.py
molecule_name = "GQD_HEXAGON_3_3"
axis_atoms = (32, 16)          # Atom indices defining defect location

# Enable desired defect types
ENABLE_VACANCY = True          # Single atom removal
ENABLE_DIVACANCY = True        # Double atom removal
ENABLE_SPLIT_VACANCY = True    # Move + remove
ENABLE_STONE_WALES = True      # Bond rotation

# Run
python GQD_basic_defects.py
```

### 3. GPAW Electronic Structure Analysis (gpaw_analysis.py)

Comprehensive DFT analysis of electronic properties (Linux only).

```python
from gpaw_analysis import GPAWAnalyzer

# Create analyzer
analyzer = GPAWAnalyzer(
    molecule_name="Coronene",
    xc='PBE',
    mode_cutoff=300,
    spinpol=True
)

# Run full analysis
results = analyzer.run_full_analysis(
    include_optical=True,
    include_phonons=False
)

# Access results
print(f"Band gap: {results['homo_lumo']['gap']:.3f} eV")
```

**Calculated properties**:
- Ground state energy
- HOMO/LUMO eigenvalues and band gap
- Electronic density (cube format for visualization)
- Optical absorption spectrum
- Dipole moment
- Vibrational frequencies (optional)

## Configuration

### Adding New Molecules

Edit `config.py`:

```python
molecules_data = {
    "Your_Molecule": {
        "path": os.path.join("data", "your_molecule.mol"),
        "cell": 25  # Cell size in Angstroms
    }
}
```

Place MOL files in `data/` directory.

### Calculator Selection

The code automatically selects the calculator based on the operating system:
- **Linux**: GPAW (DFT)
- **Windows**: SevenNet (ML potential)

## Output Files

### Defect Mapping
- `formation_energy_map.png`: Spatial visualization of defect energies
- `energy_distribution.png`: Histogram with Gaussian fit
- `summary.txt`: Statistical analysis and top/bottom rankings
- `energy_map.txt`: Raw data (atom indices and energies)
- `relaxed_structures_xyz/`: Optimized structures for each defect
- `relaxed_structures_mol/`: Structures with bond information preserved

### GPAW Analysis
- `relaxed_{molecule}.xyz`: Optimized structure
- `homo_lumo.txt`: Electronic eigenvalues
- `energy_levels.png`: Energy level diagram
- `density.cube`: Electron density (for VESTA/VMD visualization)
- `density_slice.png`: 2D density cross-section
- `absorption_spectrum.png`: Optical properties
- `phonon_frequencies.txt`: Vibrational modes
- `analysis_summary.txt`: Complete results summary

## Supported Defect Types

1. **Vacancy**: Single atom removal
   - Formation energy: E_form = E_defect - E_pristine + μ_element

2. **Divacancy**: Double atom removal (adjacent atoms)
   - Identifies all atom pairs within bonding distance

3. **Stone-Wales (STW)**: 90° bond rotation
   - Topological defect without atom removal

4. **Doping**: Substitutional replacement with heteroatoms
   - Supported elements: B, N, P, Si, Ge, etc.

5. **Functionalization**: Addition of chemical groups
   - Examples: -NH2, -COOH, -OH, -BH2

## Computational Details

### GPAW Settings (Linux)
- **XC functional**: PBE (GGA)
- **Cutoff energy**: 300-400 eV
- **k-points**: Γ-point (1×1×1) for molecules
- **Spin polarization**: Enabled (essential for radicals/defects)
- **Convergence**: 0.05 eV/Å force criterion

### SevenNet Settings (Windows)
- **Model**: 7net-l3i5 with 'mpa' mode
- **Training data**: MPtrj + sAlex dataset
- **Performance**: ~100× faster than DFT, good accuracy for C-based materials

### Element-Specific Parameters

Bond distance thresholds (for divacancy/STW analysis):
- **Carbon**: divacancy 1.7 Å, STW 1.8 Å
- **Silicon**: divacancy 2.3 Å, STW 2.6 Å
- **Germanium**: divacancy 2.5 Å, STW 2.8 Å

## Tips and Best Practices

1. **Use DEBUG_MODE first**: Set `DEBUG_MODE = True` to visualize atom indices before running calculations

2. **Start with small molecules**: Test workflow with coronene or small GQDs before large systems

3. **Check convergence**: Verify relaxed structures have max forces < 0.05 eV/Å

4. **Reuse relaxed structures**: Set `USE_SAVED_RELAXED = True` in GQD_basic_defects.py to skip pristine structure relaxation

5. **Parallel calculations**: For large-scale screening, use map.py which saves each result independently

6. **Visualization**:
   - Use VESTA or VMD to visualize `.cube` density files
   - Use ASE GUI for `.xyz` and `.traj` files: `ase gui file.xyz`

## Troubleshooting

### GPAW convergence issues
- Increase `maxiter` in calculator settings
- Try different mixing parameters
- Use smaller `mode_cutoff` for initial testing
- Enable `spinpol=True` for open-shell systems

### Memory issues
- Reduce `mode_cutoff` (e.g., 300 instead of 400 eV)
- Use smaller molecules or cell sizes
- On Windows, SevenNet uses much less memory than GPAW

### Platform-specific issues
- **Linux**: Ensure GPAW PAW datasets are installed
- **Windows**: SevenNet requires PyTorch but works on CPU

## Citation

If you use this code, please cite the relevant papers for the calculators:

**GPAW**:
- J. J. Mortensen et al., Phys. Rev. B 71, 035109 (2005)

**SevenNet**:
- Park et al., arXiv:2409.06676 (2024)

**ASE**:
- A. H. Larsen et al., J. Phys.: Condens. Matter 29, 273002 (2017)

## License

This project is for academic research purposes.

## Contact

For questions or issues, please refer to the respective calculator documentation:
- GPAW: https://wiki.fysik.dtu.dk/gpaw/
- SevenNet: https://github.com/MDIL-SNU/SevenNet
- ASE: https://wiki.fysik.dtu.dk/ase/
