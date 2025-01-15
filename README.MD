# Molecule Transformation using ASE and ORB Force Field

This code demonstrates how to apply various transformations to a molecule using the Atomic Simulation Environment (ASE) and the ORB force field calculator. The transformations include removing atoms, creating vacancies, and applying the Stone-Wales transformation (bond rotation).

## Prerequisites

To run this code, you need to have the following libraries and modules installed:

- ASE (Atomic Simulation Environment)
- NumPy
- ORB force field and calculator from the `orb_models` library

## Configuration

1. Set the `DEBUG_MODE` variable to `True` if you only want to display certain atom indices, or `False` if you want to perform the actual calculations.

2. Specify the molecule to work with by setting the `molecule_name` variable. The corresponding file path and cell size are retrieved from the `molecules_data` dictionary defined in the `config.py` file.

3. Define the transformations to be applied to the molecule. The code provides examples of the following transformations:
   - Simple removal of one atom (`remove_atoms_transforms`)
   - Removal of both atoms in a pair (`di_vacancy_transform`)
   - Split vacancy transformation, where one atom is moved to the middle of a bond and the other atom is removed (`split_vacancy_transform`)
   - Stone-Wales transformation, which involves rotating a bond (`stw_transform`)

   You can customize the transformations by modifying the `axis_atoms` variable and uncommenting/commenting the corresponding entries in the `task_names` and `transforms_list` variables.

## Usage

1. Set the `DEBUG_MODE` variable to `True` if you only want to display certain atom indices, or `False` if you want to perform the actual calculations.

2. Run the code using Python.

3. If `DEBUG_MODE` is set to `False`, the code will:
   - Set up the ORB force field calculator
   - Optimize the structure using the FIRE optimizer
   - Apply the selected transformations to the optimized structure
   - Write the relaxed structure to an XYZ file

4. If `DEBUG_MODE` is set to `True`, the code will only display the atom indices and not perform the actual calculations.

## Customization

- You can customize the molecule to work with by modifying the `molecule_name` variable and providing the corresponding file path and cell size in the `config.py` file.

- The transformations can be customized by modifying the `axis_atoms` variable and defining new transformation functions in the code.

- The `track_core_structure` function from the `utils` module is used to track the core structure of the molecule during the transformations. You can modify its parameters, such as the maximum force criterion (`fmax`), the atoms object, the list of transformations, the calculator, and the task name.

## Additional Notes

- The code uses the ORB force field calculator, which requires the `orb_models` library to be installed.

- The `utils` module contains utility functions used in the code, such as `rotate_bond_transform`, `delete_atoms_transform`, `move_atom_transform`, `get_distance`, and `track_core_structure`.

- The `config.py` file contains the configuration data for the molecules, such as the file path and cell size.


## Environment Setup Instructions

This guide will walk you through the steps to set up your environment for the project.

### Prerequisites

Before proceeding, ensure that you have the following:
- A compatible operating system (Windows, macOS, or Linux)
- Internet connection to download and install the necessary tools and packages

### Step 1: Install PyCharm

1. Visit the official PyCharm website: [https://www.jetbrains.com/pycharm/download/](https://www.jetbrains.com/pycharm/download/)
2. Download the appropriate version for your operating system (Windows, macOS, or Linux)
3. Follow the installation wizard to complete the installation process

### Step 2: Install Git

1. Visit the official Git website: [https://git-scm.com/downloads](https://git-scm.com/downloads)
2. Download the appropriate version for your operating system
3. Run the installer and follow the prompts to install Git

### Step 3: Install Conda

1. Go to the official Anaconda website: [https://docs.anaconda.com/miniconda/](https://www.anaconda.com/products/individual)
2. Download the appropriate version for your operating system
3. Run the installer and follow the instructions to 

### Step 4: Create a new Conda environment

1. Open a terminal or command prompt
2. Run the following command to create a new environment named "qd_defects" with Python 3.10:
   ```
   conda create --name qd_defects python=3.10 numpy=1.23.5
   ```
3. Press 'y' and hit Enter to confirm the installation of the necessary packages

### Step 5: Activate the Conda environment

1. In the terminal or command prompt, run the following command to activate the "qd_defects" environment:
   ```
   conda activate qd_defects
   ```

### Step 6: Install PyTorch and related packages

1. Check your CUDA version by running the following command in the terminal:
   ```
   nvidia-smi
   ```
2. Based on your CUDA version, run one of the following commands to install PyTorch:
   - For CUDA 11.8:
     ```
     conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
     ```
   - For CUDA 12.1:
     ```
     conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
     ```
   - For CPU only:
     ```
     conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 cpuonly -c pytorch
     ```

### Step 7: Install additional packages

1. In the terminal or command prompt, run the following commands to install the required packages:
   ```
   pip install orb-models
   pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"
   ```
