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

1. Go to the official Anaconda website: [https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)
2. Download the appropriate version for your operating system
3. Run the installer and follow the instructions to install Anaconda

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
