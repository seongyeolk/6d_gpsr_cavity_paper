6D phase space reconstruction based on GPSR method
====

This repository includes data and code files for both simulation and experimental demonstration of 6-dimensional phase space reconstruction using accelerating cavity and dipole magnet, related to the paper "Deployment and validation of predictive 6-dimensional beam diagnostics through generative reconstruction with standard accelerator elements."
Here, the GPSR method is originally presented in https://github.com/roussel-ryan/gpsr.

## Installation
```shell
git clone https://github.com/roussel-ryan/gpsr.git
cd 6d_gpsr_cavity_paper
conda env create -f environment.yml
conda activate ps-reconstruction
pip install -e .
```

In addition, we need to install Bmad-X simulation code (https://github.com/bmad-sim/Bmad-X).
Please go to the repository and follow the instruction. 

## Examples
There are two examples, one with experimental data and the other with simulation data.
Each case is included in "Experimental_Demo" and "Simulation_Demo."

