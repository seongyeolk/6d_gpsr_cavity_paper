6D phase space reconstruction based on GPSR method
====

This repository includes data and code files for both simulation and experimental demonstrations of 6-dimensional phase space reconstruction using accelerating cavity and dipole magnet, related to the paper "Deployment and validation of predictive 6-dimensional beam diagnostics through generative reconstruction with standard accelerator elements."

The original GPSR method is presented in https://github.com/roussel-ryan/gpsr, while the present code is based on "https://github.com/roussel-ryan/gpsr_6d_paper"

## Installation
```shell
https://github.com/seongyeolk/6d_gpsr_cavity_paper
cd 6d_gpsr_cavity_paper
conda env create -f environment.yml
conda activate gpsr_paper_demo
pip install -e .
```

In case of Pytorch installation, libraries related to cuda calculation should be properly installed according to the GPU installed in your system. Please go to https://pytorch.org/ and check the version of libraries. 

In addition, we need to install Bmad-X simulation code (https://github.com/bmad-sim/Bmad-X) while the conda environment is activated.
Please go to the repository and follow the instruction. 

## Examples
There are two examples, one with experimental data and the other with simulation data.
Each case is included in "Experimental_Demo" and "Simulation_Demo."

