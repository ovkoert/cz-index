## Conley-Zehnder indices in the restricted three-body problem
A python script for computing Conley-Zehnder indices of orbits in the restricted three-body problem. This repository is a companion to the arxiv paper:

arXiv:2308.03391: Symplectic geometry and space mission design by
Cengiz Aydin, Urs Frauenfelder, Otto van Koert, Dayung Koh, Agustin Moreno



### Installation and running
Clone the repository.

A working environment is given in the requirements file. For simplicity, I recommend anaconda. The following environment works for me (ubuntu 20.04 and ubuntu 22.04)

conda create --name cz_index python=3.11

conda activate cz_index

conda install numpy matplotlib jupyter nb_conda_kernels sympy scipy

pip install heyoka


Then open jupyter-notebook with the cz_rtbp notebook. This also works on windows. On MacOS (M*) I have been told that the following works instead of pip install heyoka:

conda config --add channels conda-forge

conda config --set channel_priority strict

conda install heyoka.py

The notebook maslov_illustration_pv is for visualization of the Maslov cycle and requires pyvista.
This can be installed with

pip install 'pyvista[all]' jupyterlab

The widgets are interactive, but need to be executed in order to work.


### orbits_se
This directory contains some data for orbits for convenient reproduction of the results of the paper, including the halo-polar orbit.
The data is not neatly ordered by energy, nor is the mass parameter everywhere the same, so please read the parameters before using..
