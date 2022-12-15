This repository contains the Jupyter Notebooks and Python scripts used to visualize the uncertainty results with conformal prediction based on saved checkpoints of neural network force fields. 

# Content

- `train` folder contains the example script to train a neural network force field with Gaussian Multipole (GMP) fingerprinting scheme and SingleNN neural network structures. 
- `uncertainty` folder contains the example Jupyter Notebooks used to generate and quantify sharpness and calibration of different uncertainty quantification methods such as ensemble, dropout, negative least likelihood estimation and conformal prediction. The dataset used here is for MD17-Aspirin dataset. 

# Citation

The preprint associated with this repository is on arXiv: 

> [Robust and scalable uncertainty estimation with conformal prediction for machine-learned interatomic potentials](https://https://arxiv.org/abs/2208.08337)\
> Yuge Hu, Joseph Musielewicz, Zachary Ulissi, Andrew J. Medford\
> _arXiv.2208.08337_.

# Dependencies
- [AmpTorch](https://github.com/ulissigroup/amptorch)
- pykdtree (Install using pip or conda install)