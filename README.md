# Implementation of SQM
## Dependencies 
1. The config file is loaded using [hydra](https://hydra.cc/docs/intro/).
2. To run real-world data experiments (including multidimensional data), additional packages of [pandas](https://pandas.pydata.org/) and [folktable](https://github.com/socialfoundations/folktables) are needed.

 ## MPC simulation

> python mpc_simulation_pca.py

> python mpc_simulation_lr.py


These scripts simulates the end-to-end performance of SQM on a single machine. You can vary the data dimensions, number of clients, and time delay for message passing.


## Experiments on PCA

> python main_hydra.py

This script shows the performance of SQM/centralized/local dp baselines. To alter the setup of (SSM, local, and centralized DP), change the 'setting' variable in the config file.

## Experiments on LR 

> python lr_SQM.py  data_type=folktable_income folk_state=CA gamma=1024 mu_normalized=10.43 q=0.001 epoch=2 

This script shows the performance SQM on California (CA) with effective normalized mu=10.43 and scaling parameter 1024. 

The scripts ```lr_sqm_7.sh, lr_sqm_10.sh, lr_sqm_13.sh, lr_sqm_16.sh``` run SQM on all four states (CA, TX, FL, NY) with eps=0.5, 1, 2, 4, 8

> python ldp_sgd.py folk_state=CA q=0.001 sigma_multi=0 epoch=15 seed=4 gaussian_sigma=13.285 flip_y=0.437

This script shows the performance of the local DP baseline for California (CA). Features in the dataset are perturbed with Gaussian noises and the labels are randomly flipped.

> python dpsgd.py folk_state=CA q=0.001 sigma_multi=1.15 epoch=2 

This script shows the performance of the central DPSGD for California (CA). 
