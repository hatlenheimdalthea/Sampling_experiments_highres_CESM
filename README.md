# Sampling_experiments_highres_CESM

!! Code and documentation in progress !!

Code to implement the pCO2-Residual method (Bennington et al. 2022; https://doi.org/10.1029/2021ms002960) using the high-resolution CESM testbed (Krumhardt et al., 2024; https://doi.org/10.1016/j.pocean.2024.103314).

This code is set up to be run on the LEAP Pangeo computing platform (see documentation here: https://leap-stc.github.io/intro.html)

Brief summary on each notebook:

mCDR_utils: Supporting functions for notebooks 01 and 02
01_create_dataframe : Creates a dataframe with all variables needed to run notebook 02
02_ML : Reconstructs pCO2-Residual using XGBoost, and adds pCO2-T to pCO2-Residual to get pCO2
Figures#1, Figures#2, Figures#3, _Val_mapping.ipynb: Notebooks to create figures
