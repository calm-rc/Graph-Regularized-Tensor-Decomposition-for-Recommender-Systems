# Graph Regularized Tensor Decomposition (GRCP) for Recommender Systems

This repository contains the code for the Master of Science Thesis "Graph Regularized Tensor Decomposition for Recommender Systems" carried out under [Dr.ir. K. Batselier](https://www.tudelft.nl/staff/k.batselier/?cHash=bc8a8a032dbc0c2e49df471ee3538c27) and [Elvin Isufi](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/multimedia-computing/people/elvin-isufi) in Systems and Control at Delft University of Technology. For any questions or suggestions, please e-mail [Rohan Chandrashekar](R.Chandrashekar@student.tudelft.nl).

# Code Structure 

The code is written on MATLAB R2021b. The required dependencies have been included in the repository as data files (data) and code files (utils and core). 

The data files comprising of .mat files, contain two datasets namely [MovieLens](https://grouplens.org/datasets/movielens/) 100k and 1M. The contents of the data files are User Id, Item Id, Rating Value and Time of Rating (in Unix Seconds since 1/1/1970 UTC). 

The code files comprising of .m script files, contains the necessary mathematical operations (Tensor and Graph theory) and algorithms (ALS and Conjugate Gradient) to evalaute the GRCP model.  

# Code Usage 

1. Download/clone the repository. 
2. The model framework is evaluated for the hyperparameters and settings set in `GRCP_Init.m`. Detailed comments have been provided in the initialization file on type and configuration of these parameters. 
3. Open MATLAB and navigate to the destination folder and run `GRCP_Main.m` from the Command Window.
4. Once completing model evaluation, k-cross validation of a model with chosen hyperparameters can be carried out by running `GRCP_kCross.m`.


# References

[1] Perraudin Nathanaël, Johan Paratte, David Shuman, Lionel Martin, Vassilis Kalofolias, Pierre Vandergheynst and David K. Hammond}, "GSPBOX: A toolbox for signal processing on graphs," [arxiv e-print](https://arxiv.org/abs/1408.5781), 08-2014

[2]  Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.4, www.tensortoolbox.org, September 21, 2022

# Copyrights

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the conditions listed in the 'LICENSE' file are met.
