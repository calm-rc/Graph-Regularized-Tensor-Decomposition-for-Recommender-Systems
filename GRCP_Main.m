%% Master of Science Thesis: Rohan Chandrashekar (St Id: 5238382)
%% Topic: Graph Regularized Canonical Polyadiac (GRCP) Tensor Decomposition 
clear; clc; 
%Adding paths for required files to train the model
addpath('core'); addpath(genpath('utils')); addpath('data');
feature('numCores'); 
%Initialization of GRCP model
GRCP_Init;

%% Graph Regularized Canonical Polyadiac (GRCP) Tensor Decomposition 
for i = 1:length(opts.NmReg)
    opts.NReg = opts.NmReg(i);
    [A((i-1)*idx+1:i*idx),lambda((i-1)*idx+1:i*idx),f((i-1)*idx+1:i*idx),...
        Train_NMSE((i-1)*idx+1:i*idx), Test_NMSE((i-1)*idx+1:i*idx),R_Time((i-1)*idx+1:i*idx)]...
        = ML_ALS(Data,opts);
end

%% Storing obtained results.
clear i idx
A = reshape(A,sz);
f = reshape(f,sz);
lambda = reshape(lambda,sz);
R_Time = reshape(R_Time,sz);
Train_NMSE = reshape(Train_NMSE,sz);
Test_NMSE = reshape(Test_NMSE,sz);

if (opts.fsave == 1)
    save(opts.filename);
end
