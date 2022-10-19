%% Master of Science Thesis: Rohan Chandrashekar (St Id: 5238382)
%% Topic: Graph Regularized Canonical Polyadiac (GRCP) Tensor Decomposition 
clear; clc; 
addpath('gspbox'); addpath('core'); addpath('utils');
feature('numCores'); 
GRCP_Init;

%% Graph Regularized Canonical Polyadiac (GRCP) Tensor Decomposition 
for i = 1:length(opts.NmReg)
    opts.NReg = opts.NmReg(i);
    [A((i-1)*idx+1:i*idx),lambda((i-1)*idx+1:i*idx),f((i-1)*idx+1:i*idx),...
        Train_RMSE((i-1)*idx+1:i*idx), Test_RMSE((i-1)*idx+1:i*idx),R_Time((i-1)*idx+1:i*idx)]...
        = ML_ALS(Data,opts);
end

%% Storing obtained results.
clear i idx
A = reshape(A,sz);
f = reshape(f,sz);
lambda = reshape(lambda,sz);
R_Time = reshape(R_Time,sz);
Train_RMSE = reshape(Train_RMSE,sz);
Test_RMSE = reshape(Test_RMSE,sz);

if (opts.fsave == 1)
    save(opts.filename);
end
