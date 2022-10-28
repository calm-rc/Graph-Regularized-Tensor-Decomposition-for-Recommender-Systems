%% Master of Science Thesis: Rohan Chandrashekar (St Id: 5238382)
%% Topic: Graph Regularized Canonical Polyadiac (GRCP) Tensor Decomposition 
clear; clc; 
%Adding paths for required files to train the model
addpath('core'); addpath(genpath('utils')); addpath('data');
feature('numCores'); 
%Initialization of GRCP model
GRCP_Init;

%% k-Cross Validation 
%Initialize parameters chosen for the model to be cross validated
if(opts.cT == 'YY')
    opts.Rank = 15;         
    opts.LReg = 1e-1;
else
    opts.Rank = 36;
    opts.LReg = 1e-2;
end
opts.NReg = 1;
opts.iterM = 1;

%Number of k-fold validations to be performed
n = numel(Data(:,1));
k_fold = int32(n/((1-opts.Tr_Split)*n));

%Data split into training and testing sets using KCross_Val
[Training_Set, Test_Set] = KCross_Val(Data,k_fold,opts.shuffle);

%% Graph Regularized Canonical Polyadiac (GRCP) Tensor Decomposition 
for i = 1:k_fold
    kData = [Training_Set{i,1} Training_Set{i,2} Training_Set{i,3} Training_Set{i,4};Test_Set{i,1} Test_Set{i,2} Test_Set{i,3} Test_Set{i,4}];
    [kA(i),klambda(i),kf(i),kTrain_NMSE(i),kTest_NMSE(i),kR_Time(i)] = ML_ALS(kData,opts);
end

%% Storing obtained results.
if(opts.fsave == 1)
    save(opts.kcfilename);
end
