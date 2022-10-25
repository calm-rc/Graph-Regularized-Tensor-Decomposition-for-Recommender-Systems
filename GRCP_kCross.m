%% Master of Science Thesis: Rohan Chandrashekar (St Id: 5238382)
%% Topic: Graph Regularized Canonical Polyadiac (GRCP) Tensor Decomposition 
clear; clc; 
addpath('gspbox'); addpath('core'); addpath('utils');
feature('numCores'); 
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

for i = 1:k_fold
    kData = [Training_Set{i,1} Training_Set{i,2} Training_Set{i,3} Training_Set{i,4};Test_Set{i,1} Test_Set{i,2} Test_Set{i,3} Test_Set{i,4}];
    [kA(i),klambda(i),kf(i),kTrain_RMSE(i),kTest_RMSE(i),kR_Time(i)] = ML_ALS(kData,opts);
end

if(opts.fsave == 1)
    save(opts.kcfilename);
end