%% Master of Science Thesis: Rohan Chandrashekar (St Id: 5238382)
%% Topic: Graph Regularized Canonical Polyadiac (GRCP) Tensor Decomposition 
%% Configuration File for Data Preparation and Initialization
%Input Requirements: Data in the format NxM where,
%Data(:,1) are Set of Users,
%Data(:,2) are Set of Items,
%Data(:,3:M-2) are corresponding Set of Contexts
%Data(:,M-1) corresponds to User-Item Ratings,
%Data(:,M) are Timestamps given in Unix seconds. If no Timestamps, set DataT = 0. 

%Specifying choice of dataset (100K or 1M)
opts.dT = "100K";
%Specifying if the dataset has Timestamps or not
opts.DataT = 0;

if(opts.dT == '100K')
    if(opts.DataT)
        load('ml100k_data.mat');
    else
        load('ml100k_data_notime.mat');
    end
else
    if(opts.DataT)
        load('ml1m_data.mat');
    else
        load('ml1m_data_notime.mat');
    end
end
Data = Data(randperm(size(Data,1)),:);

%Specifying time period for data analysis - 'DD' - Days, 'MM'-Months, 'YY'-Years
if(opts.DataT)
    opts.cT = 'YY';         
    if (opts.cT == 'YY')
        dT = string(datetime(Data(:,end),'ConvertFrom','posixtime','Format','y'));
        dTm = str2double(dT);
        Time = string(min(dTm):1:max(dTm));
        Id = 1:numel(Time);
    elseif (opts.cT == 'MM')
        dT = string(datetime(Data(:,end),'ConvertFrom','posixtime','Format','MMMM'));
        if (strcmp(opts.dT,'100K'))
            Time = ["September","October","November","December","January","February","March","April"];
        else
            Time = ["April" "May" "June" "July" "August" "September","October","November","December" "January","February","March"];
        end
        Id = 1:numel(Time);
    else
        dT = string(datetime(Data(:,end),'ConvertFrom','posixtime','Format','eeee'));
        Time = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
        Id = 1:numel(Time);
    end

    Data(:,end) = Data(:,end-1);
    
    for i = 1:size(dT)
        Data(i,end-1) = Id(strcmp(dT(i),Time));
    end
end

clear i dT dTm Time Id

%% Initializations for the Algorithm
%Training - Test Data (0 - 1) Split Percentage
opts.Tr_Split = 0.8;    
%Maximum Iterations for ALS Algorithm
opts.maxALSIter = 200;   
%Convergence Tolerance for ALS Algorithm
opts.tolALS = eps;       
%Maximum Iterations for Conjugate Gradient Descent Method
opts.cgmaxIter = 30;    
%Convergence Tolerance for Conjugate Gradient Descent Method
opts.cgtol = 1e-12;    
%Print ALS iterations (1) or not (0) 
opts.printin = 0;      
%Choice of Computation: CPU (single/double precision) or GPU (gpuArray).
%GPU compatibility required on the machine using gpuArray
%https://www.mathworks.com/help/releases/R2022a/parallel-computing/run-matlab-functions-on-a-gpu.html
opts.type = "single";
%Data shuffle setting for k-cross validation given by 'on' or 'off'
opts.shuffle = 'off';

%Rank of CP Decomposition: Testing of varying model ranks can be done by 
%providing the CP ranks as [R1 R2 R3 ... RN].
opts.Rank = 1:50;         

%Regularization Coefficients: Testing of varying model regularization can 
%be done by providing the regularization coefficients [a1 a2 a3 ... aN].
%Laplacian Regularization 
opts.LReg = [0 1e-1 1e-2 1e-3];
%Nuclear Norm Regularization
opts.NmReg = [0 1 5 10];

%Number of Iterations: Provide iterM > 0 to replicate results for a given 
%combination of CP Rank and Regularization Coefficients
opts.iterM = 1;

sz = [length(opts.NmReg) opts.iterM length(opts.Rank) length(opts.LReg)];
opts.idx = prod(sz(2:end));

%Display Laplacian matrices? (1) to show, (0) to not show
opts.dispLap = 0;
%Save simulation results? (1) to save, (0) to not save
opts.fsave = 0;
%Filename for saving built models
opts.filename = "GRCP_dataset_misc";
%Filename for saving k-Cross validation models
opts.kcfilename = "GRCP_kCross_dataset_misc";
