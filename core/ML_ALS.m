function [A,lambda,f,Train_NMSE,Test_NMSE,R_Time] = ML_ALS(Data,opts)
%Importing User-Item-Time-Ratings from base data
U = Data(:,1);
I = Data(:,2);
T = Data(:,3);
R = Data(:,4);

%Splitting the data into Training and Validation
TE = numel(U);
Train_Split = floor(opts.Tr_Split*TE);
Nu = max(U);
Ni = max(I);
Nt = max(T);

%Creating Rating tensor and Observable data tensor for training the model
Rating_Mat = zeros(Nu,Ni,Nt);   
W = zeros(Nu,Ni,Nt);
ndimI = ndims(Rating_Mat);
oG_dims = size(Rating_Mat);

%Initializing the Training tensor
for i = 1:Train_Split
    Rating_Mat(U(i),I(i),T(i)) = R(i);
    W(U(i),I(i),T(i)) = 1;
end

%% Initializing parameters to build the Laplacian for each tensor dimension
if(strcmp(opts.cT,'YY') && strcmp(opts.dT,'100K'))
    knn_graph = [30 40 1];
elseif(strcmp(opts.cT,'YY') && strcmp(opts.dT,'1M'))
    knn_graph = [30 40 3];
else
    knn_graph = [30 40 4];
end

%%Initializing parameters to store the Trained RS models
rank = opts.Rank;
reg = opts.LReg;
sz = [size(knn_graph,1) opts.iterM size(rank,2) size(reg,2)];
%Factor matrices and normalizing coefficients of the Trained models
A = cell(prod(sz),1);
lambda = cell(prod(sz),1);
%Regularization coefficients for graph Laplacians
alpha = cell(1,ndimI);
%Percentage of Fit of the Trained models
f = cell(prod(sz),1);
%Training NMSE of the Trained models
Train_NMSE = cell(prod(sz),1);

%% Optimization Algorithm - ALS with Conjugate Gradient
%Setting up Laplacian edge weights using Pearson Correlation Coefficient
fun = @(x,sigma) (x/sigma);
param.weight_kernel = fun;

disp('Initialisation done!');
disp('Starting Iterations!');
for t = 1:sz(1)
    disp('Laplacians being built...');
    L = cell(1,ndimI);
    nn_graph = knn_graph(t,:);
    if(opts.dispLap == 1)
        figure("Visible","on");
    else
        figure("Visible","off");
    end
    tiledlayout(3,3,'TileSpacing','Compact','Padding','Compact');
    for i = 1:ndimI    
        clear G iL;
        param.k = nn_graph(i);
        iL = Tensor_nMode(Rating_Mat,i);
        %Building the kNN graph using the GSP toolbox
        G = gsp_nn_graph(iL,param);
        %Building Laplacian matrix
        L{i} = Laplacian(G);
        %Displaying built Laplacian matrices
        if(opts.dispLap == 1)
            nexttile
            %Plotting the n-mode Matricized Tensor
            spy(iL);
            nexttile
            [V, ~] = eigs(L{i}, 2, 'SA');
            [~, p] = sort(V(:,2));
            hold on
            %Plotting the graph Laplacian built on the n-mode Matricized
            %Tensor
            spy(L{i}(p,p));
            nexttile
            %Plotting the Eigen Values of the graph Laplacian
            plot(sort(V(:,2)), '.-');
        end
    end
    for m = 1:sz(2)
        for y = 1:sz(3)
            r = rank(y);
            for z = 1:sz(4)
                alpha(:) = {reg(z)};
                idx = sub2ind(sz,t,m,y,z);
                disp(['NReg: ',num2str(opts.NReg),', Iter: ',num2str(m), ', CP-Rank: ',num2str(y),', LReg: ',num2str(z)]);
                tic
                %CP ALS algorithm to obtained trained model as implemented in Algorithm (4)
                [A{idx},lambda{idx},f{idx},Train_NMSE{idx}] = CP_ALSLap(Rating_Mat,W,L,alpha,r,opts);
                %Storing time taken to train the model
                R_Time(idx) = toc;
            end
        end
    end
end
disp('Ending Iterations!');

%% Testing Dataset
%Creating Rating tensor and Observable data tensor for testing the model
Test_Mat = zeros(oG_dims);
Test_W = zeros(Nu,Ni,Nt);

%Initializing the Testing tensor
for i = Train_Split+1:TE
    Test_Mat(U(i),I(i),T(i)) = R(i);
    Test_W(U(i),I(i),T(i)) = 1; 
end

%Calculating the Testing Error given the Trained model
normX = norm(Test_Mat(:));
for i = 1:prod(sz)
    X = CP_Reconstruct(A{i},lambda{i});
    Test_X = reshape(Test_W(:).*X(:),oG_dims);
    nXbar = norm(Test_X(:));
    normresidual = sqrt(normX^2 + nXbar^2 - 2 * Tensor_Inner(Test_Mat,Test_X));
    Test_NMSE(i) = normresidual / normX;
end


