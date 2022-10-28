function [A,lambda,f,Train_NMSE] = CP_ALSLap(T,W,L,alpha,R,opts)
%%Implementation of the CP ALS tensor decomposition as discussed in Algorithm (4). The function 
%%requires the rating and observable space tensors 'T' and 'W,' respectively. Matrices L = I_n x 
%%I_n with alpha as the graph Laplacian regularization value. R is the chosen CP rank of the model.

%%[A,lambda,f,Train_NMSE] = CP_ALS(T,W,L,alpha,R,opts) returns the factor matrices 'A' and 
%%normalizing coefficient lambda. The function also returns the Percentage of Fit (f) and model
%%training NMSE (Train_NMSE).

%Initializing the ALS Algorithm
printitn = opts.printin;
d = ndims(T);
oG_dims = size(T);
normX = norm(T(:));
Enew = 0;

if(opts.type == "gpuArray")
    X = gpuArray(reshape(W(:).*T(:),oG_dims));
else
    X = reshape(W(:).*T(:),oG_dims);
end

%Initializing the factor matrices for model training
gA = cell(d,1);
AtA = zeros(R,R,d,opts.type);
for n = 2:d
    Q = Tensor_nMode(X,n);
    if(size(Q,1) < R)
        gA{n} = max(Q(:))*rand(size(Q,1),R,opts.type);
    else
        [gA{n},~,~] = svds(Q,R,'L');
    end
end

for n = 1:d
    if ~isempty(gA{n})
        AtA(:,:,n) = gA{n}'*gA{n};
    end
    if opts.type == "gpuArray"
        gA{n} = gpuArray(gA{n});
        gL{n} = gpuArray(L{n});
    else
        gA{n} = single(gA{n});
        gL{n} = L{n};
    end
end

%CP-ALS Iterations
for iter = 1:opts.maxALSIter
    Eold = Enew;
    % Iterate over all d modes of the tensor
    for n = 1:d
        idx = [1:n-1 n+1:d];
        %           M = gA(idx); U = mKr(M{1:end});

        %Khatri-rao product of all factor matrices except the nth one.
        Y = prod(AtA(:,:,idx),3);

        %Tensor nMode Product between Rating tensor and factor matrices
        Anew = Tensor_MulnMode(X,gA,R,n);

        %Conjugate Gradient Solver
        Anew = ConjGrad(alpha{n},gL{n},Y,Anew,opts);
        Anew = reshape(Anew,[],R);

        %Normalize each factor matrix to prevent singularities
        if iter == 1
            glambda = sqrt(sum(Anew.^2,1))';        %2-norm
        else
            glambda = max(max(abs(Anew),[],1),1)';  %max-norm
        end
        Anew = bsxfun(@rdivide, Anew, glambda');
        gA{n} = Anew;
        AtA(:,:,n) = gA{n}'*gA{n};
    end

    %Computing the Training NMSE and Percentage of Fit (POF)
    Xbar = CP_Reconstruct(gA,glambda);
    %Ratings being discrete (0-5), need to be rounded to the nearest integer.
    Xbar = round(Xbar);
    
    Train_X = reshape(W(:).*Xbar(:),oG_dims);
    nXbar = norm(Xbar(:));
    nXtrain = norm(Train_X(:));

    if normX == 0
        %Training NMSE
        gTrain_NMSE(iter) = nXtrain^2 - 2*Tensor_Inner(X,Train_X);
        %Percentage of Fitness
        Enew = nXbar^2 - 2 * Tensor_Inner(X,Xbar);
    else
        normresidual = sqrt(normX^2 + nXbar^2 - 2 * Tensor_Inner(X,Xbar));
        normresidualT = sqrt(normX^2 + nXtrain^2 - 2 * Tensor_Inner(X,Train_X));
        %Training NMSE
        gTrain_NMSE(iter) = normresidualT / normX;
        %Percentage of Fitness
        Enew = 1 - (normresidual / normX);
    end
    fitchange = abs(Eold - Enew);
    gf(iter) = Enew;
    %Checking for convergence of algorithm
    if (mod(iter,printitn)==0) || ((printitn>0) && (flag==0))
        fprintf(' Iter %2d: pfit = %e f-delta = %7.1e\n', iter, Enew, fitchange);
    end

    if (iter > 1) && (fitchange < opts.tolALS)
        break;
    end
end

%Trained model after convergence
gA = Fix_Signs(gA,glambda);
if(opts.type == "gpuArray")
    A = gather(gA);
    lambda = gather(glambda);
    f = gather(gf);
    Train_NMSE = gather(gTrain_NMSE);
else
    A = gA;
    lambda = glambda;
    f = gf;
    Train_NMSE = gTrain_NMSE;
end
end
