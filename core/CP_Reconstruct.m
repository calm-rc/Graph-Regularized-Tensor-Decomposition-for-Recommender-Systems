function [X_hat] = CP_Reconstruct(A,lambda)
%%Reconstruction of a tensor having used a CP Tensor decomposition method. The function requires 
%%the obtained factor matrices (cell input) and corresponding normalizing coefficient lambda.

%%X_hat = CP_Reconstruct(A,lambda) returns the reconstructed tensor X_hat using
%%its factor matrices 'A' (given as a cell input) and normalizing coefficient 'lambda'.
    n_fac = size(A,1);
    A{end} = full(double(A{end}) * spdiags(lambda,0,length(lambda),length(lambda)));
    A{end} = single(A{end});
    for i=1:n_fac
        idx(i) = size(A{i},1);
    end
    X_hat = A{1}*(mKr(A{2:end})');
    X_hat = reshape(X_hat,idx);
end