function [Out] = Tensor_Inner(X,Y)
%%Implementation of the tensor inner product which returns the elementwise product 
%%of two same sized (dimension) tensors. The function requires two tensors to be 
%%given as input.

%%[Out] = Tensor_Ex2Inner(X,Y) computes the inner product of X and Y
    if (~all(size(X) == size(Y)))
        error('Tensors are not of same dimensions!')
    else
        X = reshape(X, [1, numel(X)]);
        Y = reshape(Y, [numel(Y),1]);
        Out = X*Y;
    end
end