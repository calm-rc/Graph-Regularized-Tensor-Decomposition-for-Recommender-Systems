function [Out] = mKr(varargin)
%%Implementation of Katri-rao product of matrices. The function requires matrices (as cell inputs)
%%with each matrix cell having same number of columns.

%%[Out] = mKr(A), where A -> cell input of matrices
    A = varargin;
    Order = length(A):-1:1;
    N = size(A{1},2);
    Out = A{Order(1)};
    for i = Order(2:end)
        Out = bsxfun(@times, reshape(A{i},[],1,N),reshape(Out,1,[],N));
    end
    Out = reshape(Out,[],N);
end