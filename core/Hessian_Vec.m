function [M] = Hessian_Vec(aL,C,x)
%%Implementation of the Hessian Matrix - Vector multiplication as discussed in Algorithm
%%(3). The function requires the matrices aL = I_n x I_n, C = R x R, x =
%%I_n x R.

%%[M] = Hessian_Vec(aL,C,x) returns the Hessian Matrix - Vector multiplication of the system 
%%being solved for.
    M = reshape(aL*x + x*C,[],1);
end