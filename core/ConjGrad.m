function [x] = ConjGrad(alpha,L,C,x0,opts)
%%Implementation of the linear Conjugate Solver as discussed in Algorithm (2). The function 
%%requires the matrices L = I_n x I_n, C = R x R, x0 = I_n x R with alpha as the graph Laplacian 
%%regularization value.

%%[x] = ConjGrad(alpha,L,C,x0,opts) returns the solution of the system being
%%solved for.

%Initializing the Conjugate Gradient solver
[sz,R] = size(x0);
b = reshape(x0,[],1); 
x = reshape(x0,[],1);
aL = alpha.*L + opts.NReg.*eye(sz);

%Computing the residual (error). Hessian Matrix - Vector multiplication is
%carried out with the initial point.
r = b - Hessian_Vec(aL,C,reshape(x,[sz,R]));
%Initial basis vector deciding the conjugate direction
p = r;

%ConjGrad Iterations
for k = 1:opts.cgmaxIter
    %Previously computed residual is stored
    r0 = r;
    %Hessian Matrix - Vector multiplication carried out with the basis
    %vector
    v = Hessian_Vec(aL,C,reshape(p,[sz,R]));
    %Computing the adjustable step length by performing a line search
    zMz = r0'*r0;
    a = zMz/(p'*v);
    %Updation of factor matrix and residuals
    x = x + a*p;
    r = r0 - a*v;
    %Break if update increment is below tolerance
    if norm(r) < opts.cgtol 
        break; 
    end
    %Updation of conjugate search direction using the Polak - Ribiere
    %method of calculating the search direction
    p = r + max(0,((r'*(r-r0))/zMz))*p;
end
end