function [Out] = Tensor_MulnMode(varargin)
%%Implementation of Mode-n product of a given tensor and vector/matrix. The function requires a 
%%tensor X, factor matrices (as cell inputs) Y and the nth-mode 'k' that the tensor is going to be 
%%matricized as to be given as inputs. The given mode must be less than or equal to the order of 
%%the tensor.

%%[Xy] = Tensor_MulnMode(X,Y,R,k), where X -> arbitrary tensor, Y -> vector/matrix cell, 
%%R -> chosen CP rank, k -> mode
    X = varargin{1};
    A = varargin{2};
    R = varargin{3};
    k = varargin{4};
    N = ndims(X);
    if k ~= 1
        m_sz = prod(size(X,1:k-1));
    end
    if k ~= N
        r_sz = prod(size(X,k+1:N));
    end
    n_sz = size(X,k);
    if k == 1
        Ar = mKr(A{2:N});
        Y = reshape(X,n_sz,r_sz);
        Out =  Y * Ar;
    elseif k == N
        A1 = mKr(A{1:N-1});
        Y = reshape(X,m_sz,n_sz);
        Out = Y' * A1;
    else
        Al = mKr(A{k+1:N});
        Ar = reshape(mKr(A{1:k-1}),m_sz,1, R);
        Y = reshape(X,[],r_sz);
        Y = Y * Al;
        Y = reshape(Y,m_sz,n_sz,R);
        Out = zeros(n_sz,R);
        for r =1:R
            Out(:,r) = Y(:,:,r)'*Ar(:,:,r);
        end
    end
end
