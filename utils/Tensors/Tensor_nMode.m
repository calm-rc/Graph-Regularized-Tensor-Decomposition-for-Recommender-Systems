function [Xn] = Tensor_nMode(X,k)
%%Implementation of Mode-n matricization of a given tensor. The function requires a 
%%tensor X and the nth-mode 'k' it is to be matricized as to be given as inputs. The given 
%%mode must be less than or equal to the order of the tensor.

%%[Xn] = Tensor_nMode(X,k), where X -> arbitrary tensor, k -> mode
    if (nargin < 2)
        if(nargin == 0)
            error('No arguments passed! Need a tensor and its mode!')
        else
            error('Only one tensor passed! Need the matricization mode!');
        end
    else
        dim = size(X);
        if (length(dim) < k || k < 1)
            error('nMode is not within the order range of tensor X');
        end
        Xn = shiftdim(X,k-1);
        Xn = reshape(Xn,[dim(k), prod(dim(:))/dim(k)]);
    end
end