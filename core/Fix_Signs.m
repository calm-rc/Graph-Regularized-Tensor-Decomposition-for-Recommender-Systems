function [A] = Fix_Signs(A,lambda)
%%Implementation of Fix_Signs for the factor matrices after minimization. This prevents the 
%%factor matrices from having negative values. The function requires the factor matrices 'A' 
%%(given as a cell input) and normalizing coefficient lambda.

%%[A] = Fix_Signs(A,lambda) makes the negative entries of the factor matrices that
%%positive.

    R = length(lambda);
    for r = 1 : R
        for n = 1:numel(A)
            [~,idx(n)] = max(abs(A{n}(:,r)));    
            sgn(n) = sign(A{n}(idx(n),r));
        end

        negidx = find(sgn == -1);
        negate = 2 * floor(numel(negidx)/2);

        for i = 1:negate
            n = negidx(i);
            A{n}(:,r) =  -A{n}(:,r);
        end
    end
end