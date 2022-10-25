function [L] = Laplacian(varargin)
%%Function to calculate the Laplacian of a given graph object created using the GSP toolbox.
%%Function requires the graph object as input. Laplacian type can be optionally mentioned, 
%%otherwise the default laplacian type is used.

%%L = Laplacian(G) computes the Laplacian of the graph object 'G' built using the GSP toolbox.

ltype = ' ';
    if nargin < 1
        disp('Error! No arguments passed!');
    else
        G = varargin{1};
        d = sum(G.W,2);
        if nargin < 2
            ltype = 'combinatorial';
        else
            ltype = 'normalized';
        end
    end
    switch ltype
    case 'combinatorial'
        L = diag(d)-G.W;
    case 'normalized'
        D = diag(d.^(0.5));
        L = speye(G.N) - D*G.W*D;
    otherwise
        disp('Unknown Laplacian type');
    end
end