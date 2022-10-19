function [trainSet, testSet] = kfcv(x,K,shuffle)  
%% K-Fold Cross Validation
%
% KFCV takes each column as an independent component and creates folds for
% each component in x. It will find the lowest amount of samples that satisfy
% the fold conditions based on the length of the data and desired number of folds.
%
% The data will be truncated if the length of the data is not
% divisible by the desired number of folds, K.
%
% Function:
%       [trainSet, testSet] = kfcv(x,K,shuffle)  
%
% Inputs:
%       Deafult is set to 10-Fold Cross Validation without shuffle,
%       with x as the only input
%
%       x - is a column vector or column matrix
%       K - is the desired amount of folds for training and testing
%       shuffle - can be 'on' or 'off'. When 'on', the Input data is first
%       randomly shuffled, then partitioned.
%
% Outputs:
%       trainSet - is a cell matrix of K by number of colums with the
%       training portion of data from each column
%       testSet - is a cell matrix of K by number of columns with the
%       testing portion of data from each column
%
%
% Examples:
%       If the input is formated as ...
%           x = [1, 2, 3, 4]
%               [5, 6, 7, 7]
%
%       It will need to be transposed to ...
%           x = [1, 5]
%               [2, 6]
%               [3, 7]
%               [4, 8]
%
%       [train, test] = KFCV(x,5,'off');
%           Provides train and test partitions with 5-Folds without randomly
%           shuffled data
%
%       [train, test] = KFCV(x,7,'on');
%           Provides train and test partitions with 7-Folds with randomly
%           shuffled data
%
%       [train, test] = KFCV(x,5);
%           Uses 5-Folds without randomly shuffled data
%
%       [train, test] = KFCV(x,'on');
%           Uses the default 10-Folds with randomly shuffled data
%
%       [train, test] = KFCV(x);
%           Uses the default 10-Folds without randomly shuffled data

%%
if nargin > 3
    error('Too many Input arguments.')
end

switch nargin
    case 1 % defaults to 10-Fold Cross Validation
        fprintf('Default to 10 Folds, No shuffle.\n')
        K = 10;
        shuffle = 'off';
    case 2
        if ischar(K) && strcmp(K,'on') == 1 
            K = 10; % defaults to 10-Fold Cross Validation
            shuffle = 'on';
        elseif ischar(K)
            if strcmp(K,'off') == 1
                K = 10;
                shuffle = 'off';
            else
                error('Input must be an integer, not a %s',class(K))
            end
        end
        shuffle = 'off';
    case 3
        if ischar(x) || ischar(K)
            error('Input must be an integer, not a %s',class(K))
        elseif ~ischar(shuffle) || strcmp(shuffle,'on') ~= 1
            warning('Input shuffle set to ''off'' ')
            shuffle = 'off';
        end
end

        % finds length of data and number of components
        [L,N] = size(x);
        
        % attempts to correct the matrix if the number of components is
        % larger than the length of the data
        if N > L
            warning('Input was transposed because the program detected more components than data samples.')
            x = x';
            [L,N] = size(x);
        end
        
        % catches impossible scenerios
        if K>L
            error('Number of folds, K, cannot be greater than length of data.')
        elseif K<0
            error('Number of folds, K, must be positive.')
        end
        
        % truncates the data if the length of the data is not perfectly divisible by
        % the desired number of fold
        if rem(L,K) > 0
            fprintf('Input was truncated by %d',rem(L,K));fprintf(' data samples.\n')
            x(L-rem(L,K)+1:end,:) = [];
            [L,N] = size(x);
        end
        
        if strcmp(shuffle,'on') == 1
            x = shuffle_input(x);
        end
        
        % points to use in testing set
        testSize = L/K;
        % training & testing matrix creation
        trainSet = cell(K,N);
        testSet = cell(K,N);

    %k-Fold Cross Validation
    for j = 1:N    
        for i = 1:K
            logicSet = false(L,1);
            % logical matrix determines which data to be used in training and
            % testing each iteration, ensuring all points get used.
            logicSet((testSize*(i-1)+1:testSize*i),1) = true;
            testSet{i,j} = x(logicSet(:),j);
            trainSet{i,j} = x(~logicSet(:),j);
        end
    end
    
    % subfunction to shuffle the data if desired
    function shuffled_output = shuffle_input(input)
        [l,n] = size(input);
        shuffled_output = zeros(l,n);
        for k = 1:n
            shuffled_output(:,k) = input((randperm(l,l)'),k); % creates the new order indicies and applys the new random order
        end
    end
end