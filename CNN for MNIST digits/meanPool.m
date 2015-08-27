function pooledFeatures = meanPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
% 

% Initiate some useful variables
numDigits = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);

% Calculate the size of each pooling patch
resizedDim = floor(convolvedDim / poolDim);

% Initiate a zero matrix of pooledFeatures.
pooledFeatures = zeros(numFeatures, numDigits, resizedDim, resizedDim);

% For each image and for each features. Do pooling.
for imageNum = 1:numDigits
    for featureNum = 1:numFeatures
        for poolRow = 1:resizedDim
            for poolCol = 1:resizedDim
                offsetRow = (poolRow-1)*poolDim;
                offsetCol = (poolCol-1)*poolDim;
                poolPatch = squeeze(convolvedFeatures(featureNum,imageNum,offsetRow+1:offsetRow+poolDim,offsetCol+1:offsetCol+poolDim));
                poolPatch = poolPatch(:);
                meanValue = mean(poolPatch);
                pooledFeatures(featureNum,imageNum,poolRow,poolCol) = meanValue;
            end
        end
    end
end