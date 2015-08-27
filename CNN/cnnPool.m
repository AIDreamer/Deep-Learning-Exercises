function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
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
numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);

% Calculate the size of each pooling patch
resizedDim = floor(convolvedDim / poolDim);

% Initiate a zero matrix of pooledFeatures.
pooledFeatures = zeros(numFeatures, numImages, resizedDim, resizedDim);

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------

for imageNum = 1:numImages
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

end

