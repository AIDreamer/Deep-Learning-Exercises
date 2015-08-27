function convolvedFeatures = convolve(patchDim, numFeatures, data , W, b)
%convolve Returns the convolution of the features given by W and b with
%the given digit data
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  data - large images to convolve with, matrix in the form
%           data(r, c, digit number)
%  W, b - W, b for features from the sparse autoencoder
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, digitNum, digitRow, digitCol)
numDigits = size(data, 3);
digitDim = size(data, 1);

% Instantiate a zero matrix of convolved features.
convolvedFeatures = zeros(numFeatures, numDigits, digitDim - patchDim + 1, digitDim - patchDim + 1);

for digitNum = 1:numDigits
  for featureNum = 1:numFeatures
      
    % ------------------------
    % convolution of image with feature matrix for each channel
    convolvedImage = zeros(digitDim - patchDim + 1, digitDim - patchDim + 1);
    
    % ------------------------
    % Feature bank
    feature = W(featureNum,:);
    % Obtain the feature (patchDim x patchDim) needed during the convolution
    feature = reshape(feature,patchDim,patchDim);
    % Flip the feature matrix because of the definition of convolution, as explained later
    feature = rot90(squeeze(feature),2);
  
    % ------------------------
    % Obtain the digit
    digit = squeeze(data(:, :, digitNum));

    % ------------------------
    % Convolve "feature" with "digit", adding the result to convolvedImage
    % be sure to do a 'valid' convolution
    convolvedImage = conv2(digit,feature,'valid');
    
    % ------------------------
    % Subtract the bias unit (correcting for the mean subtraction as well)
    % Then, apply the sigmoid function to get the hidden activation

    convolvedImage = convolvedImage + b(featureNum,1);
    convolvedImage = sigmoid(convolvedImage);
    
    % ------------------------
    % The convolved feature is the sum of the convolved values for all channels
    convolvedFeatures(featureNum, digitNum, :, :) = convolvedImage;
  end
end


end

%% SIGMOID FUNCTION -------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
