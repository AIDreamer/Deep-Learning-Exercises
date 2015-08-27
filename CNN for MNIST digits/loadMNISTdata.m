function loadMNISTdata()
% This function will load MNIST data and process them into suitable
% formats. These processing include:
% 1. Remove 4-pixel padding borders
% 2. Changing label 0 to 10 since MNIST labels start from 1.

fprintf('-- Loading MNIST data...\n');
% These helper functions help load training data and labels into images, 
% labels, testImages, and testLabels
X = loadMNISTImages('train-images.idx3-ubyte'); % 784 x 60000
y = loadMNISTLabels('train-labels.idx1-ubyte');
y(y == 0) = 10; % Remap 0 to 10

testX = loadMNISTImages('t10k-images.idx3-ubyte'); % 784 x 10000
testy = loadMNISTLabels('t10k-labels.idx1-ubyte');
testy(testy == 0) = 10; % Remap 0 to 10

% Original MNIST datas are 28 x 28 images with 4 padding pixels as border.
% We can remove them to increase the speed of training.
fprintf('-- Remove 4-pixel padding borders...\n');
X = removePaddingPixels(X);
testX = removePaddingPixels(testX);

% Transform 2D to 3D data
%  Original data is 400 x 60000
%  Transform the data into 20 x 20 x 60000
%  We need to transform so that each image is also a matrix, which is
%  better for sampling purpose.

% digit dim
digitDim = 20;

% reshaping the matrix from 2D to 3D based on the dimension of the digits
[~,c] = size(X);
transformedX = permute(reshape(X',[c,digitDim,digitDim]),[2,3,1]);

[~,c] = size(testX);
transformedTestX = permute(reshape(testX',[c,digitDim,digitDim]),[2,3,1]);

% Save the MNIST data for later use
save('MNISTdata.mat', 'X', 'y', 'testX', 'testy', 'transformedX', 'transformedTestX');

end