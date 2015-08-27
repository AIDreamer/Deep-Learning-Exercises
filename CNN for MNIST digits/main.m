%% CONVOLUTIONAL NEURAL NETWORK FOR MNIST DIGITS

%  A software to build a convolutional neural network to recognize
%  handwritten digits. There are 3 steps.
%  
%  1. Detect features using sparse autoencoders
%  
%  2. Use such features to convolve and pool new information. The new
%  information is a re-representation of the old input in a new and more
%  meaningful way.
%  
%  3. After having the new kind of input, the network will put it into a 
%  softmax classification and then train the network to recognize MNIST 
%  digits.

%% PART 0: INITIALIZATION
%  Clear everything off the screne
clear ; close all; clc;
fprintf('PART 0: INITIALIZATION\n');

%  Add some useful path
addpath '..\Softmax Exercise';

%  Initialize some useful parameters for the training.
patchDim = 5;
visibleSize = patchDim * patchDim;   % number of input units
hiddenSize = 25;     % number of hidden units 
sparsityParam = 0.03; % sparsity parameter or desired average activation of the hidden units.
% Note: sparsityParam is commonly written as rho

lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term
%  Randomly initialize the parameters (for the purpose of 'Symmetry
%  Breaking')

%% PART 1: LOAD MNIST DATA
%  Load the MNIST data
fprintf('PART 1: LOADING MNIST DATA\n');
loadMNISTdata();
load MNISTdata.mat

%% PART 2: DETECT LEVEL 1 FEATURES
%  Detect Level 1 Features
fprintf('PART 2: DETECT LEVEL 1 FEATURES\n');
detectLevel1Features(transformedX);
load level1Features.mat;

% Convert given optTheta into W and b for convolution
W = reshape(opttheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

fprintf('-- Visualizing features detected...\n');
figure(2);
display_network(W');

%% PART 3: CONVOLVING FEATURES
fprintf('PART 3: CONVOLVE THE FEATURES\n');

fprintf('-- Convolving train features...\n');
tic;
convolvedTrainFeatures = convolve(patchDim, hiddenSize, transformedX, W, b);
fprintf('-- Convolving test features...\n');
convolvedTestFeatures = convolve(patchDim, hiddenSize, transformedTestX, W, b);
toc;

%% PART 4: POOLING FEATURES
fprintf('PART 4: POOL THE FEATURES\n');
% Determine pooling size
poolDim = 4; % Reduce each side by 4 times
% Pool train features
fprintf('-- Pooling train features, shrinking each side of representation by %d times. This should take about 15 minutes...\n', poolDim);
tic;
pooledTrainFeatures = meanPool(poolDim, convolvedTrainFeatures);
toc;

% Pool test features
fprintf('-- Pooling test features...\n', poolDim);
tic;
pooledTestFeatures = meanPool(poolDim, convolvedTestFeatures);
toc;

% Save pooled features
fprintf('-- Saving pooled features...\n');
save('pooledFeatures.mat', 'pooledTrainFeatures', 'pooledTestFeatures');

%% STEP 5: TRAIN SOFTMAX CLASSIFIER
%  Now, you will use your pooled features to train a softmax classifier,
%  using softmaxTrain from the softmax exercise.
%  Training the softmax classifer for 1000 iterations should take less than
%  10 minutes.

% Add the path to your softmax solution, if necessary
addpath /path/to/solution/

fprintf('PART 5: TRAINING THE SOFTMAX CLASSIFIER\n');
% Initialize some useful variables
numTrainDigits = size(transformedX,3);
% Setup parameters for softmax
softmaxLambda = 1e-4;
numClasses = 10;
% Reshape the pooledFeatures to form an input vector for softmax
softmaxX = permute(pooledTrainFeatures, [1 3 4 2]);
softmaxX = reshape(softmaxX, numel(pooledTrainFeatures) / numTrainDigits,...
    numTrainDigits);
softmaxY = y;

options = struct;
options.maxIter = 1000;
options.display = 'off';
fprintf('-- Training...\n');
softmaxModel = softmaxTrain(numel(pooledTrainFeatures) / numTrainDigits,...
    numClasses, softmaxLambda, softmaxX, softmaxY, options);

%% STEP 6: TEST THE SOFTMAX MODEL
% test the sfotmax model for accuracy
fprintf('PART 5: TESTING THE MODEL FOR ACCURACY\n');
numTestDigits = size(testy,1);

testX = permute(pooledTestFeatures, [1 3 4 2]);
testX = reshape(testX, numel(pooledTestFeatures) / numTestDigits, numTestDigits);

[pred] = softmaxPredict(softmaxModel, testX);
acc = (pred(:) == testy(:));
acc = sum(acc) / size(acc, 1);
fprintf('-- Accuracy: %2.3f%%\n', acc * 100);