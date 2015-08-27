%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.

%--------------------------------------
%% STEP 0: Initialize useful parameters
%  Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

% Clear everything
clear ; close all; clc;

inputSize  = 28 * 28;
numLabels  = 5;
hiddenSize = 200;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 400;

%---------------------------------------
%% STEP 1: Load data from MNIST database
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load MNIST database files
mnistData   = loadMNISTImages('train-images.idx3-ubyte');
mnistLabels = loadMNISTLabels('train-labels.idx1-ubyte');

% Set Unlabeled Set (All Images)

% Simulate a Labeled and Unlabeled set
labeledSet   = find(mnistLabels >= 0 & mnistLabels <= 4);
unlabeledSet = find(mnistLabels >= 5);
% We only tries to predict 5 classes from 5 to 9
numClasses = 5;

% Take out indices for train set and test set.
numTrain = round(numel(labeledSet)/2);
trainSet = labeledSet(1:numTrain);
testSet  = labeledSet(numTrain+1:end);

% Create unlabled data using indices of the unlabeled set
unlabeledData = mnistData(:, unlabeledSet);

% Create train data using indices of the train set
trainData   = mnistData(:, trainSet);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5

% Create test data using indices of the test set
testData   = mnistData(:, testSet);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5

% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));

%--------------------------------------
%% STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images.

% Add access to Sparse Autoencoder and Softmax Exercise folder.
addpath '../Sparse Autoencoder/';
addpath '../Sparse Autoencoder/minFunc';
addpath '../Softmax Exercise';

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

% Find opttheta by running the sparse autoencoder on
% unlabeledData

% set opttheta equals to the randomized theta, this will become our trained
% theta.
opttheta = theta;

[cost, grad] = sparseAutoencoderCost(theta, inputSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, unlabeledData);

% Mimize the cost using LBFGS
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

tic;
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, unlabeledData), ...
                              theta, options);
toc;
                       
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network(W1');

%------------------------------------------------------
%% STEP 3: Extract Features from the Supervised Dataset
%  You need to complete the code in feedForwardAutoencoder.m so that the 
%  following command will extract features from the data.

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);

%--------------------------------------
%% STEP 4: Train the softmax classifier
%  Use softmaxTrain.m from the previous exercise to train a multi-class
%  classifier. 

%  Use lambda = 1e-4 for the weight regularization for softmax

softmaxModel = struct; 

% You need to compute softmaxModel using softmaxTrain on trainFeatures and
% trainLabels

% Set the new lambda
lambda = 1e-4;

% Train the newly represented input a2 with the Sparse Autoencoder to get
% seconde-level feature.
options.maxIter = 400;
softmaxModel = softmaxTrain(hiddenSize, numClasses, lambda, ...
                            trainFeatures, trainLabels, options);

%-----------------
%% STEP 5: Testing 
% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel

[pred] = softmaxPredict(softmaxModel, testFeatures);

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

% (note that we shift the labels by 1, so that digit 0 now corresponds to
%  label 1)
%
% Accuracy is the proportion of correctly classified images
% The results for our implementation was:
%
% Accuracy: 98.3%
