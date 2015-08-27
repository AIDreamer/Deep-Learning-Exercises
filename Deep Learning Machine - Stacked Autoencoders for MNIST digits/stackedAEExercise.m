%% Stacked autoencoders to classify MNIST digits
%  Original code developed by Andrew Ng (?) at 
%  http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial
%  Modified by Son Pham
%
%  This file tries to build a deep learning machine using stacked
%  autoencoders to classify MNIST digits.
%  Training data: 784 x 60,000
%  Testing data:  784 x 10,000

%---------------------------------
%% STEP 0: Initialize useful value
%  In this step we will try to initialize some useful variables for the
%  deep learning machine.

% Clear everything
clear ; close all; clc;

% Initialize some useful values.
inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (also called sparsity parameter)
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term

% Note: THere's noo need for learning rate alpha since this method use
% L-FBGS to find optima.

%---------------------------------------
%% STEP 1: Load data from MNIST database
%  Loads our training and testing data from MNIST database file.

% Get MNIST training images
trainData = loadMNISTImages('mnist/train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');
trainLabels(trainLabels == 0) = 10; % Remap 0 to 10

% Get MNIST labelled test images
% Note that we apply the same kind of preprocessing as the training set
testData = loadMNISTImages('mnist/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
testLabels(testLabels == 0) = 10; % Remap 0 to 10

% Note: We remap 0 to 10 because MATLAB matrix indices start a 1

%--------------------------------------------------------------
%% STEP 2: Train the first hidden layer with sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled training
%  images.

% Add Sparse Auto Encoder and Softmax folders into the file
addpath '../Sparse Autoencoder/';
addpath '../Sparse Autoencoder/minFunc/';
addpath '../Softmax Exercise/';

% Randomly initialize the parameters of sae1Theta
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

% Set sae1OptTheta equals to the initialialized sae1Theta, this will become
% our trained parameters.
sae1OptTheta = sae1Theta;

options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

% Train sae1OptTheta using Sparse Autoencoder.
tic;
[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, trainData), ...
                              sae1OptTheta, options);
toc;

%---------------------------------------------------------------
%% STEP 3: Train the second hidden layer with sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.

% This part re-represent the original input as features found in
% sae1OptTheta
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

% Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

% Set sae2OptTheta equals to the initialialized sae2Theta
sae2OptTheta = sae2Theta;

% Train sae2OptTheta using Sparse Autoencoder.
tic;
[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, sae1Features), ...
                              sae2OptTheta, options);
toc;
%--------------------------------------
%% STEP 4: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.

% This part re-represent the original input as features found in
% sae2OptTheta
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

% Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);

% Train the softmax classifier, the classifier takes in input of dimension 
% "hiddenSizeL2" corresponding to the hidden layer size of the 2nd layer.
% You should store the optimal parameters in saeSoftmaxOptTheta

softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, lambda, ...
                            sae2Features, trainLabels, options);
                        
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

%-------------------------------------
%% STEP 5: Fine-tune the whole network

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
% StackedAEtheta is the weights of the softmax concantenated with all
% weights of two other hidden layers.
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

% Gradient checking
% checkStackedAECost;

stackedAEOptTheta = stackedAETheta;

tic;
[stackedAEOptTheta, cost] = minFunc( @(p)stackedAECost(p, inputSize, hiddenSizeL2, ...
                                              numClasses, netconfig, ...
                                              lambda, trainData, trainLabels), ...
                              stackedAEOptTheta, options);
toc;

%%-------------
%% STEP 6: Test 
% Using the new trained system to predict the digits in the testing set

% Predict with weights before fine-tuning (StackedAETheta)
[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Predict with weights after fine-tuning (StackedAETheta)
[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% The results I got were
% Before Finetuning Test Accuracy: 87.74%
% After Finetuning Test Accuracy:  97.59%

%-------------
%% Store theta
% Store the trained weights

% Save the weights of the stackedAEOptTheta into a file.
save('trainedWeight.dat', 'stackedAEOptTheta', '-ASCII');

%% Visualize weights.
%  Visualizing the weights.
% Store the weights of two hidden layers
hiddenWeights1 = stack{1}.w';
hiddenWeights2 = stack{2}.w';

hiddenVisual2 = zeros(size(hiddenWeights1));
for i=1:200
   for j=1:200
        hiddenVisual2(:,i) = hiddenVisual2(:,i) + hiddenWeights1(:,j) * hiddenWeights2(j,i);
   end
end

% Display the visualization of the second layer
figure(1);
display_network(hiddenWeights1, 60);
figure(2);
display_network(hiddenVisual2);
