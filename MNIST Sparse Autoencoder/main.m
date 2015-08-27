%% Extracting feature from MNIST digits using sparse autoencoder. 
%  Originally written by Andrew Ng (?) for CS249A/CS249W class at Stanford
%  Modified by: Son Pham
%  The original code can be found here
%  http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
% 
%  This file will use sparse auto encoder to extract meaningful features
%  from MNIST digits set.

%% ===== PART 0: INITIALIZATION ===== %%
%  Clear everything off the screne
clear ; close all; clc;

%% ===== PART 1: INITIALIZE PARAMETERS ===== %%
%  Initialize some useful parameters for the training.

visibleSize = 20*20;   % number of input units 
hiddenSize = 100;     % number of hidden units 
sparsityParam = 0.01; % sparsity parameter or desired average activation of the hidden units.
% Note: sparsityParam is commonly written as rho

lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term

% Initialize theta with randomized weights (for the purpose of 'Symmetry Breaking')
theta = initializeParameters(hiddenSize, visibleSize);

%% ===== PART 1: LOAD MNIST DATA ===== %%
fprintf('Loading MNIST data...\n');
% These helper functions help load training data and labels into images, 
% labels, testImages, and testLabels
X = loadMNISTImages('train-images.idx3-ubyte'); % 784 x 60000
y = loadMNISTLabels('train-labels.idx1-ubyte');
testX = loadMNISTImages('t10k-images.idx3-ubyte'); % 784 x 10000
testy = loadMNISTLabels('t10k-labels.idx1-ubyte');

% Original MNIST datas are 28 x 28 images with 4 padding pixels as border.
% We can remove them to increase the speed of training.
fprintf('Remove 4-pixel padding borders...\n');
X = removePaddingPixels(X);
testX = removePaddingPixels(testX);

% Display sample images (to make sure we get correct MNIST digits.
figure(1);
display_network(X(:,randi(size(X,2),200,1)),8);

%% ===== PART 3: CALCULATING COST AND GRADIENT CHECKING ===== %%
%  Use sparseAutoencoderCost to compute the cost and the gradient.
%  This part is used to check the sparseAutoencoderCost and is not needed.
[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, X);
                                 
%% ===== PART 4: GRADIENT CHECKING ===== %%
%  This part makes sure the gradient is accurately computed by comparing
%  analytical gradient with numerical gradient.

% SON'S NOTE: I turn this off to speed up the training process. There's no
% need to check if we know for sure that the Gradient is correctly computed.

% numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
                                                  % hiddenSize, lambda, ...
                                                  % sparsityParam, beta, ...
                                                  % patches), theta);

% Use this to visually compare the gradients side by side
% disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
% diff = norm(numgrad-grad)/norm(numgrad+grad);
% disp(diff); % Should be very very small. THat means the analytical
% gradient is very close to the numerical gradient and, thus, accurately
% computed.

%% ===== PART 5: TRAINING THE AUTOENCODER ===== %%
%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. 
                          
% Generally, for minFunc to work, you need a function pointer with two 
% outputs: the function value and the gradient. In our problem, 
% sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

tic;
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, X), ...
                              theta, options);
toc;
%%======================================================================
%% ===== PART 6: VISUALIZATION ===== %%

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
figure(2);
title('Visualizing hidden units');
display_network(W1', 12);

print -djpeg weights.jpg   % save the visualization to a file 


