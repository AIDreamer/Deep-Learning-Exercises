function detectLevel1Features(transformedX)
% This function detect level1 features from input X. These features will
% be saved in level1Features, which will in turn be used to be convolved
% with the current input.

%% STEP 0: INITIALIZATION
%  Here we initialize some useful parameters and add some useful path.
fprintf('-- Initializing\n');
% Add some useful path
addpath '../MNIST Sparse Autoencoder';
addpath 'minFunc';

% Initalize some useful variables for collecting data
numpatches = 50000;
patchDim = 5;
imagesize = 20; % 20 is the size of each digits
numimages = size(transformedX,3);

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 50000 columns. 
patches = zeros(patchDim*patchDim, numpatches);

%% STEP 3: TAKE RANDOM PATCH FROM THE DATA SET.
fprintf('-- Sampling random %d patches of size %d x %d from data...\n', numpatches, patchDim, patchDim);

for i = 1:numpatches % for each patch of image
   % randomly pick the first (x,y) position for each patch
   randX = randi(imagesize - patchDim + 1);
   randY = randi(imagesize - patchDim + 1);
   % Sample the patch out of IMAGES
   samplepatch = transformedX(randX : randX+patchDim-1, randY : randY+patchDim-1, randi(numimages));
   % Assign the sample to variable patches
   patches(:,i) = samplepatch(:);
end

% Display sample images (to make sure we get correct MNIST digits.
figure(1);
display_network(patches(:,randi(size(patches,2),200,1)),9);

%% STEP 3: DETECT FEATURES (FILTERS)
%  sparseAutoencoderCost is correct, You can start training your sparse
%  autoencoder with minFunc (L-BFGS).

fprintf('-- Detecting features of size %d x %d...\n', patchDim, patchDim);

%  Initialize some useful parameters for the training.
visibleSize = patchDim * patchDim;   % number of input units
hiddenSize = 25;     % number of hidden units 
sparsityParam = 0.03; % sparsity parameter or desired average activation of the hidden units.
% Note: sparsityParam is commonly written as rho

lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term
%  Randomly initialize the parameters (for the purpose of 'Symmetry
%  Breaking')
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 1000;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'off';

tic;
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);
toc;

%% ===== STEP 4: VISUALIZATION AND SAVE THE FEATURES
fprintf('-- Saving and visualizing features detected...\n');
% Save the features to be used for convolution
save('level1Features.mat', 'opttheta', 'transformedX');