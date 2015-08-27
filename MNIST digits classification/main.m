% Name: Son Pham
% Classifying MNIST digits using Logistic Regression
% Research with Professor Brian King

% Simply runs this program to classify MNIST digits using Logistic 
% Regression

%% ============ INITIALIZATION ============ %%
clear ; close all; clc;

%% ============ PART 1: LOAD MNIST DATA ============ %%
fprintf('Loading MNIST data...\n');
% These helper functions help load training data and labels into images, 
% labels, testImages, and testLabels
X = loadMNISTImages('train-images.idx3-ubyte');
y = loadMNISTLabels('train-labels.idx1-ubyte');
testX = loadMNISTImages('t10k-images.idx3-ubyte');
testy = loadMNISTLabels('t10k-labels.idx1-ubyte');

% Transpose images and testImages so that each row (instead of column)
% represents one image. Each row will contains darkness value of every
% pixels in the image

X = X'; % becomes 60000 x 784
testX = testX'; % becomes 10000 x 784

%% ============ PART 2: REMOVE THE PADDING PIXELS ============ %%
% Initial data sets has 28 x 28 images. The MNIST digit is only 20 x 20.
% There are additional 4 pixels padding around the digits and can be
% removed.
fprintf('Removing 4-pixel padding borders...\n');
X = removePaddingPixels(X); % X becomes 60000 x 400
testX = removePaddingPixels(testX); % becomes 10000 x 400

%% ============ PART 3: CONSTANTS & VARIABLES ============ %%

% The size of each minibatch
BATCH_SIZE = 20;
% The number of steps used to descent
ITERATIONS = 100000;
% Learning rate
ALPHA = 0.03;

% Weight matrix is a 400 x 10 matrix, each column corresponds to a digit 
% from 0 to 9 and contains 400 weights (of each pixel) associated with that
% digit. (400 because the size of an image is 20 x 20 = 400 pixels)
theta = zeros(400,10);
% fprintf('Initial cost: %d\n',computeCost(X,y,theta)); OFF due to wrong
% function

%% ============ PART 4: STOCHASTIC GRADIENT DESCENT ============ %%
% Train the model using Stochastic Gradient Descent
tic;
[theta, J_history] = stochasticGradientDescent(X, y, theta, ALPHA, ITERATIONS, BATCH_SIZE);
toc; 

%% ============ PART 5: TESTING RESULTS ============ %%
% Predict all the labels for testing set
predictedY = predictY(testX,theta);
% Calculate all predictions that are correct and divide by the size of the
% testing set
accuracy = sum(predictedY == testy) / size(testy,1);
% Show accuracy
fprintf('Accuracy on testing set:\n');
disp(accuracy);
