function Y = predictY(X,theta)
% Predict all digits from the training examples
% X = 784 x 60000 matrix, each row is a vector input of an image
% theta = 784 x 10 weight matrix

% Calculate the probability of each class of all input vectors using
% softMaxAll function. Each row will now contains the probability of each
% digit for each input vector
allProb = logisticRegressionAll(X,theta);

% Take the digit that has highest probability as prediction
% Simply speaking, for each row, whatever the digit has the highestValue is
% the actual prediction.
[~,Y] = max(allProb,[],2);
% Classes in MATLAB start fgrom 1 to 10, take 1 to represent each digit
% from 0 to 9.
Y = Y - 1;

end