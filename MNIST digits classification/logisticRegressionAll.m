function allProb = logisticRegressionAll(X,theta)
% Calculate all probability of each digit for every input vector
% X = n x 400 matrix
% theta = 400 x 10 weight matrix
allProb = 1 ./ (1 + exp(-X*theta));
end