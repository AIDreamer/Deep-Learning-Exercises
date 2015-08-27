function prob = logisticRegression(x,theta)
% Calculate probability based on logistic regression function.
% x = 784 x 1 input vector
% theta = 784 x 10 weight matrix

% Calculate probability using logistic regression function
prob = 1 ./ (1 + exp(-(theta' * x)));

end