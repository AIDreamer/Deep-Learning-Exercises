function index = predictDigit(x,theta)
% Predict the digit that input vector x represents
% x = 784 x 1 input vecotr
% theta = 784 x 10 weight matrix

% Calculate probability of each class using softMax function
prob = logisticRegression(x,theta);

% Take the digit that has the highest probability as prediction
[~,index] = max(prob);
end