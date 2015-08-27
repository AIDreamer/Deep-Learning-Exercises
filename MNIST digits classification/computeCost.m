function J = computeCost(X,y,theta)
% This function computes the cost of the model using the negative of
% log-likelihood equation.

% Calculate the probability of each digit of every training example
allProb = logisticRegression(X,theta);
% Plus 1 to all correct labels so that they can be accurately indexed
% (indices in MATLAB run from 1 to 10, while digits from 0 to 9)
indices = y + 1;

% Take out all the probability of the supposedly correct answer.
% Supposedly, we want these probabilities of correct classes to be high
correctProb = allProb(sub2ind(size(allProb),(1:length(indices))',indices));

% Cost is the negative of log-likelihood all all correct probabilities
J = 1/ (2* size(X,1)) * (correctProb - 1) .^ 2;

end
