function [theta, J_history] = stochasticGradientDescent(X, y, theta, ...
    alpha, num_iters, batch_size)
% Perform stochastic gradient descent by num_iters step to learn theta.
% Because the number of inputs is too large, SGD takes a batch_size of
% random training examples to perform each step instead.
% X = m * 400 input matrix
% Y = m-dimensional vector represengint labels
% theta = 400 x 10 weight matrix
% alpha = learning_rate
% num_iters = number of steps the algorithm will take
% batch_size = the size of random batch for each descending step.

% Initialize some useful behavior
J_history = zeros(num_iters, 1); % To record cost after each iteration
change = zeros(size(theta));
m = size(X,2);

for iter = 1:num_iters
    % Take a batch_size of random training examples from the training set.
    batch_index = randi(size(X,1),[batch_size,1]);
    % Create training_batch and label_batch
    training_batch = X(batch_index,:);
    label_batch = y(batch_index);
    % Calculate all probability of the training_batch, for later use.
    allProb = logisticRegressionAll(training_batch,theta);
    
    % Perform one step of Gradient Descent in the chosen training_batch
    for i = 1:size(theta,2)
        change(:,i) = alpha * 1 / batch_size * ...
            (training_batch' * (allProb(:,i) - (label_batch==repmat(i-1,batch_size,1))));
    end
    theta = theta - change;
    
    % Calculate J_History to plot and keep track if SGD works correctly
    % J_history(iter) = computeCost(X,y,theta); OFF due to wrong function.
    
    % Show training progress
    disp(sprintf('Training... %d / %d done', iter, num_iters));
end
end