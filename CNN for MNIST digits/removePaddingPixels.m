function X = removePaddingPixels(X)
% This function removes the 4 padding pixels around each image in input
% matrix X.
% X = 784 x n input vector
% return
% X = 400 x n input vector (only 20 x 20 image in the center is retained)

% Remove the first and last 4 rows of pixels (first and last 112 pixels)
X([1:112,673:784],:) = [];

% Repeatedly removing 8 padding pixels in each row.
for i = 0:19
    c = i * 20;
    X([(c+1):(c+4),(c+25):(c+28)],:) = [];
end

end