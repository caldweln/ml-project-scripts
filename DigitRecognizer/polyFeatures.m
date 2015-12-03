function [X_poly] = polyFeatures(X, p)
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x n) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%
	[m n] = size(X);
	X_poly = zeros(m, n*p);
	X_poly(:,1:n) = X;

	for i = 2:p,
		X_poly(:,n*(i-1)+1:n*i) = X_poly(:,n*(i-2)+1:n*(i-1)) .* X_poly(:,1:n);
	end;


end
