function [J, grad] = lrCostFunction(theta, X, y, lambda)
%	J = lrCostFunction(theta, X, y, lambda)
%	computes the cost of using theta for regularized logistic regularizaion
%	and the gradient of the cost w.r.t the parameters

	if(nargin < 4)
		lambda = 0;
	endif
	
	m = size(y,1);
	J =  0;
	grad = zeros(size(theta));

	% cost
	h = sigmoid(X * theta);
	J = (-1/m) * (y' * log(h) + (1 .- y)' * log(1 .- h));

	% regularization
	theta_unbias = theta;
	theta_unbias(1,:) = 0;
	J = J + ((lambda / (2 * m)) * (theta_unbias' * theta_unbias));

	% grad
	grad = (1/m) * X' * (h - y);

	% regularization
	grad = grad + (lambda / m) .* theta_unbias;

	grad = grad(:);

end
