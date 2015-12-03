function g = sigmoid(z)
%	g = sigmoid(z)
%	computes the sigmoid of z

	g = 1 ./ (1 + exp(-z));
end
