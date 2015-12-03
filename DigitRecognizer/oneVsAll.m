function all_theta = oneVsAll(X, y, num_labels, lambda)
%	all_theta = oneVsAll(X, y, num_labels, lambda)
%	trains logistic regression classifiers for each label

	[m n] = size(X);
	
	all_theta = zeros(num_labels,n+1);
	X = [ones(m,1) X];

	options = optimset('GradObj', 'on', 'MaxIter', 5);
	init_theta = zeros(n+1, 1);

	for k=0:(num_labels-1),
		theta = fminunc(@(theta)lrCostFunction(theta, X, (y==k), lambda), init_theta, options);
		all_theta(k+1,:) = theta';
	end


end
