function p = predictOneVsAll(all_theta, X)
%	p = predictOneVsAll(all_theta, X)
%	predict class for each X row given theta param for each class

	m = size(X,1);
	p = zeros(m,1);

	X = [ones(m,1) X];

	[v,p] = max(sigmoid(X * all_theta'),[],2);
	p = p .- 1;
	
end
