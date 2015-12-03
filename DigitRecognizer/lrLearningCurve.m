function [error_train, error_cv] = ...
    lrLearningCurve(X, y, X_cv, y_cv, num_labels, lambda)
%   [error_train, error_cv] = ...
%       LRLEARNINGCURVE(X, y, X_cv, y_cv, num_labels, lambda) returns
%		learning curve data for training and cross validation set errors.

	m = size(X, 1);

	error_train = zeros(10, 1);
	error_cv	= zeros(10, 1);


	for i = (m/10):(m/10):m,
	 
		J_train = 0;
		J_cv = 0;

		for j = 1:50,
		
			sel = randperm(size(X,1));
			sel = sel(1:i);
			X_train = X(sel,:);
			y_train = y(sel,:);
			
			all_theta = oneVsAll(X_train, y_train, num_labels, lambda);
				
			for k = 0:(num_labels-1),
				
				theta = all_theta(k+1,:)';
				X_train = [ones(size(X_train,1),1) X_train];
				X_cv = [ones(size(X_cv,1),1) X_cv];
				
				J_train = J_train + (lrCostFunction(theta, X_train, (y_train==k), lambda) / (50*num_labels));
				J_cv = J_cv + (lrCostFunction(theta, X_cv, (y_cv==k),lambda) / (50*num_labels));

				X_train = X_train(:,2:end);
				X_cv = X_cv(:,2:end);		
						
			end;
		end;

		error_train(i/(m/10)) = J_train;
		error_cv(i/(m/10)) = J_cv;
	end;

end
