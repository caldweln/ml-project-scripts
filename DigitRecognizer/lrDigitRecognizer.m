
%% Initialization
clear ; close all; clc


% Load Training Data
fprintf('Loading Training Data ...\n');
fflush(stdout);
data = csvread('./train.csv');

% Clean & Organize Training Data
data = reshape(data(2:end,:),size(data,1)-1, size(data,2)); % remove headers

i_60 = length(data)*6/10;
i_80 = length(data)*8/10;

y_train = data(1:i_60,1);		% 60%
y_cv = data(i_60+1:i_80,1);		% 20%
y_test = data(i_80+1:end,1);	% 20%

X_train = data(1:i_60,2:end);		% 60%
X_cv = data(i_60+1:i_80,2:end);		% 20%
X_test = data(i_80+1:end,2:end);	% 20%

fprintf('\nNext: Training\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

% Training - choose theta to minimize cost over training data
fprintf('\nTraining One-vs-All Logistic Regression... \n');
fflush(stdout);

num_labels = 10;
lambda = 0.001;

all_theta = oneVsAll(X_train, y_train, num_labels, lambda);

pred_train = predictOneVsAll(all_theta, X_train);
accuracy_train = mean(double(pred_train == y_train)) * 100
pred_cv = predictOneVsAll(all_theta, X_cv);
accuracy_cv = mean(double(pred_cv == y_cv)) * 100

fprintf('\nNext: Diagnostics\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

% Diagnostics - choose polynomial/regularization to minimize cost over cross-validation data
fprintf('\nDiagnostic Learning Curves... \n');
fflush(stdout);

[error_train error_cv] = lrLearningCurve(X_train, y_train, X_cv, y_cv, num_labels, lambda);

m = size(X_train,1);
plot((m/10):(m/10):m, error_train, (m/10):(m/10):m,  error_cv);
title('Learning curve for logistic regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')

% polyFeatures - add to address high bias
% featureNormalize - if skewed, quicker convergence
% lambda - decrease/increase regularization to address high bias/variance

fprintf('\nNext: Testing\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

% Testing - Estimate generalization error on test set
fprintf('\nTesting Performance On Test Data... \n');
fflush(stdout);

pred_test = predictOneVsAll(all_theta, X_test);
accuracy_test = mean(double(pred_test == y_test)) * 100

