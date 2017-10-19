function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);

t1 = theta(1);

shift_theta = theta(2:size(theta));

theta_reg = [0;shift_theta];

%theta(1,:) = []; 	%Exclude theta(1)
					%Can also do: theta(2:length(theta))
					% ^Selects all columns except the first one 	



J = ((-y)'*log(h)-(1-y)'*log(1-h))/m + ((lambda/(2*m)) * (theta_reg'*theta_reg));

grad = (1/m)*(X'*(h-y)+lambda*theta_reg);

%	ALTERNATE METHOD
%grad1 = (x1'*(h - y))/m
%grad2 = (theta' * (X'*(h - y))/m + ((lambda/m))*theta)
%grad = [grad1 ; grad2]

% =============================================================

end
