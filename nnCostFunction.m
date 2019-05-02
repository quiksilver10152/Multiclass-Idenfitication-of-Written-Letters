function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(m,1)  X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
dTot3 = zeros(num_labels,size(Theta2,2)); %10x26
dTot2 = zeros(size(Theta1),size(X,2)); %25x401
ynew = eye(num_labels)(y,:); %5000x10
onnes = ((ones(m,num_labels))-ynew)'; %10x5000
z2 = Theta1*X';
a2 = sigmoid(z2);  %25x5000
a2 = [ones(1,m); a2]; %26x5000
z3 = Theta2*a2;
a3 = sigmoid(z3); %10x5000
J = (1/m)*sum(sum((-ynew'.*log(a3)) - ((onnes).*(log(ones(num_labels,m)-a3))))) + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

for i=1:m
  a1B = X(i,:); %1x401
  z2B = Theta1*a1B'; %25x1
  a2B = sigmoid(z2B); %25x1
  a2B = [1;a2B]; %26x1
  z3B = Theta2*a2B; %10x1
  a3B = sigmoid(z3B); %10x1
  d3 = a3B - ynew(i,:)'; %10x1
  sGrad = sigmoidGradient(z2B); %25x1
  sGrad = [1; sGrad]; %26x1
  d2 = Theta2'*d3 .* sGrad; %26x1
  dTot3 = dTot3 + d3*a2B'; %10x26
  dTot2 = dTot2 + d2(2:end)*a1B; %25x401
endfor

Theta1_grad = [(1/m)*dTot2(:,1), ((1/m)*dTot2(:,2:end)+(lambda/m)*Theta1(:,2:end))]; %25x401
Theta2_grad = [(1/m)*dTot3(:,1), ((1/m)*dTot3(:,2:end)+(lambda/m)*Theta2(:,2:end))]; %10x26
  
  
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
