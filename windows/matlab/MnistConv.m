function [W1, W3, W4] = MnistConv(W1, W3, W4, X, D)
%
%

alpha = 0.01;
beta  = 0.95;

momentum1 = zeros(size(W1));
momentum5 = zeros(size(W3));
momentumo = zeros(size(W4));

N = length(D);

bsize = 100;  
blist = 1:bsize:(N-bsize+1);

% One epoch loop
%
for batch = 1:length(blist)
  dW1 = zeros(size(W1));
  dW3 = zeros(size(W3));
  dW4 = zeros(size(W4));
  
  % Mini-batch loop
  %
  begin = blist(batch);
  for k = begin:begin+bsize-1
    % Forward pass = inference
    %
    
    x  = X(:, :, k);                    % Input,           28x28
    Z1 = Conv(x, W1);                   % Convolution,  20x20x20
    A1 = ReLU(Z1);                      %
    Z2 = Pool(A1);                  % Pooling,      10x10x20
    A2 = reshape(Z2, [], 1);       %
    Z3 = W3*A2;                        % ReLU,             2000
    A3 = ReLU(Z3);                      %
    Z4  = W4*A3;                        % Softmax,          10x1
    A4  = Softmax(Z4);                  %

    % One-hot encoding
    d = zeros(10, 1);
    d(sub2ind(size(d), D(k), 1)) = 1;

    % Backpropagation

    % Layer 4
    dZ_4  = d - A4;

    % Layer 3
    dA_3     = W4' * dZ_4;             % Hidden(ReLU) layer
    g_prime_3 = (A3 > 0);
    dZ_3 = g_prime_3 .* dA_3;

    % Layer 2
    dA_2_vector   = W3' * dZ_3;                % Pooling layer
    dA_2_vector_tensor     = reshape(dA_2_vector, size(Z2));

    % Layer 1
    dA_1 = zeros(size(A1));           
    temp = ones(size(A1)) / (2*2);
    for c = 1:20
      dA_1(:, :, c) = kron(dA_2_vector_tensor(:, :, c), ones([2 2])) .* temp(:, :, c);
    end
    
    g_prime_1 = (A1 > 0);
    dZ_1 = g_prime_1 .* dA_1;          % ReLU layer 
    
    % Accumulate gradients
    for c = 1:20
        x_slice = x(:, :);
        dZ_1_rotated = rot90(dZ_1(:, :, c), 2);
        
        dW1(:, :, c) = dW1(:, :, c) + conv2(x_slice, dZ_1_rotated, 'valid');
    end % loop over channels
    dW3 = dW3 + dZ_3 * A2';    
    dW4 = dW4 + dZ_4 * A3';
  end % loop over examples in batch
  
  % Update weights
  dW1 = dW1 / bsize;
  dW3 = dW3 / bsize;
  dW4 = dW4 / bsize;
  
  
  %% TODO - turn back on mumentum!!!!
  
%   momentum1 = alpha*dW1 + beta*momentum1;
%   W1        = W1 + momentum1;
  W1        = W1 + alpha*dW1;
  
%   momentum5 = alpha*dW3 + beta*momentum5;
%   W3        = W3 + momentum5;
	W3        = W3 + dW3;
   
%   momentumo = alpha*dW4 + beta*momentumo;
%   W4        = W4 + momentumo;  
    W4 = W4 + dW4;

end % loop over batches

end % Function definition
