function [W1, W2, W3] = MnistConv(W1, W2, W3, X, D)
%
%

alpha = 0.01;
beta  = 0.95;

momentum1 = zeros(size(W1));
momentum5 = zeros(size(W2));
momentumo = zeros(size(W3));

N = length(D);

bsize = 100;  
blist = 1:bsize:(N-bsize+1);

% One epoch loop
%
for batch = 1:length(blist)
  dW1 = zeros(size(W1));
  dW2 = zeros(size(W2));
  dW3 = zeros(size(W3));
  
  % Mini-batch loop
  %
  begin = blist(batch);
  for k = begin:begin+bsize-1
    % Forward pass = inference
    %
    x  = X(:, :, k);               % Input,           28x28
    Z1 = Conv(x, W1);              % Convolution,  20x20x20
    A1 = ReLU(Z1);                 %
    pooled = Pool(A1);                 % Pooling,      10x10x20
    vec = reshape(pooled, [], 1);       %
    Z2 = W2*vec;                    % ReLU,             2000
    A2 = ReLU(Z2);                 %
    Z3  = W3*A2;                    % Softmax,          10x1
    A3  = Softmax(Z3);               %

    % One-hot encoding
    %
    d = zeros(10, 1);
    d(sub2ind(size(d), D(k), 1)) = 1;

    % Backpropagation
    %
    e      = d - A3;                   % Output layer  
    delta  = e;

    e5     = W3' * delta;             % Hidden(ReLU) layer
    delta5 = (A2 > 0) .* e5;

    e4     = W2' * delta5;            % Pooling layer
    
    e3     = reshape(e4, size(pooled));

    e2 = zeros(size(A1));           
    temp = ones(size(A1)) / (2*2);
    for c = 1:20
      e2(:, :, c) = kron(e3(:, :, c), ones([2 2])) .* temp(:, :, c);
    end
    
    delta2 = (A1 > 0) .* e2;          % ReLU layer
  
    delta1_x = zeros(size(W1));       % Convolutional layer
    for c = 1:20
      delta1_x(:, :, c) = conv2(x(:, :), rot90(delta2(:, :, c), 2), 'valid');
    end
    
    dW1 = dW1 + delta1_x; 
    dW2 = dW2 + delta5*vec';    
    dW3 = dW3 + delta *A2';
  end 
  
  % Update weights
  %
  dW1 = dW1 / bsize;
  dW2 = dW2 / bsize;
  dW3 = dW3 / bsize;
  
  momentum1 = alpha*dW1 + beta*momentum1;
  W1        = W1 + momentum1;
  
  momentum5 = alpha*dW2 + beta*momentum5;
  W2        = W2 + momentum5;
   
  momentumo = alpha*dW3 + beta*momentumo;
  W3        = W3 + momentumo;  
end

end
