function [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D)
%
%

alpha = 0.01;
beta  = 0.95;

momentum1 = zeros(size(W1));
momentum5 = zeros(size(W5));
momentumo = zeros(size(Wo));

N = length(D);

bsize = 100;  
blist = 1:bsize:(N-bsize+1);

% One epoch loop
%
for batch = 1:length(blist)
  dW1 = zeros(size(W1));
  dW5 = zeros(size(W5));
  dWo = zeros(size(Wo));
  
  % Mini-batch loop
  %
  begin = blist(batch);
  for k = begin:begin+bsize-1
    % Forward pass = inference
    %
    x  = X(:, :, k);               % Input,           28x28
    y1 = Conv(x, W1);              % Convolution,  20x20x20
    
    
    y2 = ReLU(y1);                 %
    
%     size_y2 = size(y2)  % y2 is 20x20x20 'valid  => 28x28x20 'same'
    
    y3 = Pool(y2);                 % Pooling,      10x10x20
    y4 = reshape(y3, [], 1);       %
    
%     size_W5 = size(W5)  % W5 is 100 x 2000
%     size_y4 = size(y4)  % y4 is 2000 x 1 'valid  => 3920 x 1 'same'

    
    v5 = W5*y4;                    % ReLU,             2000
    y5 = ReLU(v5);                 %
    v  = Wo*y5;                    % Softmax,          10x1
    y  = Softmax(v);               %

    % One-hot encoding
    %
    d = zeros(10, 1);
    d(sub2ind(size(d), D(k), 1)) = 1;

    % Backpropagation

    % Loss = Cross-Entropy
    % Activation of final layer = Soft-max

    % Modern notation:
    % dA = d_loss / da
    % dZ = d_loss / dz
    %    = (d_loss / da) * (da / dz)
    %    =  dZ           * g_prime(Z)
    % dW = dZ * X

    
    % Logistic Regression:
    % One-layer neural-network
    % Activation: sigmoid: A = g(Z)
    % Loss: Cross-entropy: L(A,Y)
    % Cost: Sum of losses: J(W,b) = 1/M * sum(L(A,Y), 1, M)
    
    
    % e = dA_L
    % delta = dZ_L                    % Modern notation dims (TODO: Modify to change to these classical dims)
    dA2      = d - y;                % N2 x M (In modified modern notation)
    delta  = dA2;
    
    g_prime_2 = (y5 > 0)
    pause
    
    delta5 =  .* e5;
    
    
    e5     = Wo' * delta;             % Hidden(ReLU) layer


    e4     = W5' * delta5;            % Pooling layer
    
    e3     = reshape(e4, size(y3));

    e2 = zeros(size(y2));           
    W3 = ones(size(y2)) / (2*2);
    for c = 1:20
      e2(:, :, c) = kron(e3(:, :, c), ones([2 2])) .* W3(:, :, c);
    end
    
    delta2 = (y2 > 0) .* e2;          % ReLU layer
  
    delta1_x = zeros(size(W1));       % Convolutional layer
    for c = 1:20
      delta1_x(:, :, c) = conv2(x(:, :), rot90(delta2(:, :, c), 2), 'valid');
    end
    
    dW1 = dW1 + delta1_x; 
    dW5 = dW5 + delta5*y4';    
    dWo = dWo + delta *y5';
  end 
  
  % Update weights
  %
  dW1 = dW1 / bsize;
  dW5 = dW5 / bsize;
  dWo = dWo / bsize;
  
  momentum1 = alpha*dW1 + beta*momentum1;
  W1        = W1 + momentum1;
  
  momentum5 = alpha*dW5 + beta*momentum5;
  W5        = W5 + momentum5;
   
  momentumo = alpha*dWo + beta*momentumo;
  Wo        = Wo + momentumo;  
end

end

