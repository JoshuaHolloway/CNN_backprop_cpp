clc

N0 = 1;
N1 = 2;
N2 = 4;

N = 6
x = zeros(N,N);
for i = 1:N
    for j = 1:N
        x(i,j) = (i - 1) * N + j - 1
    end
end
x


W1 = ones(3,3,N1)
W3 = ones(4,8);
W4 = ones(4,4);

% Replaced custom conv with built-in conv
%y1 = Conv(x, W1)              % 'valid' convb  20x20x20
Z1 = Conv(x, W1);
A1 = ReLU(Z1);
Z2 = Pool(A1);                 % ave-pool      10x10x20
A2 = reshape(Z2, [], 1);       % vectorize (10x10x20)x1
Z3 = W3 * A2;                    % ReLU,             2000
A3 = ReLU(Z3);
Z4 = W4*A3;                    % Softmax,          10x1
A4 = Softmax(Z4); % Predictions

%% Debug:
A4 = [0; 0.75; 0; 0]

% % DEBUG: 4-examples with labels {1,2}
M = 2; % M examples
k = 2; % First example
D = [1, 2]; % Labels from MNIST

% One-hot encoding
d = zeros(N2, 1);
d(sub2ind(size(d), D(k), 1)) = 1

% % Cross entropy: dZ2 = D - Y
dZ_4  = d - A4;
dA_3  = W4' * dZ_4;
g_prime_3 = (A3 > 0);
dZ_3 = g_prime_3 .* dA_3;

% Layer 2
dA_2     = W3' * dZ_3;            % Pooling layer
e3     = reshape(dA_2, size(Z2));

% Layer 1
dA_1 = zeros(size(A1));           
temp = ones(size(A1)) / (2*2);

% Change this loop to the number of channels of first filter
for c = 1:2
   e3_slice = e3(:, :, c);
   kronek = kron(e3_slice, ones([2 2]));
   hadamard_temp = kronek .* temp(:, :, c)
   
  dA_1(:, :, c) = hadamard_temp
end

g_prime_1 = (A1 > 0);
dZ_1 = g_prime_1 .* dA_1;          % ReLU layer

delta1_x = zeros(size(W1));       % Convolutional layer
for c = 1:2
    x_slice = x(:, :);
    %dZ_1_rotated = rot90(dZ_1(:, :, c), 2);
    dZ_1_not_rotated = dZ_1(:, :, c);

    %delta1_x(:, :, c) = conv2(x_slice, dZ_1_rotated, 'valid');
    delta1_x(:, :, c) = conv2(x_slice, dZ_1_not_rotated, 'valid');
end
delta1_x



% Test cpp with golden reference here in matlab
[error] = froben(e3, data_from_cpp)










            %
% ============================================================
function [error] = froben(matlab, cpp)
    % Frobenius norm
    if size(matlab,1) ~= size(cpp,1) ... 
            | size(matlab,2) ~= size(cpp,2) ... 
            | size(matlab,3) ~= size(cpp,3)
        disp('error')
    end

    % Compute Frobenius-norm
    sum = 0;
    for i = 1:size(matlab,1)
        for j = 1:size(matlab,2)
            for k = 1:size(matlab,3)
                sum = sum + (matlab(i,j,k) - cpp(i,j,k))^2;
            end
        end
    end
    error = sqrt(sum)
end


% ============================================================
function y = Softmax(x)
  ex = exp(x- max(x));
  y  = ex / sum(ex);
end