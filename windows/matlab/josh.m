N0 = 1;
N1 = 2;

N = 8
x = zeros(N,N);
for i = 1:N
    for j = 1:N
        x(i,j) = (i - 1) * N + j - 1
    end
end
x


W1 = ones(3,3,N1)
W2 = ones(4,8);
Wo = ones(4,4);

% Replaced custom conv with built-in conv
%y1 = Conv(x, W1)              % Convolution,  20x20x20
Z1 = Conv(x, W1);
A1 = ReLU(Z1);
Z2 = Pool(A1);                 % Pooling,      10x10x20

% Frobenius norm
if size(Z2,1) ~= size(data_from_cpp,1) ... 
        | size(Z2,2) ~= size(data_from_cpp,2) ... 
        | size(Z2,3) ~= size(data_from_cpp,3)
    disp('error')
end

% Compute Frobenius-norm
sum = 0;
for i = 1:size(Z2,1)
    for j = 1:size(Z2,2)
        for k = 1:size(Z2,3)
            sum = sum + (Z2(i,j,k) - data_from_cpp(i,j,k))^2;
        end
    end
end
error = sqrt(sum)




%y2 = ReLU(y1);                 %
%y3 = Pool(y2);                 % Pooling,      10x10x20
%y4 = reshape(y3, [], 1);       %
% v5 = W5*y4;                    % ReLU,             2000
% y5 = ReLU(v5);                 %
% v  = Wo*y5;                    % Softmax,          10x1
% y  = Softmax(v)               %


function y = Softmax(x)
  ex = exp(x);
  y  = ex / sum(ex);
end