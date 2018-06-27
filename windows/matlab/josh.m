clc

N0 = 1;
N1 = 2;

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
A4 = Softmax(Z4);

% Test cpp with golden reference here in matlab
[error] = froben(A4, data_from_cpp)




%y2 = ReLU(y1);                 %
%y3 = Pool(y2);                 % Pooling,      10x10x20
%y4 = reshape(y3, [], 1);       %
% v5 = W5*y4;                    % ReLU,             2000
% y5 = ReLU(v5);                 %
% v  = Wo*y5;                    % Softmax,          10x1
% y  = Softmax(v)               %
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