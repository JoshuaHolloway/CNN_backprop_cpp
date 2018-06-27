N0 = 1;
N1 = 2;

x = [0 1 2 3;
    4 5 6 7;
    8 9 10 11;
    12 13 14 15];

W1 = ones(3,3,N1)
W2 = ones(4,8);
Wo = ones(4,4);

% Replaced custom conv with built-in conv
%y1 = Conv(x, W1)              % Convolution,  20x20x20
Z1 = Conv(x, W1);
A1 = ReLU(Z1);

% Frobenius norm
if size(A1,1) ~= size(data_from_cpp,1) ... 
        | size(A1,2) ~= size(data_from_cpp,2) ... 
        | size(A1,3) ~= size(data_from_cpp,3)
    disp('error')
end

% Compute Frobenius-norm
sum = 0;
for i = 1:size(A1,1)
    for j = 1:size(A1,2)
        for k = 1:size(A1,3)
            sum = sum + (A1(i,j,k) - data_from_cpp(i,j,k))^2;
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