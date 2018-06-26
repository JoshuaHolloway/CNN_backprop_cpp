
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
[fm_out] = Conv_josh(x, W1)
[fm_out] = Conv(x, W1)







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