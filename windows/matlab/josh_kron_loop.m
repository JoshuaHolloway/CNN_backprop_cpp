clc, clear, close all;


e3 = zeros(4, 4, 2);
for channel = 1:2
    for i = 1:4
        for j = 1:4
            x(i,j, channel) = (i - 1) * 4 + j - 1
        end
    end
end
e3

dA_1 = zeros(4,4,2);           
temp = ones(4,4,2) / (2*2);

% Change this loop to the number of channels of first filter
for c = 1:2
   e3_slice = e3(:, :, c);
   kronek = kron(e3_slice, ones([2 2]))
  dA_1(:, :, c) = kronek .* temp(:, :, c)
end