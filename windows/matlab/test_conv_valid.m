
% Create synthetic image filled with count
N = 4
x = zeros(N,N);
for i = 1:N
    for j = 1:N
        x(i,j) = (i - 1) * N + j;
    end
end
x

% Odd filter 
h_odd = ones(3, 3);

% Even filter
h_even = ones(2, 2);

%% Do odd-valid filtering
y_same_odd = conv2(x, h_odd)
y_valid_odd = conv2(x, h_odd, 'valid')


%% Do even-valued filtering
y_same_even = conv2(x, h_even)
y_valid_even = conv2(x, h_even, 'valid')