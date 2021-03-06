clc

neurons = [1, 20, 20, 2000, 100, 10];

Images = loadMNISTImages('t10k-images.idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10

%X = Images(:, :, 8001:10000);
% D = Labels(8001:10000); // MOD!!!! Using from first label

% N = 6
% x = zeros(N,N);
% for i = 1:N
%     for j = 1:N
%         x(i,j) = (i - 1) * N + j - 1
%     end
% end



W1 = ones(3, 3, 20);
%W3 = ones(100, 2000);
W3 = ones(100, 3380);
W4 = ones(10, 100);

% TODO - add loop over batches
epochs = 1;
examples = 8000;
%for epoch = 1:epochs
% DEBUG: One epoch
epoch = 1;
    
    % Initialize Gradient with zeros for l-epoch
    dW1 = zeros(size(W1));
    dW3 = zeros(size(W3));
    dW4 = zeros(size(W4));
    %% DEBUG 
    %% DEBUG 
    %% DEBUG  - change the first index 
    %% DEBUG  - change the first index 
    %% DEBUG  - change the first index 
    %for example = example_index:examples 
    % DEBUG: Run over only one example:
    example = example_index
    

        k = example;
        x = Images(:,:,k);
 
        Z1 = Conv(x, W1);
        A1 = ReLU(Z1);
        Z2 = Pool(A1);                
        A2 = reshape(Z2, [], 1);      
        Z3 = W3 * A2;
        A3 = ReLU(Z3);
        Z4 = W4*A3; 
        A4 = Softmax(Z4); % Predictions

        % One-hot encoding
        d = zeros(10, 1);
        d(sub2ind(size(d), Labels(k) + 1, 1)) = 1; % Changed to have matlab 1-based indexing
  

        % % Cross entropy: dZ2 = D - Y
        dZ_4  = d - A4;
        dA_3  = W4' * dZ_4;
        g_prime_3 = (A3 > 0);
        dZ_3 = g_prime_3 .* dA_3;

        dA_2     = W3' * dZ_3;            % Pooling layer
        e3     = reshape(dA_2, size(Z2));

        dA_1 = zeros(size(A1));           
        temp = ones(size(A1)) / (2*2);

        for c = 1:20
           e3_slice = e3(:, :, c);
           kronek = kron(e3_slice, ones([2 2]));
           hadamard_temp = kronek .* temp(:, :, c);

          dA_1(:, :, c) = hadamard_temp;
        end

        g_prime_1 = (A1 > 0);
        dZ_1 = g_prime_1 .* dA_1;

        delta1_x = zeros(size(W1));
        for c = 1:20
            x_slice = x(:, :);
            %dZ_1_rotated = rot90(dZ_1(:, :, c), 2);
            dZ_1_not_rotated = dZ_1(:, :, c);
            
            %delta1_x(:, :, c) = conv2(x_slice, dZ_1_rotated, 'valid');
            delta1_x(:, :, c) = conv2(x_slice, dZ_1_not_rotated, 'valid');
        end
        delta1_x;

        % Accumulate gradients
        dW1 = dW1 + delta1_x;
        dW3 = dW3 + dZ_3 * A2';         
        dW4 = dW4 + dZ_4 * A3';
        
        % Test cpp with golden reference here in matlab
        % Don't use with dW1
        figure(1), imshow(x);
        label = Labels(k)
        [error] = froben(dW4, data_from_cpp);
    
    % DEBUG: Remove this comment
    %end % end loop over examples

% DEBUG: Remove this comment 
%end % end loop over batches          %
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