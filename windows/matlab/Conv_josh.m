function [fm_out] = Conv_josh(fm_in, filter)

    R = size(fm_in, 1);
    C = size(fm_in, 2);
    N0 = size(fm_in, 3);
    N1 = size(filter,3);
   
    fm_out = zeros(R, C, N1);
    for i = 1:N1 % Itterate over output channels
        fm_temp = zeros(R, C);
        for j = 1:N0 % Itterate over input channels
            W1_slice = filter(:,:,j);
            fm_slice = fm_in(:,:,j);
            fm_temp = fm_temp + conv2(fm_slice, W1_slice, 'same');
        end
        fm_out(:,:, i) = fm_temp;
    end
end