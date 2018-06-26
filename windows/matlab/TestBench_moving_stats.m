function [img_in_MATLAB, PSNR] = TestBench_moving_stats(img_from_OpenCV)

  % Read in image into MATLAB and cast to double grayscale:
  img_in_MATLAB = double(rgb2gray(imread('image1.png')));
  
%   % Assign value to variable in specified workspace
%   size_of_img_in_MATLAB
  
  % Compare OpenCV to MATLAB using PSNR quality metric
  PSNR = psnr(img_in_MATLAB,img_from_OpenCV)
end