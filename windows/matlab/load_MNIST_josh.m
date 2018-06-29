Images = loadMNISTImages('t10k-images.idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10