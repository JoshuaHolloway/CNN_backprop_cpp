// Josh Holloway
//=============================================================================
#include "matlabClass.h"
#include "helper.h"
#include <array>
using std::array;
using framework::Tensor;
//=============================================================================
//=============================================================================
string ExePath() {
	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	const auto pos = string(buffer).find_last_of("\\/");
	return string(buffer).substr(0, pos);
}
//=============================================================================
void do_main()
{
	// Instantiate matlab engine object
	matlabClass matlabObj;

	// Change to active directory 
	matlabObj.command("desktop");
	string current_path = "cd " + ExePath();
	matlabObj.command(current_path); // Move to current directory of generated .exe
	matlabObj.command("cd ../../../matlab"); // Move to location of .m files

	// Reset matlab environment
	matlabObj.command("clc, clear, close all;");

	// Read im MNIST from MATLAB here
	matlabObj.command("load_MNIST_josh()");
	//matlabObj.command("Images = reshape(Images, 28, 28, [])");
	//matlabObj.command("img = Images(:,:,1)"); // Store 1st MNIST image as matrix
	//matlabObj.command("rows = size(img, 1)"); 
	//matlabObj.command("cols = size(img, 2)"); 



	// MATLAB:
	static constexpr size_t M = 1;
	static constexpr size_t R[3] = { 28, 26, 13 }; // TEMP! Change to 9x9
	static constexpr size_t C[3] = { 28, 26, 13 }; // TEMP! Change to 9x9
	static constexpr size_t neurons[6] = { 1,  20, 20, 3380, 100, 10 }; // TEMP! Change to 2000!
	static constexpr size_t K = 3; // TEMP! Change to 9x9
	//                              N0  N1  N2   N3   N4

	// x:  28 x 28 x 1													R  x C x	N[0]      1    x  R[0] x  C[0] x N[0]
	// W1: 9 x 9 x 20														K  x K x N0         N[1] x  N[0] x  K    x K
	// W3: 100 x 2000														N4 x N3             1    x  1    x  N[4] x N[3]
	// W4: 10 x 100                             N5 x N4             1    x  1    x  N[5] x N[4]
	// d:  10 x 1     with 1 in  7 position




	//// Weights:
	Tensor<double> W1(neurons[1], neurons[0], K, K);		W1.ones();
	Tensor<double> W3(1, 1, neurons[4], neurons[3]);					W3.ones();
	Tensor<double> W4(1, 1, neurons[5], neurons[4]);					W4.ones(); // Two outpus

	static constexpr size_t batches = 1;
	static constexpr size_t examples = 8000;
	for (int batch = 0; batch != batches; ++batch)
	{
		// Initialize gradients to zero (done in constructor)
		Tensor<double> dW1(W1.filters, W1.channels, W1.rows, W1.cols);	dW1.zeros();
		Tensor<double> dW3(W3.filters, W3.channels, W3.rows, W3.cols);	dW3.zeros();
		Tensor<double> dW4(W4.filters, W4.channels, W4.rows, W4.cols);	dW4.zeros();

		for (int example = 0; example != examples; ++example)
		{
			// Read single image example and corresponding label
			string image_string = "img = Images(:, : ," + std::to_string(example + 1) + ")";
			matlabObj.command(image_string);
			cout << image_string << "\n";

			image_string = "label = Labels(" + std::to_string(example + 1) + ")";
			matlabObj.command(image_string);
			cout << image_string << "\n";

			// Copy single image into x
			cv::Mat x = matlabObj.return_matrix_as_cvMat_from_matlab("img");

			// Copy single label into y
			cv::Mat y = matlabObj.return_matrix_as_cvMat_from_matlab("label");

			// Copy opencv image into raw pointer
			double* image = new double[x.rows * x.cols];
			for (int i = 0; i != x.rows; ++i)
				for (int j = 0; j != x.cols; ++j)
					image[i * x.cols + j] = x.at<double>(i, j);
			//display_image(image, 28, 28, false);

			Tensor<double> X(1, neurons[0], R[0], C[0], image);

			auto Z1_valid = conv_valid(X, W1);
			auto A1 = relu(Z1_valid);
			auto Z2 = ave_pool(A1);
			auto A2 = Z2.transpose().vectorize(); // Transpose each channel, then vectorize
			auto Z3 = mult_2D(W3, A2);
			auto A3 = relu(Z3);
			auto Z4 = mult_2D(W4, A3);
			auto A4 = softmax(Z4);

			// Instantiate 10d-column vector and initialize with zeros, then simply place 1 in label index
			Tensor<double> d(1, 1, neurons[5], 1); d.zeros(); // redundant
			d.set(0, 0, (int)y.at<double>(0, 0), 0, 1); // label is stored in (0,0)th element of cv::Mat y.
			
			cout << " example = " << example << " and label = " << (int)y.at<double>(0, 0) << "\n";
			cout << "d:\n";
			d.print();
			//display_image(image, 28, 28, false);

			auto dZ_4 = d.sub(A4);


			// dA_3 = W4' * dZ_4;             % Hidden(ReLU) layer
			auto W4T = W4.transpose();
			auto dA_3 = mult_2D(W4T, dZ_4);

			// g_prime_3 = (A3 > 0);
			auto g_prime_3 = g_prime(A3);

			// dZ_3 = g_prime_3.*dA_3;
			auto dZ_3 = hadamard(g_prime_3, dA_3);

			//	dA_2 = W3' * dZ_3;       
			auto dA_2_vector = mult_2D(W3.transpose(), dZ_3);
			//cout << "dA_2:\n";
			//dA_2.print();

			//	dA_2_tensor = reshape(dA_2, size(Z2)); // De-Vec => Matricize

			auto dA_2_tensor = dA_2_vector.tensorize_3D(Z2.channels, Z2.rows, Z2.cols);
			//cout << "dA_2_tensor:\n";
			//dA_2_tensor.print();

			//dA_1 = zeros(size(A1));
			Tensor<double> dA_1(A1.filters, A1.channels, A1.rows, A1.cols);
			dA_1.zeros();

			// De-convolution
			//for c = 1:20
			//	kronek = kron(dA_2_tensor(:, : , c), ones([2 2]));
			//	dA_1(:, : , c) = kronek.*temp(:, : , c);
			//end
			de_conv(dA_2_tensor, dA_1);

			//g_prime_1 = (A1 > 0);
			auto g_prime_1 = g_prime(A1);

			//dZ_1 = g_prime_1.*dA_1;
			auto dZ_1 = hadamard(g_prime_1, dA_1);
			
			//delta1_x = zeros(size(W1));       % Convolutional layer
			//for c = 1:20
			//		x_slice = x(:, :);
			//		dZ_1_rotated = rot90(dZ_1(:, : , c), 2);

			//		delta1_x(:, : , c) = conv2(x_slice, dZ_1_rotated, 'valid');
			//end

	
			Tensor<double> delta1_x(W1.filters, W1.channels, W1.rows, W1.cols);

			// delta1_x is 2 x 1 x 3 x 3

			// Accumulate one channel of dW1
			Tensor<double> dZ_1_slice(1, 1, dZ_1.rows, dZ_1.cols);
			for (int channel = 19; channel < 20; ++channel) // What is up with the channels? This fails if more than a few iterations
			{
				// TODO - Implement 180deg rotation of dZ_1
				// dZ_1 is 1 x 2 x 4 x 4

				// Do 2D conv between x_slice and one channel of dZ_1
				dZ_1_slice.zeros();
				for (int row = 0; row != dZ_1.rows; ++row) 		// Copy over channel-slice of dZ_1
					for (int col = 0; col != dZ_1.cols; ++col)
						dZ_1_slice.set(0, 0, row, col, dZ_1.at(0, channel, row, col));
				
				// Pass matrix (row x col) into method to accumulate single channel
				dW1.accumulate_channel(conv_valid(X, dZ_1_slice), channel);
			} // End loop over channels for dW1

			// Accumulate gradients:
			//dW1 = dW1 + delta1_x;
			//dW3 = dW3 + dZ_3 * A2';    
			//dW4 = dW4 + dZ_4 * A3';
			//dW1.accumulate(delta1_x);
			dW3.accumulate(mult_2D(dZ_3, A2.transpose()));
			dW4.accumulate(mult_2D(dZ_4, A3.transpose()));

			// NOTE: only pass in 3D data => don't pass in dW1, it is (20 x 1 x K x K)
			cout << "Sending data to MATLAB...\n";
			matlabObj.command("example_index = " + std::to_string(example + 1) );
			matlabObj.tensor_2_matlab(dW4);
			display_image(image, 28, 28, false);


			delete[] image;
		} // end loop over examples in one batch
		
		// TODO:
		// TODO:
		// TODO:
		// Apply momentum

		// Did not test below!
		// Did not test below!
		// Did not test below!
		// Did not test below!
		// Did not test below!
		// Did not test below!
		// Did not test below!
		// Did not test below!
		// Did not test below!
		// Did not test below!
		// Did not test below!
		// Did not test below!

		// Update params
		double alpha = 0.1;
		dW1.scale(alpha); // Apply learning-rate alpha
		dW3.scale(alpha); // Apply learning-rate alpha
		dW4.scale(alpha); // Apply learning-rate alpha
		W1.accumulate(dW1);
		W3.accumulate(dW3);
		W4.accumulate(dW4);

	} // end loop over batches

	// Run the script with the synthetic data
	//matlabObj.command("	x = [0 1 2 3;	4 5 6 7; 8 9 10 11;	12 13 14 15] ");
	//matlabObj.command("W1 = ones(3, 3, 2)");
	//matlabObj.command("fm_out = Conv(x, W1)");
	getchar();
}
//---------------------------------------------------------------------
int main(int argc, char* argv[])
{
	try
	{
		do_main();
		return 0;
	}
	catch (std::exception const& err)
	{
		cout << err.what() << "\n";
		getchar();
		return -1;
	}
}
//---------------------------------------------------------------------