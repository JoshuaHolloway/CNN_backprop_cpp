// Josh Holloway
//=============================================================================
#include "matlabClass.h"
#include "helper.h"
#include <array>
using std::array;
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

	// Read and display image:																																						
	// Clear command window
	matlabObj.command("clc, clear, close all;");


	// TODO - Read im MNIST from MATLAB here
	// TODO - Read im MNIST from MATLAB here
	// TODO - Read im MNIST from MATLAB here
	// TODO - Read im MNIST from MATLAB here
	// TODO - Read im MNIST from MATLAB here
	// TODO - Read im MNIST from MATLAB here
	// TODO - Read im MNIST from MATLAB here
	// TODO - Read im MNIST from MATLAB here


	// CNN stuffs
	using framework::FeatureMap;
	using framework::Tensor;
	using framework::Matrix;

	static constexpr size_t examples = 1;
	static constexpr size_t row_features = 6, col_features = 6;
	static constexpr size_t features = row_features * col_features; // Number of features
	static constexpr size_t layers = 5; // Number of layers in network
	static constexpr array<size_t, layers> neurons = { 1, 2, 8, 4, 4 }; // Number of neurons in each layer (input has features+bias)


	// Layer: 0          1            2       
	//      image     conv+relu      max            vec       fc+relu 
	//     1x28x28 -> 20x28x28 -> 20x14x14 ->  -> 
	//                20x9x9         2x2      

	// DEBUG:
	// Layer: 0     1       2               3         4
	//  image   conv+relu  max     vec    fc+relu  fc+softmax
	//      1x4x4 -> 2x4x4 -> 2x2x2 -> 8x1 ->  4x1  ->  4x1
	//          2x1x3x3    2x2             4x8      4x4

	// Syntethic image
	Tensor<double> X(1, 1, row_features, col_features);   X.count();

	// Step 1: Read in MNIST in MATLAB stored in X (28 x 28 x 8000)
	// Step 2.a: Compute Z1 in MATLAB
	// Step 2.b: Compute Z1 in C++
	// Step 3: Send Z1 from C++ into MATLAB
	//	c++ -> MATLAB
	// Step 4: Compute L2-norm in MATLAB

	//// Weights:
	Tensor<double> W1(2, 1, 3, 3);		W1.ones();
	Tensor<double> W3(1, 1, neurons[3], neurons[2]);					W3.ones();
	Tensor<double> W4(1, 1, neurons[4], neurons[3]);					W4.ones(); // Two outpus

	const size_t batches = 1;
	for (int batch = 0; batch != batches; ++batch)
	{
		// Initialize gradients to zero (done in constructor)
		Tensor<double> dW1(W1.filters, W1.channels, W1.rows, W1.cols);	dW1.zeros();
		Tensor<double> dW3(W3.filters, W3.channels, W3.rows, W3.cols);	dW3.zeros();
		Tensor<double> dW4(W4.filters, W4.channels, W4.rows, W4.cols);	dW4.zeros();

		for (int example = 0; example != examples; ++example)
		{
			auto Z1_valid = conv_valid(X, W1);
			auto A1 = relu(Z1_valid);
			auto Z2 = ave_pool(A1);
			auto A2 = Z2.transpose().vectorize(); // Transpose each channel, then vectorize
			auto Z3 = mult_2D(W3, A2);
			auto A3 = relu(Z3);
			auto Z4 = mult_2D(W4, A3);
			auto A4 = softmax(Z4);

			// DEBUG:
			A4.set(0, 0, 0, 0, 0);
			A4.set(0, 0, 1, 0, 0.75);
			A4.set(0, 0, 2, 0, 0);
			A4.set(0, 0, 3, 0, 0);

			A4.print();

			// Training examples:
			//Y = [1, 2] out of neurons[4] = 4 outputs: {1,2,3,4}

			// One hot first example
			//d = [1; 0; 0; 0]

			Tensor<double> d(1, 1, neurons[4], 1);

			d.set(0, 0, 0, 0, 1);
			d.set(0, 0, 1, 0, 0);
			d.set(0, 0, 2, 0, 0);
			d.set(0, 0, 3, 0, 0);

			auto dZ_4 = d.sub(A4);


			// dA_3 = W4' * dZ_4;             % Hidden(ReLU) layer
			auto W4T = W4.transpose();


			auto dA_3 = mult_2D(W4T, dZ_4);


			// g_prime_3 = (A3 > 0);
			auto g_prime_3 = g_prime(A3);

			// dZ_3 = g_prime_3.*dA_3;
			auto dZ_3 = hadamard(g_prime_3, dA_3);

			//	dA_2 = W3' * dZ_3;       
			auto dA_2 = mult_2D(W3.transpose(), dZ_3);
			//cout << "dA_2:\n";
			//dA_2.print();

			//	e3 = reshape(dA_2, size(Z2)); // De-Vec => Matricize

			auto e3 = dA_2.tensorize_3D(Z2.channels, Z2.rows, Z2.cols);
			//cout << "e3:\n";
			//e3.print();

			//dA_1 = zeros(size(A1));
			Tensor<double> dA_1(A1.filters, A1.channels, A1.rows, A1.cols);
			dA_1.zeros();

			// De-convolution
			//for c = 1:20
			//	kronek = kron(e3(:, : , c), ones([2 2]));
			//	dA_1(:, : , c) = kronek.*temp(:, : , c);
			//end
			de_conv(e3, dA_1);

			//g_prime_1 = (A1 > 0);
			auto g_prime_1 = g_prime(A1);

			//dZ_1 = g_prime_1.*dA_1;
			auto dZ_1 = hadamard(g_prime_1, dA_1);
			dZ_1.print_dims();


			//delta1_x = zeros(size(W1));       % Convolutional layer
			//for c = 1:20
			//		x_slice = x(:, :);
			//		dZ_1_rotated = rot90(dZ_1(:, : , c), 2);

			//		delta1_x(:, : , c) = conv2(x_slice, dZ_1_rotated, 'valid');
			//end
			Tensor<double> delta1_x(W1.filters, W1.channels, W1.rows, W1.cols);

			// delta1_x is 2 x 1 x 3 x 3

			for (int channel = 0; channel < 2; ++channel) // TODO -change the number of channels in the 
			{
				// Remember this is just one slice of the full X with all M examples;
				//auto x_slice = X;

				// TODO - Implement 180deg rotation of dZ_1
				// dZ_1 is 1 x 2 x 4 x 4

				// Do 2D conv between x_slice and one channel of dZ_1

				// Copy over channel-slice of dZ_1
				Tensor<double> dZ_1_slice(1, 1, dZ_1.rows, dZ_1.cols);
				for (int row = 0; row != dZ_1.rows; ++row)
					for (int col = 0; col != dZ_1.cols; ++col)
						dZ_1_slice.set(0, 0, row, col, dZ_1.at(0, channel, row, col));

				auto conv_temp_valid = conv_valid(X, dZ_1_slice);


				cout << "X:\n";
				X.print();

				cout << "dZ_1_slice:\n";
				dZ_1_slice.print();

				cout << "conv_temp_valid:\n";
				conv_temp_valid.print();

				// Copy over slice into delta1_x
				for (int row = 0; row != delta1_x.rows; ++row)
					for (int col = 0; col != delta1_x.cols; ++col)
						delta1_x.set(0, channel, row, col, conv_temp_valid.at(0, 0, row, col));
			} // End loop over channels for delta1_x

			// DEBUG:
			cout << "delta1_x:\n";
			delta1_x.print();


			// Accumulate gradients:
			//dW1 = dW1 + delta1_x;
			//dW3 = dW3 + dZ_3 * A2';    
			//dW4 = dW4 + dZ_4 * A3';
			dW1.accumulate(delta1_x);

			// DEBUG:
			cout << "inner dimensions should agree:\n";
			dZ_3.print_dims();


			dW3.accumulate(mult_2D(dZ_3, A2.transpose()));
			//dW4.accumulate(mult_2D(dZ_4, A3.transpose()));


			cout << "dW1:\n";
			dW1.print();


		} // end loop over examples in one batch
		
		// TODO:
		// Apply momentum

		// TODO:
		// Update params

	



		//matlabObj.tensor_2_matlab(e3);

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