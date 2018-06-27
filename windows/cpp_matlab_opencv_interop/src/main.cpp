// Josh Holloway
// This is a program linking together OpenCV and MATLAB
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

	cout << "dZ_3:\n";
	dZ_3.print();
	

	// dA_2 = W3' * dZ_3;            % Pooling layer
	// e3 = reshape(dA_2, size(Z2)); % De-Vec

	matlabObj.tensor_2_matlab(dZ_3);
	

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