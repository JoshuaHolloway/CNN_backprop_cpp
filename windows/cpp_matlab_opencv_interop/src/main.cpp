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
	// Instantiat matlab engine object
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
	using framework::Filter;
	using framework::Matrix;

	static constexpr size_t examples = 1e5; // Number of examples
	static constexpr size_t rows = 28, cols = 28;
	static constexpr size_t features = rows * cols; // Number of features
	static constexpr size_t layers = 3; // Number of layers in network
	static constexpr array<size_t, layers> neurons = { features + 1, 4, 1 }; // Number of neurons in each layer (input has features+bias)

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
	FeatureMap<double> X(1, 6, 6);   X.count();

	// Step 1: Read in MNIST in MATLAB stored in X (28 x 28 x 8000)
	// Step 2.a: Compute Z1 in MATLAB
	// Step 2.b: Compute Z1 in C++
	// Step 3: Send Z1 from C++ into MATLAB
	//	c++ -> MATLAB
	// Step 4: Compute L2-norm in MATLAB

	//// Weights:
	Filter<double> W1(2, 1, 3, 3);		W1.ones();
	//Matrix<float> W3(4, 8);					W3.ones();
	//Matrix<float> Wo(4, 4);					Wo.ones();

	//// Layer 1: Conv + ReLu
	//FeatureMap<float> Z1 = conv(X, W1);
	//FeatureMap<float> A1 = relu(Z1);

	//// Layer 2: Max-pool + vec
	//FeatureMap<float> Z2 = max_pool(A1);
	//Matrix<float> A2(8, 1);
	//vec(A2, Z2);

	//// Layer 3: FC + ReLu
	//Matrix<float> Z3 = mult(W3, A2);
	//Matrix<float> A3 = relu(Z3);

	//// Layer 4: FC + Soft-max
	//Matrix<float> Zo = mult(Wo, A3);
	//Matrix<float> Ao = softmax(Zo);
	FeatureMap<double> Z1_valid = conv_valid(X, W1);
	FeatureMap<double> A1 = relu(Z1_valid);
	FeatureMap<double> Z2 = ave_pool(A1);
	FeatureMap<double> A2 = Z2.transpose().vectorize(); // Transpose each channel, then vectorize
	

	//Z2.print();
	A2.print();
	

	//matlabObj.fm_2_matlab_tensor(Z2);
	matlabObj.fm_2_matlab_vector(A2);

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