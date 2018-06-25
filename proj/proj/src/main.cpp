#include "framework.h"
#include "helper.h"
#include <array>
using std::array;
//-------------------------------------------------------------------
void do_main()
{

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
	FeatureMap<float> X(1, 4, 4);   X.count();
	cout << "\nX: " << X.channels << "x" << X.rows << "x" << X.cols << " = \n";
	X.print();

	// Weights:
	Filter<float> W1(2, 1, 3, 3);		W1.ones();
	Matrix<float> W3(4, 8);					W3.ones();
	Matrix<float> Wo(4, 4);					Wo.ones();

	// Layer 1: Conv + ReLu
	FeatureMap<float> Z1 = conv(X, W1);
	FeatureMap<float> A1 = relu(Z1);

	// Layer 2: Max-pool + vec
	FeatureMap<float> Z2 = max_pool(A1);
	Matrix<float> A2(8,1);
	vec(A2, Z2);

	// Layer 3: FC + ReLu
	Matrix<float> Z3 = mult(W3, A2);
	Matrix<float> A3 = relu(Z3);

	// Layer 4: FC + Soft-max
	Matrix<float> Zo = mult(Wo, A3);
	Matrix<float> Ao = softmax(Zo);
	
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
		std::printf("%s\n", err.what());
		getchar();
		return -1;
	}
}
//---------------------------------------------------------------------