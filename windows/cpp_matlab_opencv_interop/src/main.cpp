// Josh Holloway
// This is a program linking together OpenCV and MATLAB
//=============================================================================
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "engine.h"  // MATLAB Engine Header File required for building in Visual Studio 
#include "mex.h"
#include "helper.h"
#include <array>
using std::array;
//=============================================================================
//=============================================================================
class matlabClass
{
private:
	Engine * ep;				// Pointer to a MATLAB Engine
	mxArray *mx_Arr;  // To store the image data inside MATLAB
public:
	matlabClass() // Default constructor
	{
		// Start the MATLAB engine
		ep = engOpen(NULL);

		// Reset MATLAB Environment
		engEvalString(ep, "clc, clear, close all;");
	}
	~matlabClass() // Destructor
	{
		//free(ep);
		engEvalString(ep, "close all;");
	}
	void command(const char* str)
	{
		engEvalString(ep, str);
	}

	void linearize(const cv::Mat& mat_in, double* arr_out, const size_t M, const size_t N)
	{ // Swap from row-major to col-major
		for (int row = 0; row < M; row++)
			for (int col = 0; col < N; col++)
				arr_out[col * M + row] = static_cast<double>(mat_in.at<unsigned char>(row, col));
	}

	template <typename T>
	T* transposeLin(const T* arrIn, const int M, const int N)
	{
		T* arrOut = (T*)malloc(sizeof(T) * M * N);
		for (int i = 0; i < M; ++i)
			for (int j = 0; j < N; ++j)
				arrOut[i*N + j] = arrIn[j*N + i];
		return arrOut;
	}

	void passImageIntoMatlab(const cv::Mat& img)
	{
		// Convert the Mat object into a double array
		double* linImgArrDouble = (double*)malloc(sizeof(double) * img.rows * img.cols);
		linearize(img, linImgArrDouble, img.rows, img.cols);

		// Copy image data into an mxArray inside C++ environment
		mx_Arr = mxCreateDoubleMatrix(img.rows, img.cols, mxREAL);
		memcpy(mxGetPr(mx_Arr), linImgArrDouble, img.rows * img.cols * sizeof(double));

		/// C++ -> MATLAB
		// Put variable into MATLAB workstpace
		engPutVariable(ep, "img_from_OpenCV", mx_Arr);
		engEvalString(ep, "figure, imshow(img_from_OpenCV, [],'Border','tight');");
	}
	void getAudioFromMatlab()
	{
		// Read in audio file and play sound:
		engEvalString(ep, "[y,Fs] = audioread('handel.wav');");
		engEvalString(ep, "sound(y,Fs);");
	}
	void returnScalarFromMatlab()
	{
		command("matlabVal=42;");

		// Grab value from workspace
		mxArray *cppValmxArray = engGetVariable(ep, "matlabVal");															// Pointer to MATLAB variable 
		const double* cppValDblPtr = static_cast<double*>(mxGetData(cppValmxArray));	// Pointer to C variable
		std::cout << "Value passed from MATLAB into C++ = " << *cppValDblPtr << std::endl << std::endl;
	}
	void returnVectorFromMatlab()
	{
		// Create a row-vector in MATLAB
		command("matlabVal=[1,2];");

		mxArray *cppValmxArray = engGetVariable(ep, "matlabVal");															// Pointer to MATLAB variable 
		const double* cppValDblPtr = static_cast<double*>(mxGetData(cppValmxArray));	// Pointer to C variable
		std::cout << "Vector passed from MATLAB into C++ = " << cppValDblPtr[0] << " " << cppValDblPtr[1] << std::endl << std::endl;
	}
	void returnMatrixFromMatlab()
	{
		// Careate a mat in MATLAB
		const int numRows = 2, numCols = 2;
		command("matlabVal=[1,2;3,4];");

		mxArray *cppValmxArray = engGetVariable(ep, "matlabVal");															// Pointer to MATLAB variable 
		const double* cppValDblPtr = static_cast<double*>(mxGetData(cppValmxArray));	// Pointer to C variable

		const double* cppValDblPtrTran = transposeLin(cppValDblPtr, numRows, numCols);
		std::cout << "Vector passed from MATLAB into C++ = " <<
			cppValDblPtrTran[0] << " " << cppValDblPtrTran[1] << " " <<
			cppValDblPtrTran[2] << " " << cppValDblPtrTran[3] << std::endl << std::endl;

		//mxArray *cppValMxArray = engGetVariable(ep, "matlabVal");
		//mxArray* output = mxCreateNumericMatrix(mxGetM(matlabVal), mxGetN())
	}
};
//=============================================================================
void do_main()
{
	//// Instantiat matlab engine object
	//matlabClass matlabObj;

	//// Change to active directory 
	//matlabObj.command("cd ");
	//// Read and display image:																																																																
	//// Clear command window
	//matlabObj.command("clc, clear, close all;");

	//// Read image into environment as a cv::Mat object
	////cv::Mat img = cv::imread("C:/Users/josh/source/repos/OpenCV_MATLAB_engine/OpenCV_MATLAB_engine/OpenCV_MATLAB_engine/img.png", CV_LOAD_IMAGE_GRAYSCALE);
	////cv::Mat img(512,512, CV_32FC1);
	//cv::Mat img = cv::imread("img.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//cv::imshow("OpenCV", img);

	///// C++ -> MATLAB
	//// Send image data to MATLAB workspace
	//matlabObj.passImageIntoMatlab(img);

	///// MATLAB -> C++
	//// Read audio data from MATLAB into C++ workspace
	////matlabObj.getAudioFromMatlab();

	///// MATLAB -> C++
	//// Demonstrate how to pass a value from MATLAB into C++ workspace
	//matlabObj.returnScalarFromMatlab();

	///// MATLAB -> C++
	//// Demonstrate how to pass a vector from MATLAB into C++ workspace
	//matlabObj.returnVectorFromMatlab();

	///// MATLAB -> C++
	//// Demonstrate how to pass a mat from MATLAB into C++ workspace
	//matlabObj.returnMatrixFromMatlab();

	//// Close program and exit open windows
	//cv::waitKey(0); // Wait on key-press from user








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
	FeatureMap<float> X(1, 4, 4);   X.count();
	cout << "\nX: " << X.channels << "x" << X.rows << "x" << X.cols << " = \n";


	// Weights:
	Filter<float> W1(2, 1, 3, 3);		W1.ones();
	Matrix<float> W3(4, 8);					W3.ones();
	Matrix<float> Wo(4, 4);					Wo.ones();

	// Layer 1: Conv + ReLu
	FeatureMap<float> Z1 = conv(X, W1);
	FeatureMap<float> A1 = relu(Z1);

	// Layer 2: Max-pool + vec
	FeatureMap<float> Z2 = max_pool(A1);
	Matrix<float> A2(8, 1);
	vec(A2, Z2);

	// Layer 3: FC + ReLu
	Matrix<float> Z3 = mult(W3, A2);
	Matrix<float> A3 = relu(Z3);

	// Layer 4: FC + Soft-max
	Matrix<float> Zo = mult(Wo, A3);
	Matrix<float> Ao = softmax(Zo);


	FeatureMap<float> Z1_valid = conv_valid(X, W1);
	Z1_valid.print();
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