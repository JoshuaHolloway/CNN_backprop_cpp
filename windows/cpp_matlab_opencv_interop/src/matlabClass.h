#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "engine.h"  // MATLAB Engine Header File required for building in Visual Studio 
#include "mex.h"
#include "FeatureMap.h"
using std::string;
using framework::FeatureMap;

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
	void command(string str)
	{
		engEvalString(ep, str.c_str());
	}

	template <typename T>
	void linearize(const cv::Mat& mat_in, T* arr_out, const size_t M, const size_t N)
	{ // Swap from row-major to col-major
		for (int row = 0; row < M; row++)
			for (int col = 0; col < N; col++)
				arr_out[col * M + row] = mat_in.at<T>(row, col);
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


	// 2D cv::Mat passed from C++ into MATLAB
	void passImageIntoMatlab(const cv::Mat& img)
	{
		// Convert the Mat object into a double array
		//double* linImgArrDouble = (double*)malloc(sizeof(double) * img.rows * img.cols);
		double* linImgArrDouble = new double[img.rows * img.cols];
		linearize(img, linImgArrDouble, img.rows, img.cols);

		// Copy image data into an mxArray inside C++ environment
		mx_Arr = mxCreateDoubleMatrix(img.rows, img.cols, mxREAL);
		memcpy(mxGetPr(mx_Arr), linImgArrDouble, img.rows * img.cols * sizeof(double));

		/// C++ -> MATLAB
		// Put variable into MATLAB workstpace
		engPutVariable(ep, "img_from_OpenCV", mx_Arr);
		engEvalString(ep, "figure, imshow(img_from_OpenCV, [],'Border','tight');");
		delete[] linImgArrDouble;
	}



	// 2D data stored in 1D array passed into 2D matrix in MATLAB
	void pass_2D_into_matlab(const double* data, const int M, const int N)
	{
		// Copy image data into an mxArray inside C++ environment
		mx_Arr = mxCreateDoubleMatrix(M, N, mxREAL);
		memcpy(mxGetPr(mx_Arr), data, M * N * sizeof(double));

		/// C++ -> MATLAB
		// Put variable into MATLAB workstpace
		engPutVariable(ep, "data_from_cpp", mx_Arr);
	}

	
	// 3D data stored in 1D array passed into 3D matrix in MATLAB
	void pass_3D_into_matlab(const double* data, const int dim1, const int dim2, const int dim3)
	{
		//3rd dim is (i,:,:) in C++, yet (:,:,i) in MATLAB
		//mxCreateNumericArray(mwSize ndim, const mwSize *dims,
		//	mxClassID classid, mxComplexity ComplexFlag);
		const mwSize ndim = 3;
		const mwSize dims[ndim] = { dim2, dim3, dim1 };
		mx_Arr = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);

		// Copy tensor data into an mxArray inside C++ environment
		memcpy(mxGetPr(mx_Arr), data, dim1 * dim2 * dim3 * sizeof(double));

		/// C++ -> MATLAB
		// Put variable into MATLAB workstpace
		engPutVariable(ep, "data_from_cpp", mx_Arr);
	}


	void getAudioFromMatlab()
	{
		// Read in audio file and play sound:
		engEvalString(ep, "[y,Fs] = audioread('handel.wav');");
		engEvalString(ep, "sound(y,Fs);");
	}
	void return_scalar_from_matlab(string variable)
	{
		// Grab value from workspace
		mxArray *cppValmxArray = engGetVariable(ep, variable.c_str());															// Pointer to MATLAB variable 
		const double* cppValDblPtr = static_cast<double*>(mxGetData(cppValmxArray));	// Pointer to C variable
		std::cout << variable << " = " << *cppValDblPtr << std::endl << std::endl;
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
	}

	template <typename T>
	void fm_2_matlab_tensor(FeatureMap<T> fm)
	{
		// Transpose each channels matrix and send to matlab to store from (i,:,:) -> (:,:,i)
		pass_3D_into_matlab(&fm.transpose()[0], fm.channels, fm.rows, fm.cols);

		// Execute the testbench script
		command("josh()");

		// Compute frobenius norm between MATLAB and C++
		return_scalar_from_matlab("error");
	}

	template <typename T>
	void fm_2_matlab_vector(FeatureMap<T> vect)
	{
		// Transpose each channels matrix and send to matlab to store from (i,:,:) -> (:,:,i)
		pass_2D_into_matlab(&vect[0], vect.rows, vect.cols); // column-vector => cols=1

		// Execute the testbench script
		command("josh()");

		// Compute frobenius norm between MATLAB and C++
		return_scalar_from_matlab("error");
	}
};
//=============================================================================