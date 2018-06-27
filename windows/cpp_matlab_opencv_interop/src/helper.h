#pragma once
#include "Matrix.h"
#include "FeatureMap.h"
#include "Tensor.h"
//-----------------------------------------------------------------
// C++ stuff:
#include <iostream> 
//#include <fstream>
#include <string>
//#include <vector>
#include <cassert>
using std::cout;
using std::string;
//-----------------------------------------------------------------
// C and Windows stuff:
#include <windows.h>
#include <stdio.h>
#include <math.h>
//-----------------------------------------------------------------
// Parallel CPU stuff:
#include <xmmintrin.h>  // Need this for SSE compiler intrinsics
#include <omp.h>	// Open-MP
#include <thread>
//-----------------------------------------------------------------
//-----------------------------------------------------------------
using framework::Matrix;
using framework::FeatureMap;
using framework::Tensor;
//-----------------------------------------------------------------
template <typename T>
Matrix<T> conv(const Matrix<T>& x, const Matrix<T>& h)
{
	Matrix<T> y(x.rows, x.cols);
	for (int idy = 0; idy < x.rows; ++idy)
	{
		for (int idx = 0; idx < x.cols; ++idx)
		{
			float Pvalue = 0.0f;

			int M_start_point = idy - h.rows / 2;
			int N_start_point = idx - h.cols / 2;
			for (int i = 0; i < h.rows; ++i)
			{
				for (int j = 0; j < h.cols; ++j)
				{
					if ((M_start_point + i >= 0 && M_start_point + i < x.rows)
						&& (N_start_point + j >= 0 && N_start_point + j < x.cols))
					{
						Pvalue += x.at(M_start_point + i, N_start_point + j) * h.at(i, j);
					}
				}
			}
			y.set(idy, idx, Pvalue);
		}
	}
	return y;
}
//-----------------------------------------------------------------------------
template <typename T>
FeatureMap<T> conv(const FeatureMap<T>& x, const FeatureMap<T>& h)
{
	// 2D conv with 3D feature maps with implicit matrix slice addition
	// Input: Two 3D tensors
	// Output: One 2D Matrix
	FeatureMap<T> y(1, x.rows, x.cols);
	// for (int idq = 0; idq < h.filters; ++idq) // out_channels
	for (int idy = 0; idy < x.rows; ++idy) // out_rows
	{
		for (int idx = 0; idx < x.cols; ++idx) // out_cols
		{
			float Pvalue = 0.0f;
			for (int idz = 0; idz < x.channels; ++idz) // input_channels
			{

				int M_start_point = idy - h.rows / 2;
				int N_start_point = idx - h.cols / 2;
				for (int i = 0; i < h.rows; ++i) // filter_rows
				{
					for (int j = 0; j < h.cols; ++j) // filter_cols
					{
						if ((M_start_point + i >= 0 && M_start_point + i < x.rows)
							&& (N_start_point + j >= 0 && N_start_point + j < x.cols))
						{
							Pvalue += x.at(idz, M_start_point + i, N_start_point + j) * h.at(idz, i, j);
						}
					}
				}
				y.set(0, idy, idx, Pvalue);
			}
		}
	}
	//return collapse(y);
	return y;
}
//-------------------------------------------------------------------------
template <typename T>
FeatureMap<T> conv(FeatureMap<T>& x, Tensor<T>& h)
{
	// 'same' 2D conv with 3D feature maps with implicit matrix slice addition
	// Zero-padding is also implicit
	//
	// Input: One 3D feature map and one 4D tensor (set of filters)
	// Output: One 3D feature map
	FeatureMap<T> y(h.filters, x.rows, x.cols);
	for (int idq = 0; idq < h.filters; ++idq) // out_channels
	{
		for (int idy = 0; idy < x.rows; ++idy) // out_rows
		{
			for (int idx = 0; idx < x.cols; ++idx) // out_cols
			{
				float Pvalue = 0.0f;
				for (int idz = 0; idz < x.channels; ++idz) // input_channels
				{
					int M_start_point = idy - h.rows / 2;
					int N_start_point = idx - h.cols / 2;
					for (int i = 0; i < h.rows; ++i) // filter_rows
					{
						for (int j = 0; j < h.cols; ++j) // filter_cols
						{
							if ((M_start_point + i >= 0 && M_start_point + i < x.rows)
								&& (N_start_point + j >= 0 && N_start_point + j < x.cols))
							{
								Pvalue += x.at(idz, M_start_point + i, N_start_point + j) * h.at(idq, idz, i, j);
							}
						}
					}
					y.set(idq, idy, idx, Pvalue);
				}
			}
		}
	}
	return y;
}
//-------------------------------------------------------------------------
template <typename T>
FeatureMap<T> conv_valid(FeatureMap<T>& x, Tensor<T>& h)
{
	const size_t y_rows = x.rows - h.rows + 1;
	const size_t y_cols = x.cols - h.cols + 1;


	// 'same' 2D conv with 3D feature maps with implicit matrix slice addition
	// Zero-padding is also implicit
	//
	// Input: One 3D feature map and one 4D tensor (set of filters)
	// Output: One 3D feature map
	FeatureMap<T> y(h.filters, y_rows, y_cols);
	for (int idq = 0; idq < h.filters; ++idq) // out_channels
	{
		for (int idy = h.rows/2; idy < x.rows - h.rows/2; ++idy) // out_rows
		{
			for (int idx = h.cols/2; idx < x.cols - h.cols/2; ++idx) // out_cols
			{
				float Pvalue = 0.0f;
				for (int idz = 0; idz < x.channels; ++idz) // input_channels
				{
					const size_t M_start_point = idy - h.rows / 2;
					const size_t N_start_point = idx - h.cols / 2;
					for (int i = 0; i < h.rows; ++i) // filter_rows
					{
						for (int j = 0; j < h.cols; ++j) // filter_cols
						{
							if ((M_start_point + i >= 0 && M_start_point + i < x.rows)
								&& (N_start_point + j >= 0 && N_start_point + j < x.cols))
							{
								Pvalue += x.at(idz, M_start_point + i, N_start_point + j) * h.at(idq, idz, i, j);
							}
						}
					}
					y.set(idq, idy - 1, idx - 1, Pvalue);
				}
			}
		}
	}

	return y;
}
//-------------------------------------------------------------------------
template <typename T>
Tensor<T> conv_valid(Tensor<T>& x, Tensor<T>& h)
{
	const size_t y_rows = x.rows - h.rows + 1;
	const size_t y_cols = x.cols - h.cols + 1;


	// 'same' 2D conv with 3D feature maps with implicit matrix slice addition
	// Zero-padding is also implicit
	//
	// Input: One 4D tensor and one 4D tensor
	// Output: One 4D feature map with everything stored in the 2nd, 3rd, and 4th channels
	//FeatureMap<T> y(h.filters, y_rows, y_cols);
	Tensor<T> y(1, h.filters, y_rows, y_cols);

	for (int idq = 0; idq < h.filters; ++idq) // out_channels
	{
		for (int idy = h.rows / 2; idy < x.rows - h.rows / 2; ++idy) // out_rows
		{
			for (int idx = h.cols / 2; idx < x.cols - h.cols / 2; ++idx) // out_cols
			{
				float Pvalue = 0.0f;
				for (int idz = 0; idz < x.channels; ++idz) // input_channels
				{
					const size_t M_start_point = idy - h.rows / 2;
					const size_t N_start_point = idx - h.cols / 2;
					for (int i = 0; i < h.rows; ++i) // filter_rows
					{
						for (int j = 0; j < h.cols; ++j) // filter_cols
						{
							if ((M_start_point + i >= 0 && M_start_point + i < x.rows)
								&& (N_start_point + j >= 0 && N_start_point + j < x.cols))
							{
								Pvalue += x.at(0, idz, M_start_point + i, N_start_point + j) * h.at(idq, idz, i, j);
							}
						}
					}
					y.set(0, idq, idy - 1, idx - 1, Pvalue);
				}
			}
		}
	}

	return y;
}
//-------------------------------------------------------------------------
template <typename T>
FeatureMap<T> sigmoid(const FeatureMap<T>& Z)
{
	FeatureMap<T> A(Z.channels, Z.rows, Z.cols);

	for (size_t l = 0; l != Z.channels; ++l)
		for (size_t m = 0; m != Z.rows; ++m)
			for (size_t n = 0; n != Z.cols; ++n)
				A.set(l, m, n, 1.0f / (1 + exp(-Z.at(l, m, n))));
	return A;
}
//-------------------------------------------------------------------------
template <typename T>
Tensor<T> sigmoid(const Tensor <T>& Z)
{
	Tensor<T> A(1, Z.channels, Z.rows, Z.cols);

	// All feature-maps only use the 2nd, 3rd, and 4th dimensions of the tensor
	for (size_t l = 0; l != Z.channels; ++l)	// dim-1
		for (size_t m = 0; m != Z.rows; ++m)		// dim-2
			for (size_t n = 0; n != Z.cols; ++n)	// dim-3
				A.set(l, m, n, 1.0f / (1 + exp(-Z.at(l, m, n))));
	return A;
}
//-------------------------------------------------------------------------
template <typename T>
Tensor<T> relu(Tensor<T> Z)
{
	Tensor<T> A(1, Z.channels, Z.rows, Z.cols);
	for (size_t l = 0; l != Z.channels; ++l)
		for (size_t m = 0; m != Z.rows; ++m)
			for (size_t n = 0; n != Z.cols; ++n)
			{
				T z = Z.at(0, l, m, n);
				T max_val = max(0, z);
				A.set(0, l, m, n, max_val);
			}
	return A;
}
//-------------------------------------------------------
template <typename T>
FeatureMap<T> max_pool(FeatureMap<T> x)
{
	// downsampling factor:
	const size_t K = 2;

	// output downsampled feature map:
	FeatureMap<T> y(x.channels, x.rows / K, x.cols / K); // rows, cols, channels
	for (int i = 0; i < x.channels; ++i)
	{
		for (int j = 0; j < x.rows; j += K)
		{
			for (int k = 0; k < x.cols; k += K)
			{
				// Search inside the KxK block for max value
				T max = 0;
				for (int jj = j; jj < j + K; ++jj)
				{
					for (int kk = k; kk < k + K; ++kk)
					{
						if (jj == j && kk == k)
						{
							max = x.at(i, jj, kk);
						}
						else
						{
							if (x.at(i, jj, kk) > max)
								max = x.at(i, jj, kk) + 1;
						} // end if-else
					}// end for over kk
				} // end for over jj
					//y[i][j / 2][k / 2] = max;
				y.set(i, j / 2, k / 2, max);
			} // end for over k
		} // end for over j
	} // end for over i
	return y;
}
//-------------------------------------------------------
template <typename T>
FeatureMap<T> ave_pool(FeatureMap<T> x)
{
	const size_t H = x.rows;
	const size_t W = x.cols;
	const size_t M = x.channels;
	const size_t K = 2; // downsampling factor

	FeatureMap<T> S(x.channels, H / K, W / K); // rows, cols, channels

	for (int m = 0; m < M; ++m)  // channels
	{
		for (int h = 0; h < H / K; ++h) // rows
		{
			for (int w = 0; w < W / K; ++w)
			{
				auto temp = (T)0;
				for (int p = 0; p < K; ++p)
				{
					for (int q = 0; q < K; ++q)
					{
						temp += x.at(m, K*h + p, K*w + q);
					}
				}
				S.set(m, h, w, temp / T(K*K));
			}
		}
	}

	return S;
}
//-------------------------------------------------------
template <typename T>
Tensor<T> ave_pool(Tensor<T> x)
{
	const size_t H = x.rows;
	const size_t W = x.cols;
	const size_t M = x.channels;
	const size_t K = 2; // downsampling factor

	Tensor<T> S(1, x.channels, H / K, W / K); // rows, cols, channels

	for (int m = 0; m < M; ++m)  // channels
	{
		for (int h = 0; h < H / K; ++h) // rows
		{
			for (int w = 0; w < W / K; ++w)
			{
				auto temp = (T)0;
				for (int p = 0; p < K; ++p)
				{
					for (int q = 0; q < K; ++q)
					{
						temp += x.at(0, m, K*h + p, K*w + q);
					}
				}
				S.set(0, m, h, w, temp / T(K*K));
			}
		}
	}

	return S;
}
//---------------------------------------------------
template <typename T>
Tensor<T> mult_2D(Tensor<T> A, Tensor<T> B)
{
	// Input: 
	// B is 1 x N x 1  column-vector
	// A is 1 x M x N  matrix
	// Output:
	// C is 1 x M x 1  column-vector

	assert(A.cols == B.rows);
	const size_t M = A.rows;
	const size_t N = A.cols;

	Tensor<T> C(1, 1, M, 1);
	for (size_t n = 0; n < B.cols; ++n) // For vector B, this will run only once
		for (size_t m = 0; m < A.rows; ++m)
		{
			auto sum = (T)0;
			for (size_t k = 0; k < A.cols; ++k)
			{
				sum += A.at(0, 0, m, k) * B.at(0, 0, k, n);
				cout << "sum = " << sum << " ";
			}
				
			C.set(0, 0, m, n, sum);
		}
	return C;
}
//---------------------------------------------------
template <typename T>
Matrix<T> softmax(const Matrix<T>& Z)
{
	//function y = Softmax(x)
	//	ex = exp(x);
	//	y = ex / sum(ex);
	//end

	// Find maximum value
	T max_val = static_cast<T>(0);
	for (size_t m = 0; m != Z.rows; ++m)
		for (size_t n = 0; n != Z.cols; ++n)
		{
			if (Z.at(m, n) > max_val)
				max_val = Z.at(m, n);
		}

	// reduce
	T sum = static_cast<T>(0);
	for (size_t m = 0; m != Z.rows; ++m)
		for (size_t n = 0; n != Z.cols; ++n)
			sum += exp(Z.at(m,n) - max_val);

	// softmax
	Matrix<T> A(Z.rows, Z.cols);
	for (size_t m = 0; m != Z.rows; ++m)
		for (size_t n = 0; n != Z.cols; ++n)
			A.set(m, n, exp(Z.at(m, n) - max_val) / sum);
			
	return A;
}
//---------------------------------------------------
template <typename T>
FeatureMap<T> softmax(FeatureMap<T> Z)
{
	//function y = Softmax(x)
	//	ex = exp(x);
	//	y = ex / sum(ex);
	//end

	// Find maximum value
	auto max_val = static_cast<T>(0);
	for (size_t m = 0; m != Z.rows; ++m)
		for (size_t n = 0; n != Z.cols; ++n)
		{
			auto temp = Z.at(0, m, n);
			if (temp > max_val)
				max_val = temp;
		}

	// reduce
	T sum = static_cast<T>(0);
	for (size_t m = 0; m != Z.rows; ++m)
		for (size_t n = 0; n != Z.cols; ++n)
			sum += exp(Z.at(0, m, n) - max_val);

	// softmax
	FeatureMap<T> A(1, Z.rows, Z.cols);
	for (size_t m = 0; m != Z.rows; ++m)
		for (size_t n = 0; n != Z.cols; ++n)
			A.set(0, m, n, exp(Z.at(0, m, n) - max_val) / sum);

	return A;
}
//---------------------------------------------------
template <typename T>
Tensor<T> softmax(Tensor<T> Z)
{
	//function y = Softmax(x)
	//	ex = exp(x);
	//	y = ex / sum(ex);
	//end

	// Find maximum value
	auto max_val = static_cast<T>(0);
	for (size_t m = 0; m != Z.rows; ++m)
		for (size_t n = 0; n != Z.cols; ++n)
		{
			auto temp = Z.at(0, 0, m, n);
			if (temp > max_val)
				max_val = temp;
		}

	// reduce
	T sum = static_cast<T>(0);
	for (size_t m = 0; m != Z.rows; ++m)
		for (size_t n = 0; n != Z.cols; ++n)
			sum += exp(Z.at(0, 0, m, n) - max_val);

	// softmax
	Tensor<T> A(1, 1, Z.rows, Z.cols);
	for (size_t m = 0; m != Z.rows; ++m)
		for (size_t n = 0; n != Z.cols; ++n)
			A.set(0, 0, m, n, exp(Z.at(0, 0, m, n) - max_val) / sum);

	return A;
}
//---------------------------------------------------
template <typename T>
Tensor<T> g_prime(Tensor<T> Z)
{
	// Derivative of ReLu
	Tensor<T> A(1, Z.channels, Z.rows, Z.cols);
	for (size_t l = 0; l != Z.channels; ++l)
		for (size_t m = 0; m != Z.rows; ++m)
			for (size_t n = 0; n != Z.cols; ++n)
			{
				//(condition) ? (if_true) : (if_false)
				(Z.at(0, l, m, n) > 0) ? (A.set(0, l, m, n, 1)) : (A.set(0, l, m, n, 0));
			}
	return A;
}
//---------------------------------------------------
template <typename T>
Tensor<T> hadamard(Tensor<T> A, Tensor<T> B)
{
	assert(A.filters == 1 && B.filters == 1);
	assert(A.channels == B.channels);
	assert(A.rows == B.rows);
	assert(A.cols == B.cols);
	
	Tensor<T> C(1, A.channels, A.rows, A.cols);

	for (size_t l = 0; l < A.channels; ++l)
		for (size_t m = 0; m < A.rows; ++m)
			for (size_t n = 0; n < A.cols; ++n)
				C.set(0, l, m, n, A.at(0, l, m, n) * B.at(0, l, m, n));

	return C;
}