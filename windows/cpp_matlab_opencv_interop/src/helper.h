#pragma once
#include "Matrix.h"
#include "FeatureMap.h"
#include "Filter.h"
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
using framework::Filter;
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
FeatureMap<T> conv(FeatureMap<T>& x, Filter<T>& h)
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
FeatureMap<T> relu(FeatureMap<T> Z)
{
	FeatureMap<T> A(Z.channels, Z.rows, Z.cols);
	for (size_t l = 0; l != Z.channels; ++l)
		for (size_t m = 0; m != Z.rows; ++m)
			for (size_t n = 0; n != Z.cols; ++n)
			{
				T z = Z.at(l, m, n);
				T max_val = max(0, z);
				A.set(l, m, n, max_val);
			}
	return A;
}
//-------------------------------------------------------------------------
template <typename T>
Matrix<T> relu(const Matrix<T>& Z)
{
	Matrix<T> A(Z.rows, Z.cols);
	for (size_t m = 0; m != Z.rows; ++m)
		for (size_t n = 0; n != Z.cols; ++n)
			A.set(m, n, max(0, Z.at(m, n)));
	return A;
}
//-------------------------------------------------------
template <typename T>
FeatureMap<T> max_pool(FeatureMap<T> x)
{
	// downsampling factor:
	const size_t K = 2;

	// output downsampled feature map:
	FeatureMap<T> y(x.rows / K, x.cols / K, x.channels); // rows, cols, channels
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
								max = x.at(i, jj, kk);
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
void vec(Matrix<T>& y, FeatureMap<T> x)
{
	//Matrix<T> y(x.rows * x.cols, 1);
	int count = 0;
	for (size_t l = 0; l != x.channels; ++l)
		for (size_t m = 0; m != x.rows; ++m)
			for (size_t n = 0; n != x.cols; ++n)
			{
				y.set(count, 0, x.at(l, m, n));
				count++;
			}
	//return y;
}
//---------------------------------------------------
template <typename T>
Matrix<T> mult(const Matrix<T>& A, const Matrix<T>& B)
{
	assert(A.cols == B.rows);
	Matrix<T> C(A.rows, B.cols);
	for (size_t n = 0; n < B.cols; ++n)
		for (size_t m = 0; m < A.rows; ++m)
		{
			float sum = 0.0f;
			for (size_t k = 0; k < A.cols; ++k)
				sum += A.at(m, k) * B.at(k, n);
			C.set(m, n, sum);
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