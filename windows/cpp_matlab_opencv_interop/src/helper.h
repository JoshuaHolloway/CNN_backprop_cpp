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
	// NOTE: For odd samples use floor(K/2) for output indexing
	// NOTE: For even samples use floor(K/2)-1 for output indexing
	//   My even indexing may not generalize completely, but works as an 
	//	 ad-hoc method for the moment
	
	size_t y_rows = 0;
	size_t y_cols = 0;
	assert(h.rows < x.rows);
	assert(h.cols < x.cols);
	//if (h.rows % 2 == 0) // Even rows
	//	y_rows = x.rows - h.rows;
	//else								 // Odd rows
		y_rows = x.rows - h.rows + 1;

	//if (h.cols % 2 == 0) // Even cols
	//	y_cols = x.cols - h.cols;
	//else								 // Odd cols
		y_cols = x.cols - h.cols + 1;

	// 'same' 2D conv with 3D feature maps with implicit matrix slice addition
	// Zero-padding is also implicit
	//
	// Input: One 4D tensor and one 4D tensor
	// Output: One 4D feature map with everything stored in the 2nd, 3rd, and 4th channels
	//FeatureMap<T> y(h.filters, y_rows, y_cols);
	Tensor<T> y(1, h.filters, y_rows, y_cols);

	for (int idq = 0; idq < h.filters; ++idq) // out_channels
	{
		// TODO - take this logic out of the loop
		int idy_lim = 0;
		if (h.rows % 2 == 0) // Even rows
			idy_lim = x.rows - h.rows / 2 + 1;
		else								 // Odd rows
			idy_lim = x.rows - h.rows / 2;
		for (int idy = h.rows / 2; idy < idy_lim; ++idy) // out_rows
		{
			// TODO - take this logic out of the loop
			int idx_lim = 0;
			if (h.cols % 2 == 0) // Even cols
				idx_lim = x.rows - h.cols/ 2 + 1;
			else								 // Odd cols
				idx_lim = x.rows - h.cols / 2;
			for (int idx = h.cols / 2; idx < idx_lim; ++idx) // out_cols
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

					// TODO - take this logic out of the loop
					// Ad-hoc mod 1: If h is even then write to index idx-2,idy-2 instead of idx-1,idy-1
					// This assumes the actual convolutionis proper and it is only written out wrong
					int out_idy = 0;
					int out_idx = 0;
					if (h.rows % 2 == 0) // Even rows
						out_idy = idy - 2;
					else								 // Odd rows
						out_idy = idy - 1;

					if (h.cols % 2 == 0) // Even cols
						out_idx = idx - 2;
					else								 // Odd cols
						out_idx = idx - 1;
					
					y.set(0, idq, out_idy, out_idx, Pvalue);
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
	// A is 1 x 1 x M x N  matrix
	// B is 1 x 1 x N x P  matrix

	// Output:
	// C is 1 x M x P  matrix

	assert(A.cols == B.rows);
	const size_t M = A.rows;
	const size_t N = A.cols;
	const size_t P = B.cols;

	Tensor<T> C(1, 1, M, P);
	for (size_t n = 0; n < B.cols; ++n) // For vector B, this will run only once
		for (size_t m = 0; m < A.rows; ++m)
		{
			auto sum = (T)0;
			for (size_t k = 0; k < A.cols; ++k)
				sum += A.at(0, 0, m, k) * B.at(0, 0, k, n);
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
//---------------------------------------------------
template <typename T>
Tensor<T> kronecker(Tensor<T> A, Tensor<T> B)
{
	// This kroenecker poduct assumes that B is filled with ones
	Tensor<T> C(1, 1, A.rows * B.rows, A.cols * B.cols);

	int offset_l = 0;
	int out_r = 0;
	int out_l = 0;
	for (int i = 0; i < A.rows; ++i)
	{
		int offset_r = 0;
		for (int j = 0; j < A.cols; ++j)
		{
			for (int k = 0; k < B.rows; ++k)
			{
				for (int l = 0; l < B.cols; ++l)
					C.set(0, 0, k + offset_l, l + offset_r, B.at(0, 0, k, l));
			}
			offset_r += B.cols;
		}
		offset_l += B.rows;
	}

	return C;
}
//---------------------------------------------------
template <typename T>
void de_conv(Tensor<T> e3, Tensor<T>& dA_1)
{

	//temp = ones(size(A1)) / (2 * 2);
	Tensor<double> temp(dA_1.filters, dA_1.channels, dA_1.rows, dA_1.cols);
	temp.ones();
	temp.scale(0.25);


	//for c = 1:20
	//	kronek = kron(e3(:, : , c), ones([2 2]));
	//	dA_1(:, : , c) = kronek.*temp(:, : , c);
	//end
	for (int channel = 0; channel < 2; ++channel) // TODO - change this to number of channels in first weight tensor
	{
		// Slice of e3
		// e3(:, : , c)
		Tensor<double> e3_slice(1, 1, e3.rows, e3.cols);
		for (int row = 0; row != e3.rows; ++row)
			for (int col = 0; col != e3.cols; ++col)
				e3_slice.set(0, 0, row, col, e3.at(0, channel, row, col));




		//	kronek = kron(e3(:, : , c), ones([2 2]));
		Tensor<T> Ones(1, 1, 2, 2);	Ones.ones();
		auto kron = kronecker(e3_slice, Ones);

		//cout << "e3_slice\n";
		//e3_slice.print();

		//cout << "kron\n";
		//kron.print();


		// dA_1 is output
		// e3 = ones(0,2,2,2)
		// e3_slice is ones(2,2)
		// kron is ones(4,4)


		//	dA_1(:, : , c) = kron .* temp(:, : , c);
		for (int row = 0; row < dA_1.rows; ++row)
			for (int col = 0; col < dA_1.cols; ++col)
			{
				//cout << "kron.at(0, 0, row, col) = " << kron.at(0, 0, row, col) << "\n";
				//cout << "temp.at(0, channel, row, col)= " << temp.at(0, channel, row, col) << "\n";

				auto hadamard_temp = kron.at(0, 0, row, col) * temp.at(0, channel, row, col);
				dA_1.set(0, channel, row, col, hadamard_temp);
				//cout << "hadamard_temp = " << hadamard_temp << "\n";
				//cout << "dA_1.at(0, channel, row, col) = " << dA_1.at(0, channel, row, col) << "\n";

			}		
		//cout << "dA_1.at(0, 0, 0, 0) = " << dA_1.at(0, 0, 0, 0) << "\n";
		//dA_1.print();
	}
}
//------------------------------------------
void display_image(
	double* buffer,
	int height,
	int width,
	bool color)
{
	if (color == true)
	{
		cv::Mat output_image(height, width, CV_64FC3, buffer);
		// Make negative values zero.
		cv::threshold(output_image,
			output_image,
			/*threshold=*/0,
			/*maxval=*/0,
			cv::THRESH_TOZERO);
		cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
		output_image.convertTo(output_image, CV_8UC3);
		cv::imshow("Image from display_image()", output_image);
	}
	else
	{
		cv::Mat output_image(height, width, CV_64FC1, buffer);
		// Make negative values zero.
		cv::threshold(output_image,
			output_image,
			/*threshold=*/0,
			/*maxval=*/0,
			cv::THRESH_TOZERO);
		cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
		output_image.convertTo(output_image, CV_8UC1);
		cv::imshow("Image from display_image()", output_image);
	}
	cv::waitKey(0);
}