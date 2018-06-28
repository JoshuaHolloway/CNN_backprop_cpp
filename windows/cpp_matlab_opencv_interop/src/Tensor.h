#pragma once
#include <random>
#include <chrono>
#include <iostream>
using std::cout;
namespace framework
{
	template <typename T>
	class Tensor
	{
	public:

		// (1)
		// default constructor
		Tensor() = default;

		// (2)
		// custom cunstructor
		Tensor(size_t filters, size_t channels, size_t rows, size_t cols)
			: filters{ filters }, channels{ channels },
			rows{ rows }, cols{ cols },
			length{ rows * cols * channels * filters },
			data{ new T[length] }
		{
			dim1 = filters;
			dim2 = channels;
			dim3 = rows;
			dim4 = cols;
			for (int i = 0; i != length; ++i)
				data[i] = 0;
		}

		// (3)
		// copy constructor
		Tensor(const Tensor& filt)
			: data{ new T[filt.length] },
			filters{ filt.filters },
			channels{ filt.channels },
			rows{ filt.rows },
			cols{ filt.cols },
			length{ filt.length }
		{
			dim1 = filters;
			dim2 = channels;
			dim3 = rows;
			dim4 = cols;
			for (int i = 0; i != length; ++i)
				data[i] = filt.data[i];
		}

		// TODO
		// (4)
		// move constructor

		// TODO
		// (5)
		// copy assignment
		Tensor& operator=(const Tensor &rhs)
		{
			// Create temporary dynamic array to copy into
			T* temp = new T[a.length];

			// Copy elements from rhs into temporary array
			for (int i = 0; i != rhs.length; ++i)
				temp[i] = rhs.data[i];

			// De-allocate data from rhs
			delete[] data;

			// Copy new data and attributes owned into members owned by lhs
			data = temp;
			length = rhs.length;
			return *this;
		}

		// TODO
		// (6)
		// move assignment


		// (7)
		// destructor
		~Tensor()
		{
			delete[] data;
			data = nullptr;
		}

		// Easier indexing is achieved with (output_channel, input_channel, row, col)
		inline void set(size_t i, size_t j, size_t k, size_t l, T val)
		{
			// i: dim1 - filters
			// j: dim2 - channels
			// k: dim3 - rows
			// l: dim4 - cols
			data[(i * dim4 * dim3 * dim2) + (j * dim4 * dim3) + (k * dim4) + l] = val;
		}

		inline T at(size_t i, size_t j, size_t k, size_t l) //(filters, channel, row, col)
		{
			return data[(i * dim4 * dim3 * dim2) + (j * dim4 * dim3) + (k * dim4) + l];
		}

		inline size_t lin_idx(size_t i, size_t j, size_t k, size_t l) //(filters, channel, row, col)
		{
			return (i * dim4 * dim3 * dim2) + (j * dim4 * dim3) + (k * dim4) + l;
		}

		void print()
		{
			for (int i = 0; i < filters; ++i)
			{
				for (int j = 0; j < channels; ++j)
				{
					cout << "\n ----------------- \n";
					cout << "Volume: " << i << ",  Slice: " << j << " \n";
					for (int k = 0; k < rows; ++k)
					{
						for (int l = 0; l < cols; ++l)
							cout << data[(i * dim4 * dim3 * dim2) + (j * dim4 * dim3) + (k * dim4) + l] << " ";
						cout << "\n";
					}
					cout << "\n";
				}
			}
		}

		void print_dims()
		{
			cout << "dim1 = " << dim1 << "\n";
			cout << "dim2 = " << dim2 << "\n";
			cout << "dim3 = " << dim3 << "\n";
			cout << "dim4 = " << dim4 << "\n";
		}

		void ones()
		{
			for (int i = 0; i != length; ++i)
				data[i] = static_cast<T>(1);
		}

		void ones_flipped()
		{
			for (int i = 0; i != length; ++i)
				data[i] = static_cast<T>(pow(-1, i));
		}

		void zeros()
		{
			for (int i = 0; i != length; ++i)
				data[i] = static_cast<T>(0);
		}

		void count()
		{
			for (int i = 0; i != length; ++i)
				data[i] = static_cast<T>(i);
		}

		// Direct Linear-indexing access:
		T& operator[](int i)
		{
			return data[i];
		}
		const T& operator[](int i) const
		{
			return data[i];
		}

		Tensor<T> sub(Tensor<T> rhs)
		{
			assert(filters == rhs.filters);
			assert(channels == rhs.channels);
			assert(rows == rhs.rows);
			assert(cols == rhs.cols);

			Tensor<T> out(rhs.dim1, rhs.dim2, rhs.dim3, rhs.dim4);
			for (int i = 0; i != rhs.dim1; ++i)
				for (int j = 0; j != rhs.dim2; ++j)
					for (int k = 0; k != rhs.dim3; ++k)
						for (int l = 0; l != rhs.dim4; ++l)
							out.set(i, j, k, l, at(i, j, k, l) - rhs.at(i, j, k, l));
			return out;
		}


		// mutator
		void accumulate(Tensor<T> rhs)
		{
			//assert(lhs.filters == rhs.filters);
			//assert(lhs.channels == rhs.channels);
			//assert(lhs.rows == rhs.rows);
			//assert(lhs.cols == rhs.cols);
			for (int i = 0; i != rhs.dim1; ++i)
				for (int j = 0; j != rhs.dim2; ++j)
					for (int k = 0; k != rhs.dim3; ++k)
						for (int l = 0; l != rhs.dim4; ++l)
							set(i, j, k, l, at(i, j, k, l) + rhs.at(i, j, k, l));
		}


		// non-mutator
		Tensor<T> transpose()
		{
			Tensor<T> transposed(dim1, dim2, dim4, dim3); // Tranpose (dim3 x dim4) sized matrices => output is (dim4 x dim3) sized matrices

			// Transpose each channel
			for (int i = 0; i < dim1; ++i) // channels
				for (int j = 0; j < dim2; ++j) // rows
					for (int k = 0; k < dim3; ++k) // cols
						for (int l = 0; l < dim4; ++l)
							transposed.set(i, j, l, k, at(i, j, k, l));			
			return transposed;
		}

		// non-mutator
		Tensor<T> vectorize()
		{
			Tensor<T> out(1, 1, dim1 * dim2 * dim3 * dim4, 1); // Single-channel column-vector
			for (int i = 0; i != length; ++i)
				out.set(0, 0, i, 0, data[i]);
			return out;
		}

		// non-multator
		Tensor<T> matricize(size_t dim3, size_t dim4)
		{
			// Current tensor is dimensions: 1 x 1 x (dim3 x dim4) x 1
			// New tensor is dimension: 1 x 1 x dim3 x dim4
			Tensor<T> tensor_out(1, 1, dim3, dim4);
			for (int k = 0; k < dim3; ++k) // Width of Feature Map
				for (int l = 0; l < dim4; ++l)
					tensor_out.set(0, 0, k, l, at(0, 0, k, l));
			return tensor_out;
		}

		// non-multator
		Tensor<T> tensorize_3D(size_t dim2, size_t dim3, size_t dim4)
		{
			// Current tensor is dimensions: 1 x 1 x (dim2 x dim3 x dim4) x 1
			// New tensor is dimension: 1 x dim2 x dim3 x dim4
			Tensor<T> tensor_out(1, dim2, dim3, dim4);
			size_t count = 0;
			for (int j = 0; j < dim2; ++j) // Width of Feature Map
				for (int k = 0; k < dim3; ++k) // Width of Feature Map
					for (int l = 0; l < dim4; ++l)
					{
						//cout << "at(0," << j << ", " << k << ", " << l << ") = " << at(0, j, k, l) << "   data[" << count << "] = " << data[count] << "\n";
						tensor_out.set(0, j, k, l, data[count]);
						count++;
					}
						
			return tensor_out;
		}

		// mutator
		void scale(T scale_factor)
		{
			for (int i = 0; i != length; ++i)
				data[i] *= scale_factor;
		}


		void init()
		{
			// TODO - change to xavier initialization

			std::random_device seed;
			std::mt19937 generator(seed());
			size_t mean = (T)0;
			size_t variance = (T)1;

			// Instantiate object of normal_distribution class
			std::normal_distribution<T> distribution(mean, variance);

			// Sample from normal distribution
			for (int i = 0; i != length; ++m)
				data[i] = static_cast<T>(distribution(generator));
		}

		// Extra dimension beyond that of Tensor
	public:
		size_t filters{ 0 };
		size_t channels{ 0 };
		size_t rows{ 0 };
		size_t cols{ 0 };
	private:
		size_t dim1{ 0 };
		size_t dim2{ 0 };
		size_t dim3{ 0 };
		size_t dim4{ 0 };

		size_t length{ 0 };
		T* data{ nullptr };
	};
}