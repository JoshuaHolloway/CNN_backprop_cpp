#pragma once
#include "Tensor.h"
#include <random>
#include <chrono>
#include <iostream>
using std::cout;

template <typename T>
class Filter
{
public:

	// (1)
	// default constructor
	Filter() = default;

	// (2)
	// custom cunstructor
	Filter(size_t filters, size_t channels, size_t rows, size_t cols)
		: filters{ filters }, channels{ channels }, 
			rows{ rows }, cols{ cols },
			length{ rows * cols * channels * filters },
			data { new T[length] }
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
	Filter(const Filter& filt)
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
	Filter& operator=(const Filter &rhs)
	{
	}

	// TODO
	// (6)
	// move assignment


	// (7)
	// destructor
	~Filter()
	{
		delete[] data;
		data = nullptr;
	}
	
	// Easier indexing is achieved with (output_channel, input_channel, row, col)
	void set(size_t i, size_t j, size_t k, size_t l, float val)
	{
		// i: dim1 - filters
		// j: dim2 - channels
		// k: dim3 - rows
		// l: dim4 - cols
		data[(i * dim4 * dim3 * dim2) + (j * dim4 * dim3) + (k * dim4) + l] = val;
	}

	float at(size_t i, size_t j, size_t k, size_t l) //(filters, channel, row, col)
	{
		return data[(i * dim4 * dim3 * dim2) + (j * dim4 * dim3) + (k * dim4) + l];
	}

	void print()
	{
		for (int i = 0; i < filters; ++i)
		{
			cout << "\n ----------------- \n";
			cout << "Volume " << i << " \n";
			for (int j = 0; j < channels; ++j)
			{
				cout << "\n ----------------- \n";
				cout << "Slice " << i << " \n";
				for (int k = 0; k < rows; ++k)
				{
					for (int l = 0; l < cols; ++l)
						cout << data[(i * dim4 * dim3 * dim2) + (j * dim4 * dim3) + (k * dim4) + l] << " ";
					cout << "\n";
				}
			}
		}
	}

	void ones()
	{
		for (int i = 0; i != length; ++i)
			data[i] = static_cast<T>(1);
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

	void init()
	{
		// TODO - change to xavier initialization

		std::random_device seed;
		std::mt19937 generator(seed());
		size_t mean = 0;
		size_t variance = 1;

		// Instantiate object of normal_distribution class
		std::normal_distribution<float> distribution(mean, variance);

		// Sample from normal distribution
		for (int i = 0; i != length; ++m)
				data[i] = distribution(generator);
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