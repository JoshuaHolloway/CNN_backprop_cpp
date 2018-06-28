#pragma once
#include <iostream>
using std::cout;
//--------------

namespace framework
{
	template<typename T>
	class FeatureMap;

	template<typename T>
	FeatureMap<T> operator-(const FeatureMap<T>& lhs, const FeatureMap<T>& rhs);

	template<typename T>
	class FeatureMap
	{
	public:

		// (1)
		// default constructor
		FeatureMap() = default;

		// (2)
		// custom constructor
		FeatureMap(size_t channels, size_t rows, size_t cols)
			: channels{ channels },
			rows{ rows }, cols{ cols },
			length{ rows * cols * channels },
			data{ new T[length] }
		{
			dim1 = channels;
			dim2 = rows;
			dim3 = cols;
			for (int i = 0; i != length; ++i)
				data[i] = 0;
		}

		// (3)
		// copy constructor
		FeatureMap(const FeatureMap& fm)
			: data{ new T[fm.length] },
			channels{ fm.channels },
			rows{ fm.rows },
			cols{ fm.cols },
			length{ fm.length }
		{
			dim1 = channels;
			dim2 = rows;
			dim3 = cols;
			for (int i = 0; i != length; ++i)
				data[i] = fm.data[i];
		}

		// TODO
		// (4)
		// move constructor
		FeatureMap(FeatureMap<T>&& fm)
			: data{fm.data},
			channels{ fm.channels },
			rows{ fm.rows },
			cols{ fm.cols },
			length{ fm.length }
		{
			dim1 = channels;
			dim2 = rows;
			dim3 = cols;

			// Free up data for input fm
			fm.data = nullptr;
			fm.dim1 = 0;	fm.channels = 0;
			fm.dim2 = 0;	fm.rows = 0;
			fm.dim3 = 0;	fm.cols = 0;
		}

		// TODO
		// (5)
		// copy assignment
		FeatureMap& operator=(const FeatureMap &rhs)
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
		~FeatureMap()
		{
			delete[] data;
			data = nullptr;
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
		
		FeatureMap<T> sub(FeatureMap<T> rhs)
		{
			assert(channels == rhs.channels);
			assert(rows == rhs.rows);
			assert(cols == rhs.cols);

			FeatureMap<T> out(rhs.dim1, rhs.dim2, rhs.dim3);
			for (int i = 0; i != rhs.channels; ++i)
				for (int j = 0; j != rhs.rows; ++j)
					for (int k = 0; k != rhs.cols; ++k)
						out.set(i, j, k, at(i,j,k) - rhs.at(i,j,k));
			return out;
		}



		// (channel, row, col)
		void set(size_t i, size_t j, size_t k, float val)
		{
			data[i * rows * cols + j * cols + k] = val;
		}
		float at(size_t i, size_t j, size_t k) //(channel, row, col)
		{
			return data[i * rows * cols + j * cols + k];
		}


		void print()
		{
			for (int i = 0; i < channels; ++i)
			{
				cout << "\n ----------------- \n";
				cout << "Slice " << i << " \n";
				for (int j = 0; j < rows; ++j)
				{
					for (int k = 0; k < cols; ++k)
					{
						cout << data[i * rows * cols + j * cols + k] << " ";
					}
					cout << "\n";
				}
			}
		}

		void print_dims()
		{
			cout << "Channels: " << channels << "\n";
			cout << "Rows: " << rows << "\n";
			cout << "Cols: " << cols << "\n";
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

		// non-mutator
		FeatureMap<T> transpose()
		{
			FeatureMap<T> transposed(dim1, dim2, dim3);

			// Transpose each channel
			for (int i = 0; i < dim1; ++i) // channels
				for (int j = 0; j < dim2; ++j) // rows
					for (int k = 0; k < dim3; ++k) // cols
						transposed.data[i * dim3 * dim2 + j * dim3 + k] = data[i * dim3 * dim2 + k * dim3 + j];
			return transposed;
		}

		// non-mutator
		FeatureMap<T> vectorize()
		{
			FeatureMap<T> out(1, dim1 * dim2 * dim3, 1); // Single-channel column-vector
			for (int i = 0; i != length; ++i)
				out.data[i] = data[i];
			return out;
		}

	public:
		size_t channels{ 0 };
		size_t rows{ 0 };
		size_t cols{ 0 };
	private:
		size_t dim1{ 0 };
		size_t dim2{ 0 };
		size_t dim3{ 0 };

		size_t length{ 0 };
		T* data{ nullptr };
	};
	//---------------

}