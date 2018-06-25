#pragma once
//#include <iterator>
//using std::iterator;
//#pragma warning(disable:4996)

template <typename T>
class Matrix
{
public:
	const size_t rows;
	const size_t cols;
private:
	size_t length{ 0 };
	T* data{ nullptr };

public:
	//using iterator = T * ;

	// (1)
	// default constructor
	Matrix() = default;

	// (2)
	// custom constructor(s)
	Matrix(const size_t rows, const size_t cols)
		: rows{ rows }, cols{ cols }, length{ rows * cols }, data{ new T[rows*cols] } {
		for (int i = 0; i != length; ++i) data[i] = 0;
	}

	// (3)
	// copy constructor
	Matrix(const Matrix& mat)
		: data{ new T[mat.length] },
		rows{ mat.rows },
		cols{ mat.cols },
		length{ mat.length }
	{
		for (int i = 0; i != length; ++i) data[i] = mat.data[i];
	}

	// (4)
	// move constructor

	// (5)
	// copy assignment

	// (6)
	// move assignment

	// (7)
	// Destructor
	~Matrix()
	{
		delete[] data;
		data = nullptr;
	}

	void set(size_t m, size_t n, T val) { data[m * cols + n] = val; }
	T at(size_t m, size_t n) const { return data[m * cols + n]; }

	T& operator[](int index) { return data[index]; }
	size_t size() const { return length; }

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

	void print()
	{
		for (int i = 0; i != rows; ++i)
		{
			for (int j = 0; j != cols; ++j)
			{
				cout << data[i * cols + j ] << " ";
			}
			cout << "\n";
		}
	}
};