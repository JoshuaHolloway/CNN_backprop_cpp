#pragma once
#include <iostream>

namespace framework
{
	void print();

	class test
	{
		test()
			: var{1} 
		{}
	public:
		int var{ 0 };
	};
}