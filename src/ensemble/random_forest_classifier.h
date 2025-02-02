/*
 * header file for random forest class
 */
#include "../utils/json.hpp"
#include <vector>

using namespace std;

namespace ml
{

	class RandomForestClassifier
	{
	public:
		// Number of trees
		int numberTrees;

		void fit();
	};
}
