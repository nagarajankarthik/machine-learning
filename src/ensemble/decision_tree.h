/*
 * Header file for decision tree class
 */

#include <string>

using namespace std;

namespace ml
{

	/**
	 * Implement decision tree using recursive partitioning algorithm.
	 */

	class DecisionTree
	{
	public:
		/**
		 * Criterion to quantify individual node impurities.
		 */
		string impurity_method = "gini";
	};
}
