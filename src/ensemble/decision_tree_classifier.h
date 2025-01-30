/*
 * Header file for decision tree class
 */

#include <string>
#include <vector>
#include "../utils/json.hpp"
#include "../utils/logging.cpp"

using namespace std;

namespace ml
{

	/**
	 * Implement decision tree using recursive partitioning algorithm.
	 */

	class DecisionTreeClassifier
	{
	public:
		/**
		 * Pointer to an instance of Logger.
		 */
		Logger * logger ;

		/**
		 * Criterion to quantify individual node impurities.
		 * Allowed values are 'gini' and 'entropy'.
		 */
		string impurity_method = "gini";

		/**
		 * Constructor
		 */
		DecisionTreeClassifier(nlohmann::json model_parameters, Logger * _logger);

		/**
		 * Train a decision tree using the recursive partitioning algorithm.
		 */
		void fit(const vector<vector<double>> & features, const vector<vector<int>> & outputs);



	};
}
