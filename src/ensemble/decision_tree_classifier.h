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

	class DecisionTreeClassifier
	{
	public:
		/**
		 * Criterion to quantify individual node impurities.
		 * Allowed values are 'gini' and 'entropy'.
		 */
		string impurity_method = "gini";

		/**
		 * Path to data File.
		 */
		string dataFile = "";

		/**
		 * Proportion of data to be be used for training.
		 * Must be a fraction between 0 and 1 inclusive.
		 * The rest is set aside for testing.
		 */
		double trainRatio = "";

		/**
		 * Features for training data
		 */
		vector<vector<double>> trainFeatures{};

		/**
		 * Target values for training data.
		 * Use int data type because this class is
		 * for classification.
		 */
		vector<int> trainTarget{};

		/**
		 * Features for test data
		 */
		vector<vector<double>> testFeatures{};

		/**
		 * Target values for test data
		 */

		vector<int> testTarget{};
	};
}
