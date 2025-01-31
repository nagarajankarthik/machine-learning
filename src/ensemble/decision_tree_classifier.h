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
	 * Class defining attributes of each node in a decision tree.
	 */
	struct TreeNode {
		TreeNode * left_child = nullptr;
		TreeNode * right_child = nullptr;
		int feature_split = -1;
		double value_split = std::numeric_limits<double>::min();
		vector<int> node_indices {};
	        int depth = -1;
		TreeNode(vector<int> _indices, int _depth): node_indices(_indices), depth(_depth) {};
	};	       

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
		 * Container of TreeNodes.
		 */
		vector<TreeNode*> _tree {};

		/**
		 * Constructor
		 */
		DecisionTreeClassifier(nlohmann::json model_parameters, Logger * _logger);

		/**
		 * Destructor
		 */
		~DecisionTreeClassifier() ;

		/**
		 * Train a decision tree using the recursive partitioning algorithm.
		 */
		void fit(const vector<vector<double>> & features, const vector<vector<int>> & outputs);



	};
			
}
